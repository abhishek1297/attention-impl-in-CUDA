#include "attention.hpp"
#include "cuda_utils.hpp"

#include <cmath>
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

namespace cg = cooperative_groups;

#define PARTIALS_PTR(arr, bh_idx, row, seq_len, blocks_x)                                          \
    ((arr) + ROW_OFFSET(bh_idx, row, seq_len, blocks_x))

#define TILE_DIM 32

// Kernel: strided and batched QK^T with partial max reductions only
__global__ void qk_dot_partial_reduce_v2(const float *__restrict__ Q, const float *__restrict__ K,
                                         float *attn_scores, float *row_max_partials, int seq_len,
                                         int head_dim, const float scale) {

    // cooperative-groups block object
    cg::thread_block cg_block = cg::this_thread_block();
    // create a warp-sized tile group
    auto warp = cg::tiled_partition<TILE_DIM>(cg_block);

    int bh_idx = blockIdx.z;
    int blocks_x = gridDim.x;
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = block_row + local_row;
    int col = block_col + local_col;

    const float *Qbh_base = Q_PTR(Q, bh_idx, seq_len, head_dim);
    const float *Kbh_base = K_PTR(K, bh_idx, seq_len, head_dim);
    float *attn_bh_base = ATTN_PTR(attn_scores, bh_idx, seq_len);

    __shared__ float Q_tile[TILE_DIM][TILE_DIM];
    __shared__ float Kt_tile[TILE_DIM][TILE_DIM + 1]; // padding

    float acc = 0.0f;

    // per-thread workload
    int num_tiles = (head_dim + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        int q_col = t * TILE_DIM + local_col;
        int k_row = t * TILE_DIM + local_row;

        float q_val = (row < seq_len && q_col < head_dim) ? Qbh_base[row * head_dim + q_col] : 0.0f;
        Q_tile[local_row][local_col] = q_val;

        // note transpose
        float kt_val =
            (col < seq_len && k_row < head_dim) ? Kbh_base[col * head_dim + k_row] : 0.0f;
        Kt_tile[local_col][local_row] = kt_val;

        cg_block.sync();

        // inner product for this tile vectorized in groups of 4
#pragma unroll
        for (int k = 0; k < TILE_DIM; k += 4) {
            float4 q_vec;
            q_vec.x = Q_tile[local_row][k + 0];
            q_vec.y = Q_tile[local_row][k + 1];
            q_vec.z = Q_tile[local_row][k + 2];
            q_vec.w = Q_tile[local_row][k + 3];

            float4 k_vec;
            k_vec.x = Kt_tile[local_col][k + 0];
            k_vec.y = Kt_tile[local_col][k + 1];
            k_vec.z = Kt_tile[local_col][k + 2];
            k_vec.w = Kt_tile[local_col][k + 3];

            acc = fmaf(q_vec.x, k_vec.x, acc);
            acc = fmaf(q_vec.y, k_vec.y, acc);
            acc = fmaf(q_vec.z, k_vec.z, acc);
            acc = fmaf(q_vec.w, k_vec.w, acc);
        }
        cg_block.sync();
    }

    acc *= scale;

    if (row < seq_len && col < seq_len) {
        attn_bh_base[row * seq_len + col] = acc;
    }

    int warps_per_row = (blockDim.x + warp.size() - 1) / warp.size();

    // per-warp max scores i.e each thread in a warp will hold the same max value
    float max_score = acc;
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        float other = warp.shfl_xor(max_score, offset);
        max_score = fmaxf(max_score, other);
    }

    if (warps_per_row > 1) {
        extern __shared__ float warp_max_scores[];
        int warp_id_row = threadIdx.y;
        int warp_id_col = threadIdx.x / warp.size();
        // thread 0 per-warp loads its max score
        if (warp.thread_rank() == 0) {
            warp_max_scores[warp_id_row * warps_per_row + warp_id_col] = max_score;
        }
        cg_block.sync();

        // block-level reduction for max scores per row of a block
        if (warp_id_col == 0) {
            int warp_col_fetch_id = warp.thread_rank();
            float val = (warp_col_fetch_id < warps_per_row)
                            ? warp_max_scores[warp_id_row * warps_per_row + warp_col_fetch_id]
                            : -CUDART_INF_F;
            for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
                float other = warp.shfl_xor(val, offset);
                max_score = fmaxf(val, other);
            }
            max_score = val;
        }
        cg_block.sync();
    }

    // write per-block row partial maxes to global memory from the first column in tile (local_col
    // == 0)
    if (local_col == 0 && row < seq_len) {
        int idx = ROW_OFFSET(bh_idx, row, seq_len, blocks_x) + blockIdx.x;
        row_max_partials[idx] = max_score;
    }
}

// Kernel: find global max and compute global sum
__global__ void softmax_inplace_v2(float *attention_scores,
                                   const float *__restrict__ row_max_partials, int seq_len,
                                   int partials_blocks_x) {

    int bh_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    if (row >= seq_len)
        return;

    const float *row_max_bh_base =
        PARTIALS_PTR(row_max_partials, bh_idx, row, seq_len, partials_blocks_x);

    // Find max (for numerical stability)
    float row_max;
    __shared__ float shared_row_max;
    if (tid == 0) {
        row_max = -CUDART_INF_F;
        for (int bx = 0; bx < partials_blocks_x; bx++) {
            float block_max = row_max_bh_base[bx];
            if (block_max > row_max)
                row_max = block_max;
        }
        shared_row_max = row_max;
    }
    // synchronize row max from thread 0 to all
    __syncthreads();
    row_max = shared_row_max;

    // Compute sum of exp(score - row_max) across all columns
    float *attn_bh_base = ATTN_PTR(attention_scores, bh_idx, seq_len);
    float row_sum = 0.0f;

    for (int col = tid; col < seq_len; col += blockDim.x) {
        int idx = row * seq_len + col;
        float score = expf(attn_bh_base[idx] - row_max);
        row_sum += score;
        attn_bh_base[idx] = score;
    }

    extern __shared__ float shared_sum[];
    shared_sum[tid] = row_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            shared_sum[tid] += shared_sum[tid + offset];
        __syncthreads();
    }

    // normalize attention scores
    const float norm = 1 / shared_sum[0];
    for (int col = tid; col < seq_len; col += blockDim.x) {
        attn_bh_base[row * seq_len + col] *= norm;
    }
}

// Kernel: apply softmax and multiply by V
__global__ void softmax_multV_v2(const float *__restrict__ attention_scores,
                                 const float *__restrict__ V, float *O, int seq_len, int head_dim) {

    int bh_idx = blockIdx.z;
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = block_row + local_row;
    int col = block_col + local_col;

    if (row >= seq_len)
        return;

    const float *attn_bh_base = ATTN_PTR(attention_scores, bh_idx, seq_len);
    const float *Vbh_base = V_PTR(V, bh_idx, seq_len, head_dim);
    float *Obh_base = O_PTR(O, bh_idx, seq_len, head_dim);

    __shared__ float softmax_tile[TILE_DIM][TILE_DIM];
    __shared__ float V_tile[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    int num_tiles = (seq_len + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int k = t * TILE_DIM + local_col;

        // Load softmax tile: softmax[row, k]
        if (row < seq_len && k < seq_len) {
            softmax_tile[local_row][local_col] = attn_bh_base[row * seq_len + k];
        } else {
            softmax_tile[local_row][local_col] = 0.0f;
        }

        int k_v = t * TILE_DIM + local_row;
        if (k_v < seq_len && col < head_dim) {
            V_tile[local_row][local_col] = Vbh_base[k_v * head_dim + col];
        } else {
            V_tile[local_row][local_col] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += softmax_tile[local_row][k] * V_tile[k][local_col];
        }
        __syncthreads();
    }

    if (row < seq_len && col < head_dim) {
        Obh_base[row * head_dim + col] = acc;
    }
}

// Simple attention implementation
struct VanillaAttentionV2 : public Attention {

    void forward(const float *Q, const float *K, const float *V, float *O, uint32_t batch_size,
                 uint32_t num_heads, uint32_t seq_len, uint32_t head_dim) override {

        float *attention_scores;
        float *row_max_partials;

        const float scale = 1.0f / sqrtf((float) head_dim);
        const size_t n_qkt = batch_size * num_heads * seq_len * seq_len;
        cudaMalloc(&attention_scores, n_qkt * sizeof(float));

        int blocks_x = (seq_len + TILE_DIM - 1) / TILE_DIM;
        int blocks_y = (seq_len + TILE_DIM - 1) / TILE_DIM;

        const size_t n_partials = batch_size * num_heads * seq_len * blocks_x;
        cudaMalloc(&row_max_partials, n_partials * sizeof(float));

        // Kernel 1: Compute QK^T and partial max values
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid(blocks_x, blocks_y, batch_size * num_heads);
        const int num_warps_in_x = TILE_DIM / 32;
        int shared_bytes = num_warps_in_x * sizeof(float);
        qk_dot_partial_reduce_v2<<<grid, threads, shared_bytes>>>(
            Q, K, attention_scores, row_max_partials, seq_len, head_dim, scale);
        cudaDeviceSynchronize();
        CUDA_CHECK();

        // save_device_ptr_as_buffer("QKt.bin", attention_scores, n_qkt);
        // Kernel 2:  Apply per row max and normalize with per row sum
        int device_max_threads;
        cudaDeviceGetAttribute(&device_max_threads, cudaDevAttrMaxThreadsPerBlock, 0);
        int per_row_threads = std::min(TILE_DIM * blocks_x, device_max_threads);
        per_row_threads = std::max(32, per_row_threads);
        dim3 threads2(per_row_threads, 1, 1);
        dim3 grid2(1, seq_len, batch_size * num_heads);
        shared_bytes = per_row_threads * sizeof(float);
        softmax_inplace_v2<<<grid2, threads2, shared_bytes>>>(attention_scores, row_max_partials,
                                                              seq_len, blocks_x);
        cudaDeviceSynchronize();
        CUDA_CHECK();

        // Kernel 3: Apply softmax and multiply by V
        softmax_multV_v2<<<grid, threads>>>(attention_scores, V, O, seq_len, head_dim);
        cudaDeviceSynchronize();
        CUDA_CHECK();

        cudaFree(attention_scores);
        cudaFree(row_max_partials);
        CUDA_CHECK();
    }
};

// Factory function
extern "C" Attention *create_vanilla_attention_v2() {
    return new VanillaAttentionV2();
}