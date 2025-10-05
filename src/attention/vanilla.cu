#include "attention.hpp"
#include "cuda_utils.hpp"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

#define PARTIALS_PTR(arr, bh_idx, row, seq_len, blocks_x)                                          \
    ((arr) + ROW_OFFSET(bh_idx, row, seq_len, blocks_x))

#define TILE_DIM 16

// Kernel: strided and batched QK^T with partial max reductions only
__global__ void qk_dot_partial_reduce(float *Q, float *K, float *attn_scores,
                                      float *row_max_partials, int seq_len, int head_dim) {

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
    __shared__ float Kt_tile[TILE_DIM][TILE_DIM + 1]; // avoid bank conflicts by padding

    float acc = 0.0f;
    const float scale = 1.0f / sqrtf((float) head_dim);

    int num_tiles = (head_dim + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int q_col = t * TILE_DIM + local_col;
        int k_row = t * TILE_DIM + local_row;

        if (row < seq_len && q_col < head_dim) {
            Q_tile[local_row][local_col] = Qbh_base[row * head_dim + q_col];
        } else {
            Q_tile[local_row][local_col] = 0.0f;
        }

        // K transposed directly in the shared memory tile
        if (col < seq_len && k_row < head_dim) {
            Kt_tile[local_col][local_row] = Kbh_base[col * head_dim + k_row];
        } else {
            Kt_tile[local_col][local_row] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += Q_tile[local_row][k] * Kt_tile[local_col][k];
        }

        __syncthreads();
    }

    // Apply scaling factor
    acc *= scale;

    // Write to global memory
    if (row < seq_len && col < seq_len) {
        attn_bh_base[row * seq_len + col] = acc;
    }

    // Compute partial max (only max, no exp yet)
    __shared__ float scores_tile[TILE_DIM][TILE_DIM];

    if (row < seq_len && col < seq_len) {
        scores_tile[local_row][local_col] = acc;
    } else {
        scores_tile[local_row][local_col] = -CUDART_INF_F;
    }
    __syncthreads();

    // Parallel reduction (max) across the tile columns
    for (int offset = TILE_DIM >> 1; offset > 0; offset >>= 1) {
        if (local_col < offset) {
            float a = scores_tile[local_row][local_col];
            float b = scores_tile[local_row][local_col + offset];
            scores_tile[local_row][local_col] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Write partial max to global memory
    if (local_col == 0 && row < seq_len) {
        int idx = ROW_OFFSET(bh_idx, row, seq_len, blocks_x) + blockIdx.x;
        row_max_partials[idx] = scores_tile[local_row][0];
    }
}

// Kernel: find global max and compute global sum
__global__ void softmax_inplace(float *attention_scores, const float *row_max_partials, int seq_len,
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
__global__ void softmax_multV(const float *attention_scores, const float *V, float *O, int seq_len,
                              int head_dim) {

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
struct VanillaAttention : public Attention {

    void forward(float *Q, float *K, float *V, float *O, int batch_size, int num_heads, int seq_len,
                 int head_dim) override {

        float *attention_scores;
        float *row_max_partials;

        const size_t n_qkt = batch_size * num_heads * seq_len * seq_len;
        cudaMalloc(&attention_scores, n_qkt * sizeof(float));

        int blocks_x = (seq_len + TILE_DIM - 1) / TILE_DIM;
        int blocks_y = (seq_len + TILE_DIM - 1) / TILE_DIM;

        const size_t n_partials = batch_size * num_heads * seq_len * blocks_x;
        cudaMalloc(&row_max_partials, n_partials * sizeof(float));

        // Kernel 1: Compute QK^T and partial max values
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid(blocks_x, blocks_y, batch_size * num_heads);
        qk_dot_partial_reduce<<<grid, threads>>>(Q, K, attention_scores, row_max_partials, seq_len,
                                                 head_dim);
        cudaDeviceSynchronize();

        // save_device_ptr_as_buffer("QKt.bin", attention_scores, n_qkt);
        // Kernel 2:  Apply per row max and normalize with per row sum
        int per_row_threads = max(TILE_DIM * blocks_x, 1024);
        dim3 threads2(per_row_threads, 1, 1);
        dim3 grid2(1, seq_len, batch_size * num_heads);
        int shared_bytes = per_row_threads * sizeof(float);
        softmax_inplace<<<grid2, threads2, shared_bytes>>>(attention_scores, row_max_partials,
                                                           seq_len, blocks_x);
        cudaDeviceSynchronize();

        // Kernel 3: Apply softmax and multiply by V
        softmax_multV<<<grid, threads>>>(attention_scores, V, O, seq_len, head_dim);
        cudaDeviceSynchronize();

        cudaFree(attention_scores);
        cudaFree(row_max_partials);
    }
};

// Factory function
extern "C" Attention *create_vanilla_attention() {
    return new VanillaAttention();
}