#include "attention.hpp"
#include "cuda_utils.hpp"

#include <cassert>
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

// Kernel: Strided and Batched Transpose K
__global__ void transpose_K(const float *__restrict__ K, float *__restrict__ K_transposed,
                            int seq_len, int head_dim) {

    int bh_idx = blockIdx.z;
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = block_row + local_row;
    int col = block_col + local_col;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    const float *Kbh_base = K_PTR(K, bh_idx, seq_len, head_dim);
    float *Kt_bh_base = K_PTR(K_transposed, bh_idx, head_dim, seq_len);

    tile[local_row][local_col] =
        (row < seq_len && col < head_dim) ? Kbh_base[row * head_dim + col] : 0.0f;
    __syncthreads();

    int trans_row = block_col + local_row; // head_dim dimension (was col)
    int trans_col = block_row + local_col; // seq_len dimension (was row)

    if (trans_row < head_dim && trans_col < seq_len) {
        Kt_bh_base[trans_row * seq_len + trans_col] = tile[local_col][local_row];
    }
}

__device__ float reduce_tile_row_maxes(cg::thread_block cg_block, float score,
                                       float *block_smem_scores) {
    // create a warp-sized tile group
    auto warp = cg::tiled_partition<32>(cg_block);
    int lane = warp.thread_rank();
    int warps_per_row = (TILE_DIM + warp.size() - 1) / warp.size();

    // per-warp max scores i.e each thread in a warp will hold the same max value
    float max_score = score;
    warp.sync();
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        float other = warp.shfl_xor(max_score, offset);
        max_score = fmaxf(max_score, other);
    }
    warp.sync();

    // each block-row has more than 1 warps
    if (warps_per_row > 1) {
        int warp_id_row = threadIdx.y;
        int warp_id_col = threadIdx.x / warp.size();
        // thread 0 per-warp loads its max scores in smem
        if (lane == 0) {
            int idx = warp_id_row * warps_per_row + warp_id_col;
            block_smem_scores[idx] = max_score;
        }
        cg_block.sync();

        // first warp of every block-row
        if (warp_id_col == 0) {
            // the same number of threads as the warps per block-row
            int idx = warp_id_row * warps_per_row + lane;
            max_score = (lane < warps_per_row) ? block_smem_scores[idx] : -CUDART_INF_F;
            warp.sync();

            for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
                float other = warp.shfl_xor(max_score, offset);
                max_score = fmaxf(max_score, other);
            }
        }
        cg_block.sync();
    }
    // at this point, column 0 of each block will hold max per block

    return max_score;
}

// Kernel: strided and batched QK^T with partial max reductions only
__global__ void qk_dot_partial_reduce_v2(const float *__restrict__ Q,
                                         const float *__restrict__ K_transposed, float *attn_scores,
                                         float *row_max_partials, int seq_len, int head_dim,
                                         const float scale) {

    // cooperative-groups block object
    cg::thread_block cg_block = cg::this_thread_block();

    int bh_idx = blockIdx.z;
    int blocks_x = gridDim.x;
    int block_row = blockIdx.y * TILE_DIM;
    int block_col = blockIdx.x * TILE_DIM;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = block_row + local_row;
    int col = block_col + local_col;

    const float *Qbh_base = Q_PTR(Q, bh_idx, seq_len, head_dim);
    const float *Ktbh_base = K_PTR(K_transposed, bh_idx, head_dim, seq_len);
    float *attn_bh_base = ATTN_PTR(attn_scores, bh_idx, seq_len);

    __shared__ alignas(float4) float Q_tile[TILE_DIM][TILE_DIM];
    __shared__ float Kt_tile[TILE_DIM][TILE_DIM + 1];

    float score = 0.0f;

    // per-thread workload
    int num_tiles = (head_dim + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        int q_col = t * TILE_DIM + local_col;
        int kt_row = t * TILE_DIM + local_row;

        float q_val = (row < seq_len && q_col < head_dim) ? Qbh_base[row * head_dim + q_col] : 0.0f;
        Q_tile[local_row][local_col] = q_val;

        // note transpose
        float kt_val =
            (col < seq_len && kt_row < head_dim) ? Ktbh_base[kt_row * seq_len + col] : 0.0f;
        Kt_tile[local_row][local_col] = kt_val;

        cg_block.sync();

        // inner product for this tile vectorized in groups of 4
        const float4 *q_vec_row = reinterpret_cast<const float4 *>(&Q_tile[local_row][0]);
#pragma unroll
        for (int k = 0; k < TILE_DIM; k += 4) {

            float4 q_vec = q_vec_row[k / 4];

            // float4 q_vec;
            // q_vec.x = Q_tile[local_row][k + 0];
            // q_vec.y = Q_tile[local_row][k + 1];
            // q_vec.z = Q_tile[local_row][k + 2];
            // q_vec.w = Q_tile[local_row][k + 3];

            float4 k_vec;
            k_vec.x = Kt_tile[k + 0][local_col];
            k_vec.y = Kt_tile[k + 1][local_col];
            k_vec.z = Kt_tile[k + 2][local_col];
            k_vec.w = Kt_tile[k + 3][local_col];

            score = fmaf(q_vec.x, k_vec.x, score);
            score = fmaf(q_vec.y, k_vec.y, score);
            score = fmaf(q_vec.z, k_vec.z, score);
            score = fmaf(q_vec.w, k_vec.w, score);
        }
        cg_block.sync();
    }

    score *= scale;
    if (row < seq_len && col < seq_len) {
        attn_bh_base[row * seq_len + col] = score;
    }
    if (local_col == 0 && row < seq_len) {
        extern __shared__ float block_smem_scores[];
        // reduced per block
        float max_score = reduce_tile_row_maxes(cg_block, score, block_smem_scores);
        // write partial maxes per-block to global memory
        int idx = ROW_OFFSET(bh_idx, row, seq_len, blocks_x) + blockIdx.x;
        row_max_partials[idx] = max_score;
    }
}

// Kernel: find global max and compute global sum
__global__ void softmax_inplace_v2(float *attention_scores,
                                   const float *__restrict__ row_max_partials, int seq_len,
                                   int partials_blocks_x) {

    // cooperative-groups block object
    cg::thread_block cg_block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cg_block);
    int lane = warp.thread_rank();

    int bh_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;

    int warp_id = tid / warp.size();
    int num_warps = (blockDim.x + warp.size() - 1) / warp.size();

    if (row >= seq_len)
        return;

    const float *row_max_bh_base =
        PARTIALS_PTR(row_max_partials, bh_idx, row, seq_len, partials_blocks_x);

    // use single shared memory array for both max and sum
    extern __shared__ float smem[];
    float *warp_row_maxes = smem;
    float *warp_row_sums = smem;

    // find max (for numerical stability)
    float max_score;
    if (lane == 0)
        warp_row_maxes[warp_id] = -CUDART_INF_F;
    cg_block.sync();

    int n_strides = (partials_blocks_x + blockDim.x - 1) / blockDim.x;
    for (int s = 0; s < n_strides; s++) {
        int bx = s * blockDim.x + warp_id * warp.size() + lane;
        max_score = (bx < partials_blocks_x) ? row_max_bh_base[bx] : -CUDART_INF_F;
        warp.sync();
        for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
            float other = warp.shfl_xor(max_score, offset);
            max_score = fmaxf(max_score, other);
        }
        // note that this will be strided max finds
        if (lane == 0)
            warp_row_maxes[warp_id] = fmaxf(max_score, warp_row_maxes[warp_id]);
    }
    cg_block.sync();

    // one more reduction across smem collected maxes
    // this code assumes you can launch at most 32x32 warps over a block
    if (warp_id == 0) {
        for (int i = 0; i < num_warps; i += warp.size()) {
            max_score = (lane + i < num_warps) ? warp_row_maxes[lane + i] : -CUDART_INF_F;
            warp.sync();
            for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
                float other = warp.shfl_xor(max_score, offset);
                max_score = fmaxf(max_score, other);
            }
            if (lane == 0)
                warp_row_maxes[0] = fmaxf(max_score, warp_row_maxes[0]);
            warp.sync();
        }
    }
    cg_block.sync();
    max_score = warp_row_maxes[0];

    // compute sum of exp(score - row_max) across all columns
    float *attn_bh_base = ATTN_PTR(attention_scores, bh_idx, seq_len);
    float strided_sum_score = 0.0f;

    for (int col = tid; col < seq_len; col += blockDim.x) {
        int idx = row * seq_len + col;
        float score = expf(attn_bh_base[idx] - max_score);
        strided_sum_score += score;
        attn_bh_base[idx] = score;
    }

    // at this point each thread of the single block will have the strided sum
    cg_block.sync();

    // only single reduction workload left at this point
    // per-warp sum scores i.e thread 0 will hold the final sum
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
        strided_sum_score += warp.shfl_down(strided_sum_score, offset);

    if (lane == 0)
        warp_row_sums[warp_id] = strided_sum_score;
    cg_block.sync();

    // one more reduction across smem collected sums
    // this code assumes you can launch at most 32x32 warps over a block
    if (warp_id == 0) {
        // reuse warp_row_sums for final sum
        float row_sum = 0.0f;
        for (int i = 0; i < num_warps; i += warp.size()) {
            strided_sum_score = (lane + i < num_warps) ? warp_row_sums[lane + i] : 0.0f;
            for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
                strided_sum_score += warp.shfl_down(strided_sum_score, offset);
            }
            if (lane == 0)
                row_sum += strided_sum_score;
        }
        if (lane == 0)
            warp_row_sums[0] = row_sum; // store final row sum
    }
    cg_block.sync();

    // normalize attention scores
    const float norm = 1.0f / warp_row_sums[0];
    for (int col = tid; col < seq_len; col += blockDim.x) {
        attn_bh_base[row * seq_len + col] *= norm;
    }
}

// Kernel: apply softmax and multiply by V
__global__ void softmax_multV_v2(const float *__restrict__ attention_scores,
                                 const float *__restrict__ V, float *O, int seq_len, int head_dim) {
    // cooperative-groups block object
    cg::thread_block cg_block = cg::this_thread_block();

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

    __shared__ alignas(float4) float softmax_tile[TILE_DIM][TILE_DIM];
    __shared__ float V_tile[TILE_DIM][TILE_DIM + 1];

    float acc = 0.0f;

    int num_tiles = (seq_len + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int k = t * TILE_DIM + local_col;

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
        cg_block.sync();

        // inner product for this tile vectorized in groups of 4
        const float4 *s_vec_row = reinterpret_cast<const float4 *>(&softmax_tile[local_row][0]);
#pragma unroll
        for (int k = 0; k < TILE_DIM; k += 4) {
            float4 s_vec = s_vec_row[k / 4];

            float4 v_vec;
            v_vec.x = V_tile[k + 0][local_col];
            v_vec.y = V_tile[k + 1][local_col];
            v_vec.z = V_tile[k + 2][local_col];
            v_vec.w = V_tile[k + 3][local_col];

            acc = fmaf(s_vec.x, v_vec.x, acc);
            acc = fmaf(s_vec.y, v_vec.y, acc);
            acc = fmaf(s_vec.z, v_vec.z, acc);
            acc = fmaf(s_vec.w, v_vec.w, acc);
        }
        cg_block.sync();
    }

    if (row < seq_len && col < head_dim) {
        Obh_base[row * head_dim + col] = acc;
    }
}

// Simple attention implementation
struct VanillaAttentionV2 : public Attention {

    // Kernel launch: transpose K
    void launch_transpose_K(const float *K, float *K_transposed, int batch_heads, int seq_len,
                            int head_dim, int blocks) {
        dim3 threads(TILE_DIM, TILE_DIM);
        int blocks_x = (head_dim + TILE_DIM - 1) / TILE_DIM;
        int blocks_y = blocks;
        dim3 grid(blocks_x, blocks_y, batch_heads);

        transpose_K<<<grid, threads>>>(K, K_transposed, seq_len, head_dim);
        // cudaDeviceSynchronize();
        // CUDA_CHECK();
    }

    // Kernel launch: compute QK^T with partial max
    void launch_qk_dot_partial_reduce(const float *Q, const float *K_transposed,
                                      float *attention_scores, float *row_max_partials,
                                      int batch_heads, int seq_len, int head_dim, float scale,
                                      int blocks) {
        int warp_size;
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, 0);

        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid(blocks, blocks, batch_heads);

        int warps_per_block = TILE_DIM * (TILE_DIM / warp_size);
        size_t shared_bytes = warps_per_block * sizeof(float);

        qk_dot_partial_reduce_v2<<<grid, threads, shared_bytes>>>(
            Q, K_transposed, attention_scores, row_max_partials, seq_len, head_dim, scale);
        // cudaDeviceSynchronize();
        // CUDA_CHECK();
        // save_device_ptr_as_buffer("QKt.bin", attention_scores, n_qkt);
    }

    // Kernel launch: softmax in-place
    void launch_softmax_inplace(float *attention_scores, const float *row_max_partials,
                                int batch_heads, int seq_len, int blocks_x) {

        int warp_size;
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, 0);

        int device_max_threads;
        cudaDeviceGetAttribute(&device_max_threads, cudaDevAttrMaxThreadsPerBlock, 0);
        int per_row_block_threads = std::min(TILE_DIM * blocks_x, device_max_threads);
        per_row_block_threads = std::max(warp_size, per_row_block_threads);

        int warps_per_block = per_row_block_threads / warp_size;
        size_t shared_bytes = warps_per_block * sizeof(float);

        dim3 threads(per_row_block_threads, 1, 1);
        dim3 grid(1, seq_len, batch_heads);

        softmax_inplace_v2<<<grid, threads, shared_bytes>>>(attention_scores, row_max_partials,
                                                            seq_len, blocks_x);
        // cudaDeviceSynchronize();
        // CUDA_CHECK();
    }

    // Kernel launch: softmax multiply by V
    void launch_softmax_multV(float *attention_scores, const float *V, float *O, int batch_heads,
                              int seq_len, int head_dim, int blocks) {
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 grid(blocks, blocks, batch_heads);

        softmax_multV_v2<<<grid, threads>>>(attention_scores, V, O, seq_len, head_dim);
        // cudaDeviceSynchronize();
        // CUDA_CHECK();
    }

    void forward(const float *Q, const float *K, const float *V, float *O, uint32_t batch_size,
                 uint32_t num_heads, uint32_t seq_len, uint32_t head_dim) override {

        int warp_size;
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, 0);
        assert(TILE_DIM >= warp_size && TILE_DIM % warp_size == 0);

        float *attention_scores = nullptr;
        float *K_transposed = nullptr;
        float *row_max_partials = nullptr;

        const int batch_heads = batch_size * num_heads;
        const float scale = 1.0f / sqrtf((float) head_dim);
        const size_t n_qkt = batch_heads * seq_len * seq_len;
        const size_t blocks = (seq_len + TILE_DIM - 1) / TILE_DIM; // blocks in 1 direction
        const size_t partials_blocks_x = blocks;
        const size_t n_partials = batch_heads * seq_len * partials_blocks_x;

        cudaMalloc(&attention_scores, n_qkt * sizeof(float));
        cudaMalloc(&K_transposed, batch_heads * head_dim * seq_len * sizeof(float));
        cudaMalloc(&row_max_partials, n_partials * sizeof(float));

        launch_transpose_K(K, K_transposed, batch_heads, seq_len, head_dim, blocks);
        launch_qk_dot_partial_reduce(Q, K_transposed, attention_scores, row_max_partials,
                                     batch_heads, seq_len, head_dim, scale, blocks);
        launch_softmax_inplace(attention_scores, row_max_partials, batch_heads, seq_len,
                               partials_blocks_x);
        launch_softmax_multV(attention_scores, V, O, batch_heads, seq_len, head_dim, blocks);

        cudaDeviceSynchronize();
        cudaFree(K_transposed);
        cudaFree(attention_scores);
        cudaFree(row_max_partials);
        CUDA_CHECK();
    }
};

// Factory function
extern "C" Attention *create_vanilla_attention_v2() {
    return new VanillaAttentionV2();
}

__device__ float reduce_block_row_sums(cg::thread_block cg_block, float score,
                                       float *block_smem_scores) {
    // create a warp-sized tile group
    auto warp = cg::tiled_partition<32>(cg_block);
    int lane = warp.thread_rank();
    int warps_per_row = (TILE_DIM + warp.size() - 1) / warp.size();

    // per-warp sum scores i.e thread 0 will hold the final sum
    float sum_score = score;
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        sum_score += warp.shfl_down(sum_score, offset);
    }

    // each block-row has more than 1 warps
    if (warps_per_row > 1) {
        int warp_id_row = threadIdx.y;
        int warp_id_col = threadIdx.x / warp.size();
        // thread 0 per-warp loads its sum scores in smem
        if (lane == 0) {
            int idx = warp_id_row * warps_per_row + warp_id_col;
            block_smem_scores[idx] = sum_score;
        }
        cg_block.sync();

        // first warp of every block-row
        if (warp_id_col == 0) {
            // the same number of threads as the warps per block-row
            int idx = warp_id_row * warps_per_row + lane;
            sum_score = (lane < warps_per_row) ? block_smem_scores[idx] : 0.0f;

            for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
                sum_score += warp.shfl_down(sum_score, offset);
            }
        }
        cg_block.sync();
    }
    // at this point, column 0 of each block will hold sum per block

    return sum_score;
}

__device__ float atomicMaxFloat(float *addr, float value) {
    int *addr_as_int = reinterpret_cast<int *>(addr);
    int old = *addr_as_int, assumed;

    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val >= value)
            break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}
