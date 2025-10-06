#include "attention.hpp"

#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>
#include <stdexcept>

// Kernel: compute row-wise softmax in-place for strided batched matrices
__global__ void batched_softmax_inplace(float *scores, int seq_len, int batch_heads) {
    int bh_idx = blockIdx.z;
    int row = blockIdx.y;

    if (bh_idx >= batch_heads || row >= seq_len)
        return;

    float *row_ptr = scores + BH_OFFSET(bh_idx, seq_len, seq_len) + row * seq_len;

    // Find max (for numerical stability)
    float max_val = -CUDART_INF_F;
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float val = row_ptr[col];
        if (val > max_val)
            max_val = val;
    }

    // Reduce max across threads in block
    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float other = shared_max[threadIdx.x + offset];
            if (other > shared_max[threadIdx.x])
                shared_max[threadIdx.x] = other;
        }
        __syncthreads();
    }
    max_val = shared_max[0];

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float val = expf(row_ptr[col] - max_val);
        row_ptr[col] = val;
        sum += val;
    }

    // Reduce sum across threads
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    sum = shared_sum[0];

    // Normalize
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        row_ptr[col] /= sum;
    }
}

// cuBLAS-based attention implementation
struct CublasAttention : public Attention {
    cublasHandle_t handle;

    CublasAttention() {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }

    ~CublasAttention() { cublasDestroy(handle); }

    void forward(const float *Q, const float *K, const float *V, float *O, uint batch_size,
                 uint num_heads, uint seq_len, uint head_dim) override {

        uint batch_heads = batch_size * num_heads;
        float *attention_scores;

        uint n_scores = batch_heads * seq_len * seq_len;
        cudaMalloc(&attention_scores, n_scores * sizeof(float));

        // Scale factor for attention
        float scale = 1.0f / sqrtf((float) head_dim);
        float alpha = scale;
        float beta = 0.0f;

        // Step 1: Compute Q @ K^T (strided batched matrix multiplication)
        cublasStatus_t status =
            cublasSgemmStridedBatched(handle,
                                      CUBLAS_OP_T, // K^T (transpose K)
                                      CUBLAS_OP_N, // Q (no transpose)
                                      seq_len,     // m: rows of K^T = seq_len
                                      seq_len,     // n: cols of Q = seq_len
                                      head_dim,    // k: cols of K^T = rows of Q = head_dim
                                      &alpha,      // scaling factor
                                      K,           // K matrix
                                      head_dim,    // leading dimension of K (row-major: head_dim)
                                      seq_len * head_dim, // stride between K matrices
                                      Q,                  // Q matrix
                                      head_dim, // leading dimension of Q (row-major: head_dim)
                                      seq_len * head_dim, // stride between Q matrices
                                      &beta,              // beta for C
                                      attention_scores,   // output scores
                                      seq_len,            // leading dimension of scores
                                      seq_len * seq_len,  // stride between score matrices
                                      batch_heads         // batch count
            );

        if (status != CUBLAS_STATUS_SUCCESS) {
            cudaFree(attention_scores);
            throw std::runtime_error("cuBLAS GEMM failed");
        }

        cudaDeviceSynchronize();

        // Step 2: Apply softmax row-wise
        dim3 threads(256, 1, 1);
        dim3 grid(1, seq_len, batch_heads);
        batched_softmax_inplace<<<grid, threads>>>(attention_scores, seq_len, batch_heads);
        cudaDeviceSynchronize();

        // Step 3: Compute attention_scores @ V (strided batched matrix multiplication)
        alpha = 1.0f;
        beta = 0.0f;

        status = cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N, // V (no transpose)
                                           CUBLAS_OP_N, // scores (no transpose)
                                           head_dim,    // m: cols of V = head_dim
                                           seq_len,     // n: rows of scores = seq_len
                                           seq_len,     // k: rows of V = cols of scores = seq_len
                                           &alpha,      // scaling factor
                                           V,           // V matrix
                                           head_dim, // leading dimension of V (row-major: head_dim)
                                           seq_len * head_dim, // stride between V matrices
                                           attention_scores,   // attention scores
                                           seq_len,            // leading dimension of scores
                                           seq_len * seq_len,  // stride between score matrices
                                           &beta,              // beta for O
                                           O,                  // output
                                           head_dim, // leading dimension of O (row-major: head_dim)
                                           seq_len * head_dim, // stride between O matrices
                                           batch_heads         // batch count
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            cudaFree(attention_scores);
            throw std::runtime_error("cuBLAS GEMM (scores @ V) failed");
        }

        cudaDeviceSynchronize();

        cudaFree(attention_scores);
    }
};

// Factory function
extern "C" Attention *create_vanilla_cublas_attention() {
    return new CublasAttention();
}