#include "attention.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>

// Kernel: naive matrix multiply for QK^T
__global__ void qk_dot(float* Q, float* K, float* scores,
                       int batch, int seq_len, int dim) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len) return;

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += Q[q_idx * dim + d] * K[k_idx * dim + d];
        }
        scores[q_idx * seq_len + k_idx] = dot / sqrtf((float)dim);
    }
}

// Wrapper function
void multiply_qkdot(float* Q, float* K, float* d_scores,
                    int batch, int seq_len, int dim) {
    int threads = 128;
    int blocks = (seq_len + threads - 1) / threads;
    qk_dot<<<blocks, threads>>>(Q, K, d_scores, batch, seq_len, dim);
    cudaDeviceSynchronize();
}

// Naive row-wise softmax kernel
__global__ void row_softmax(float* scores, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    // Find max for numerical stability
    float max_val = scores[row * seq_len];
    for (int j = 1; j < seq_len; j++) {
        float val = scores[row * seq_len + j];
        if (val > max_val) max_val = val;
    }

    // Compute sum of exp
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float e = expf(scores[row * seq_len + j] - max_val);
        scores[row * seq_len + j] = e;  // store temporarily
        sum += e;
    }

    // Normalize
    for (int j = 0; j < seq_len; j++) {
        scores[row * seq_len + j] /= sum;
    }
}

// Wrapper function
void softmax_forward(float* d_scores, int seq_len) {
    int threads = 128;
    int blocks = (seq_len + threads - 1) / threads;
    row_softmax<<<blocks, threads>>>(d_scores, seq_len);
    cudaDeviceSynchronize();  // wait for completion
}

// Naive matmul: out = softmax_scores * V
__global__ void softmax_times_v(float* softmax_scores, float* V, float* out,
                                int seq_len, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += softmax_scores[row * seq_len + j] * V[j * dim + d];
        }
        out[row * dim + d] = sum;
    }
}

// Wrapper function
void multiply_softmax_v(float* d_scores, float* V, float* out,
                        int seq_len, int dim) {
    int threads = 128;
    int blocks = (seq_len + threads - 1) / threads;
    softmax_times_v<<<blocks, threads>>>(d_scores, V, out, seq_len, dim);
    cudaDeviceSynchronize();
}

// Simple attention implementation
struct VanillaAttention : public Attention {
    void forward(float* Q, float* K, float* V,
                 float* out,
                 int batch, int seq_len, int dim) override {
        float* d_scores;
        cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));

        multiply_qkdot(Q, K, d_scores, batch, seq_len, dim);
        softmax_forward(d_scores, seq_len);
        multiply_softmax_v(d_scores, V, out, seq_len, dim);

        cudaMemcpy(out, d_scores, seq_len * seq_len * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        cudaFree(d_scores);
    }
};

// Factory function
extern "C" Attention* create_vanilla_attention() {
    return new VanillaAttention();
}
