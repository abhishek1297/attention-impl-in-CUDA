#include "attention.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>

// External factory
extern "C" Attention* create_vanilla_attention();

void initialize(float* data, int size) {
    for (int i = 0; i < size; i++) data[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main() {
    // Benchmark parameters
    std::vector<int> seq_lengths = {64, 128, 256, 512, 1024, 2048};
    std::vector<int> embed_dims  = {32, 64, 128, 256, 512};
    int batch = 1;

    // Open CSV file
    std::ofstream out_file("benchmarks/vanilla_benchmark.csv");
    out_file << "seq_len,embed_dim,time_ms\n";

    // Create vanilla attention
    Attention* attn = create_vanilla_attention();

    for (int seq_len : seq_lengths) {
        for (int dim : embed_dims) {
            size_t qkv_size = seq_len * dim * sizeof(float);

            // Allocate unified memory for Q, K, V, out
            float *Q, *K, *V, *out;
            cudaMallocManaged(&Q, qkv_size);
            cudaMallocManaged(&K, qkv_size);
            cudaMallocManaged(&V, qkv_size);
            cudaMallocManaged(&out, qkv_size);

            // Initialize dummy data
            initialize(Q, seq_len * dim);
            initialize(K, seq_len * dim);
            initialize(V, seq_len * dim);

            // Timing with CUDA events
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            attn->forward(Q, K, V, out, batch, seq_len, dim);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            // Write to CSV
            out_file << seq_len << "," << dim << "," << ms << "\n";
            std::cout << "seq_len=" << seq_len << " dim=" << dim << " time=" << ms << " ms\n";

            // Free memory
            cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(out);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    out_file.close();
    delete attn;
    return 0;
}
