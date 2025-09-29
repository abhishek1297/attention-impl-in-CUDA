#include "attention.h"
#include <iostream>

// External factory
extern "C" Attention* create_vanilla_attention();

int main() {
    int batch = 1, seq_len = 1000, dim = 128;
    float *Q, *K, *V, *out;
    cudaMallocManaged(&Q, seq_len * dim * sizeof(float));
    cudaMallocManaged(&K, seq_len * dim * sizeof(float));
    cudaMallocManaged(&V, seq_len * dim * sizeof(float));
    cudaMallocManaged(&out, seq_len * dim * sizeof(float));

    // Init with dummy values
    for (int i = 0; i < seq_len * dim; i++) {
        Q[i] = K[i] = V[i] = 0.1f * (i + 1);
    }

    Attention* attn = create_vanilla_attention();
    attn->forward(Q, K, V, out, batch, seq_len, dim);

    std::cout << "Output[0]: " << out[0] << std::endl;

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(out);
    delete attn;
    return 0;
}
