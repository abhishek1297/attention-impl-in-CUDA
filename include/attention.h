#pragma once
#include <cuda_runtime.h>

// Abstract interface for different attention implementations
struct Attention {
    virtual void forward(float* Q, float* K, float* V,
                         float* out,
                         int batch, int seq_len, int dim) = 0;
    virtual ~Attention() {}
};
