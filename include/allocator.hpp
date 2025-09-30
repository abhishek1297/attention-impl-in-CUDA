#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <vector>

class MatrixManager {
  public:
    MatrixManager();
    ~MatrixManager();

    float *allocate_and_fill(int n_elements, bool zeros = false);
    bool can_allocate(size_t bytes);
    void free_all();

  private:
    curandGenerator_t gen;
    std::vector<float *> allocated_;
};