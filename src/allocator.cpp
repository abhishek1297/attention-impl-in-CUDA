#include "allocator.hpp"
#include <stdexcept>

MatrixManager::MatrixManager() : gen(nullptr) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

MatrixManager::~MatrixManager() {
    free_all();
    if (gen)
        curandDestroyGenerator(gen);
}

float *MatrixManager::allocate_and_fill(int n_elements, bool zeros) {
    float *ptr;
    cudaError_t err = cudaMalloc(&ptr, n_elements * sizeof(float));
    if (err != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    allocated_.push_back(ptr);
    if (!zeros) {
        curandStatus_t curand_err = curandGenerateUniform(gen, ptr, n_elements);
        if (curand_err != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("cuRAND generation failed");
    } else
        cudaMemset(ptr, 0, n_elements * sizeof(float));
    cudaDeviceSynchronize();
    return ptr;
}

bool MatrixManager::can_allocate(size_t bytes) {
    size_t free_bytes;
    cudaMemGetInfo(&free_bytes, nullptr);
    return bytes < free_bytes;
}

void MatrixManager::free_all() {
    for (auto ptr : allocated_) {
        cudaFree(ptr);
    }
    allocated_.clear();
}