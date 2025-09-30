# Attention Implementations in CUDA

This repository provides reference implementations of various attention mechanisms in CUDA, focusing on efficient GPU computation for deep learning models. It includes classic and modern attention variants, aiming to serve as a resource for benchmarking and understanding performance trade-offs in CUDA-based attention layers.

## Installation and running
You can set up the environment using the provided **spack** environment files in this repository. Alternatively, ensure that the CUDA Toolkit is discoverable by CMake. While there is no strict CUDA version requirement, using CUDA 12 or newer is recommended as some implementations rely on cuBLAS, cuDNN, etc.

### Building
```bash
# from root
./scripts/build.sh
```

### Run the executable
```bash
./build/cuda_attention
```

# TODO:

*The aim is to perform forward pass (GEMM ➔ softmax ➔ GEMM) alone.*

- [ ] (Vanilla) Multi-Head Attention – parallel heads from the Transformer paper.
  - [x] Create a test kernel for batched/multi-headed GEMM using plain CUDA
  - [ ] Create a test kernel for batched/multi-headed GEMM using MMA tensor cores
  - [ ] Create a **baseline** kernel using cuBLAS/CUTLASS for batched/multi-headed GEMM using tensor cores
- [ ] Sparse / Local Attention – e.g., Longformer or Neighborhood Attention.
- [ ] Linformer / Linear Attention – low-rank or kernel tricks to reduce complexity.
- [ ] Performer – FAVOR+ kernel feature maps for linear-time softmax approximation.
- [ ] FlashAttention – memory-efficient, blockwise softmax on GPU.
- [ ] FlashAttention-2 – improved tiling + parallelism for long sequences.
