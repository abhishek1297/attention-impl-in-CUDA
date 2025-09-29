# Attention Implementations in CUDA

This repository provides reference implementations of various attention mechanisms in CUDA, focusing on efficient GPU computation for deep learning models. It includes classic and modern attention variants, aiming to serve as a resource for benchmarking and understanding performance trade-offs in CUDA-based attention layers.

## Installation and running
You can set up the environment using the provided **spack** environment files in this repository. Alternatively, ensure that the CUDA Toolkit is discoverable by CMake. While there is no strict CUDA version requirement, using CUDA 12 or newer is recommended since the implementation relies on cuBLAS and cuDNN.

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
- [ ] Vanilla Scaled Dot-Product Attention – the original Transformer formulation.
- [ ] Multi-Head Attention – parallel heads from the Transformer paper.
- [ ] Sparse / Local Attention – e.g., Longformer or Neighborhood Attention.
- [ ] Linformer / Linear Attention – low-rank or kernel tricks to reduce complexity.
- [ ] Performer – FAVOR+ kernel feature maps for linear-time softmax approximation.
- [ ] FlashAttention – memory-efficient, blockwise softmax on GPU.
- [ ] FlashAttention-2 – improved tiling + parallelism for long sequences.
