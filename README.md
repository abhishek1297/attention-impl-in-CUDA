# Attention Implementations in CUDA
Implementation of various Attention from Vanilla to Optimized using CUDA.

# TODO:
- [ ] Vanilla Scaled Dot-Product Attention – the original Transformer formulation.
- [ ] Multi-Head Attention – parallel heads from the Transformer paper.
- [ ] Sparse / Local Attention – e.g., Longformer or Neighborhood Attention.
- [ ] Linformer / Linear Attention – low-rank or kernel tricks to reduce complexity.
- [ ] Performer – FAVOR+ kernel feature maps for linear-time softmax approximation.
- [ ] FlashAttention – memory-efficient, blockwise softmax on GPU.
- [ ] FlashAttention-2 – improved tiling + parallelism for long sequences.
