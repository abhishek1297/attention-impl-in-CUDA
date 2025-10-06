#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define BH_ROW_IDX(bh_idx, row, seq_len) ((size_t) (bh_idx) * (seq_len) + (row))
#define BH_OFFSET(bh_idx, seq_len, dim) (BH_ROW_IDX(bh_idx, 0, seq_len) * (dim))
#define ROW_OFFSET(bh_idx, row, seq_len, dim) (BH_ROW_IDX(bh_idx, row, seq_len) * (dim))
#define ELEM_OFFSET(bh_idx, row, col, seq_len, dim) (ROW_OFFSET(bh_idx, row, seq_len, dim) + (col))

#define Q_PTR(Q, bh_idx, seq_len, head_dim) ((Q) + BH_OFFSET(bh_idx, seq_len, head_dim))
#define K_PTR(K, bh_idx, seq_len, head_dim) ((K) + BH_OFFSET(bh_idx, seq_len, head_dim))
#define V_PTR(V, bh_idx, seq_len, head_dim) ((V) + BH_OFFSET(bh_idx, seq_len, head_dim))
#define O_PTR(O, bh_idx, seq_len, head_dim) ((O) + BH_OFFSET(bh_idx, seq_len, head_dim))
#define ATTN_PTR(A, bh_idx, seq_len) ((A) + BH_OFFSET(bh_idx, seq_len, seq_len))

// Abstract interface for different attention implementations
struct Attention {
    virtual void forward(const float *Q, const float *K, const float *V, float *O,
                         uint32_t batch_size, uint32_t num_heads, uint32_t seq_len,
                         uint32_t head_dim) = 0;
    virtual ~Attention() {}
};
