#include "allocator.hpp"
#include "attention.hpp"
#include "benchmark.hpp"
#include "cuda_utils.hpp"

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// External factories
extern "C" Attention *create_vanilla_attention();
extern "C" Attention *create_vanilla_cublas_attention();

// Helper to build all combinations
template <typename... Vecs> auto cartesian_product(const Vecs &...vecs) {
    std::vector<std::tuple<typename Vecs::value_type...>> result;
    auto add = [&](auto self, auto tup, const auto &vec, const auto &...rest) -> void {
        for (auto v : vec) {
            auto new_tup = std::tuple_cat(tup, std::make_tuple(v));
            if constexpr (sizeof...(rest) == 0)
                result.push_back(new_tup);
            else
                self(self, new_tup, rest...);
        }
    };
    add(add, std::tuple<>{}, vecs...);
    return result;
}

int main() {
    MatrixManager mm;
    // Benchmark parameters
    std::vector<int> seq_lengths = {128, 512, 1024, 2048, 4096};
    std::vector<int> embed_dims = {128, 4096, 8192};
    std::vector<int> num_attn_heads = {1, 2, 8, 16};
    std::vector<int> batch_sizes = {1};

    write_gpu_info();
    // Define (filename, factory function) pairs
    std::vector<std::unique_ptr<Benchmark>> benchmarks;
    benchmarks.push_back(
        std::make_unique<Benchmark>(get_benchmark_filename("vanilla"), create_vanilla_attention));
    benchmarks.push_back(std::make_unique<Benchmark>(get_benchmark_filename("vanilla_cublas"),
                                                     create_vanilla_cublas_attention));

    for (auto [batch_size, num_heads, seq_len, embed_dim] :
         cartesian_product(batch_sizes, num_attn_heads, seq_lengths, embed_dims)) {

        assert(embed_dim % num_heads == 0);
        int head_dim = (int) embed_dim / num_heads;
        size_t qkv_elements = batch_size * num_heads * seq_len * head_dim;
        // Check if allocation is possible
        if (!mm.can_allocate(qkv_elements * 4 * sizeof(float))) {
            std::cerr << "Skipping: Not enough memory for batch_size=" << batch_size
                      << ", num_heads=" << num_heads << ", seq_len=" << seq_len
                      << ", embed_dim=" << embed_dim << std::endl;
            continue;
        }
        float *Q = mm.allocate_and_fill(qkv_elements);
        float *K = mm.allocate_and_fill(qkv_elements);
        float *V = mm.allocate_and_fill(qkv_elements);
        float *O = mm.allocate_and_fill(qkv_elements, true);
        // save_device_ptr_as_buffer("Q.bin", Q, qkv_elements);
        // save_device_ptr_as_buffer("K.bin", K, qkv_elements);
        // save_device_ptr_as_buffer("V.bin", V, qkv_elements);

        for (auto &bmark : benchmarks)
            bmark->run(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);

        // save_device_ptr_as_buffer("O.bin", O, qkv_elements);
        mm.free_all();
        std::cout << "Benchmark for batch_size=" << batch_size << ", num_heads=" << num_heads
                  << ", seq_len=" << seq_len << ", embed_dim=" << embed_dim << " finished."
                  << std::endl;
    }

    return 0;
}