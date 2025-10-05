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
#include <set>
#include <yaml-cpp/yaml.h>

// External factories
extern "C" Attention *create_vanilla_attention();
extern "C" Attention *create_vanilla_cublas_attention();

struct BenchmarkEntry {
    std::string key;
    std::unique_ptr<Benchmark> instance;
};

// Helper to run return selected benchmarks
std::vector<std::unique_ptr<Benchmark>> get_selected_benchmarks(const std::set<std::string>& keys) {
    std::vector<BenchmarkEntry> all_benchmarks;
    all_benchmarks.push_back({"vanilla", std::make_unique<Benchmark>(get_benchmark_filename("vanilla"), create_vanilla_attention)});
    all_benchmarks.push_back({"vanilla_cublas", std::make_unique<Benchmark>(get_benchmark_filename("vanilla_cublas"), create_vanilla_cublas_attention)});

    std::vector<std::unique_ptr<Benchmark>> selected;
    for (auto& entry : all_benchmarks) {
        if (keys.count(entry.key)) {
            selected.push_back(std::move(entry.instance));
        }
    }
    return selected;
}

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
    // Read benchmark parameters from YAML
    YAML::Node config = YAML::LoadFile("params.yaml");
    std::vector<int> seq_lengths = config["seq_lengths"].as<std::vector<int>>();
    std::vector<int> embed_dims = config["embed_dims"].as<std::vector<int>>();
    std::vector<int> num_attn_heads = config["num_attn_heads"].as<std::vector<int>>();
    std::vector<int> batch_sizes = config["batch_sizes"].as<std::vector<int>>();

    write_gpu_info();
    std::set<std::string> to_run;
    for (const auto& name : config["benchmarks_to_run"]) {
        to_run.insert(name.as<std::string>());
    }
    auto benchmarks = get_selected_benchmarks(to_run);

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
        save_device_ptr_as_buffer("Q.bin", Q, qkv_elements);
        save_device_ptr_as_buffer("K.bin", K, qkv_elements);
        save_device_ptr_as_buffer("V.bin", V, qkv_elements);

        for (auto &bmark : benchmarks)
            bmark->run(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);

        save_device_ptr_as_buffer("O.bin", O, qkv_elements);
        mm.free_all();
        std::cout << "Benchmark for batch_size=" << batch_size << ", num_heads=" << num_heads
                  << ", seq_len=" << seq_len << ", embed_dim=" << embed_dim << " finished."
                  << std::endl;
    }

    return 0;
}