#pragma once

#include "attention.hpp"

#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#define TOTAL_RUNS_PER_BENCHMARK 5

class Benchmark {
  public:
    Benchmark(const std::string &filename, std::function<Attention *()> factory);
    ~Benchmark();

    Benchmark(const Benchmark &) = delete;
    Benchmark &operator=(const Benchmark &) = delete;
    Benchmark(Benchmark &&) = delete;
    Benchmark &operator=(Benchmark &&) = delete;

    void start();
    void stop();

    float get_avg_elapsed_time() const;
    double get_gflops(int batch_size, int num_heads, int seq_len, int head_dim,
                      float avg_time_ms) const;

    void run(float *Q, float *K, float *V, float *O, int batch_size, int num_heads, int seq_len,
             int head_dim);

  private:
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    std::vector<float> times;
    std::string filename;
    std::ofstream csv_file;
    std::function<Attention *()> factory;
};
