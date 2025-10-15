#pragma once

#include "attention.hpp"

#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

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
    double get_tflops(uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
                      float avg_time_ms) const;

    void run(uint32_t total_runs, const float *Q, const float *K, const float *V, float *O,
             uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim);

  private:
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    std::vector<float> times;
    std::string filename;
    std::ofstream csv_file;
    std::function<Attention *()> factory;
};
