#include "benchmark.hpp"

#include <iostream>
#include <numeric>

Benchmark::Benchmark(const std::string &filename, std::function<Attention *()> factory)
    : filename(filename), factory(factory) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    csv_file.open(filename);
    csv_file << "batch_size,num_heads,seq_len,head_dim,avg_time_ms,gflops\n";
}

Benchmark::~Benchmark() {
    if (start_event)
        cudaEventDestroy(start_event);
    if (stop_event)
        cudaEventDestroy(stop_event);

    csv_file.close();
}

void Benchmark::start() {
    cudaEventRecord(start_event);
}

void Benchmark::stop() {
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event, stop_event);
    times.push_back(ms);
}

float Benchmark::get_avg_elapsed_time() const {
    if (times.empty())
        return 0.0f;
    return std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
}

double Benchmark::get_gflops(int batch_size, int num_heads, int seq_len, int head_dim,
                             float avg_time_ms) const {
    double flops = batch_size * num_heads *
                   (2.0 * seq_len * seq_len * head_dim + // QK^T
                    seq_len * seq_len +                  // softmax
                    2.0 * seq_len * seq_len * head_dim   // softmax*V
                   );
    return (flops / (avg_time_ms / 1000.0)) / 1e9;
}

void Benchmark::run(int total_runs, float *Q, float *K, float *V, float *O, int batch_size,
                    int num_heads, int seq_len, int head_dim) {

    Attention *attn = factory();

    // warm-up call
    attn->forward(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);

    for (int i = 0; i < total_runs; ++i) {
        start();
        attn->forward(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);
        stop();
    }

    float avg_time_ms = get_avg_elapsed_time();
    double gflops = get_gflops(batch_size, num_heads, seq_len, head_dim, avg_time_ms);

    csv_file << batch_size << "," << num_heads << "," << seq_len << "," << head_dim << ","
             << avg_time_ms << "," << gflops << "\n";
}
