#include "benchmark.hpp"

#include <iostream>
#include <numeric>

Benchmark::Benchmark(const std::string &filename, std::function<Attention *()> factory)
    : filename(filename), factory(factory) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    csv_file.open(filename);
    csv_file << "batch_size,num_heads,seq_len,head_dim,avg_time_ms,tflops\n";
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

double Benchmark::get_tflops(uint32_t batch_size, uint32_t num_heads, uint32_t seq_len,
                             uint32_t head_dim, float avg_time_ms) const {
    double batch_heads = static_cast<double>(batch_size) * num_heads;

    // QK^T
    double qkt_flops = batch_heads * seq_len * seq_len * (2.0 * head_dim - 1.0);

    // Softmax (approx. 4 * seq_len^2 per head)
    double softmax_flops = batch_heads * 4.0 * seq_len * seq_len;

    // Softmax * V
    double sv_flops = batch_heads * seq_len * head_dim * (2.0 * seq_len - 1.0);

    double total_flops = qkt_flops + softmax_flops + sv_flops;

    double seconds = avg_time_ms * 1e-3;
    double tflops = total_flops / (seconds * 1e12);
    return tflops;
}

void Benchmark::run(const uint32_t total_runs, const float *Q, const float *K, const float *V,
                    float *O, uint32_t batch_size, uint32_t num_heads, uint32_t seq_len,
                    uint32_t head_dim) {

    Attention *attn = factory();

    // warm-up call
    attn->forward(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);

    for (int i = 0; i < total_runs; ++i) {
        start();
        attn->forward(Q, K, V, O, batch_size, num_heads, seq_len, head_dim);
        stop();
    }

    float avg_time_ms = get_avg_elapsed_time();
    double tflops = get_tflops(batch_size, num_heads, seq_len, head_dim, avg_time_ms);

    csv_file << batch_size << "," << num_heads << "," << seq_len << "," << head_dim << ","
             << avg_time_ms << "," << tflops << "\n";
    times.clear();
}
