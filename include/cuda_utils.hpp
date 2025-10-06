#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#define CUDA_CHECK() cuda_check(__FILE__, __LINE__)

inline void cuda_check(const char *file, int line) {
    cudaError_t err = cudaGetLastError(); // get last error from runtime
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " | File: " << file
                  << " | Line: " << line << std::endl;
        std::terminate(); // stop execution; optional: throw
    }
}

inline std::string sanitize_filename(const std::string &name) {
    std::string result = name;
    // Replace spaces and special characters with underscores
    std::replace_if(
        result.begin(), result.end(), [](char c) { return !(isalnum(c) || c == '_'); }, '_');
    return result;
}

inline std::string get_benchmark_filename(const std::string &base_name) {
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::string dev_name = sanitize_filename(prop.name);
    int major = prop.major;
    int minor = prop.minor;

    return "benchmarks/" + base_name + "_" + dev_name + "_sm" + std::to_string(major) +
           std::to_string(minor) + ".csv";
}

inline void save_device_ptr_as_buffer(std::string fname, float *device_ptr, int n_elements) {
    float *host_ptr = static_cast<float *>(malloc(n_elements * sizeof(float)));
    cudaMemcpy(host_ptr, device_ptr, n_elements * sizeof(float), cudaMemcpyDeviceToHost);
    FILE *f = fopen(fname.c_str(), "wb");
    if (f) {
        fwrite(host_ptr, sizeof(float), n_elements, f);
        fclose(f);
    }
    free(host_ptr);
}

inline void write_gpu_info() {
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::string name(prop.name);
    int sm_count = prop.multiProcessorCount;
    double clock_rate_hz = static_cast<double>(prop.clockRate) * 1e3;

    // FP32 throughput estimate: 128 FMA units per SM => 256 FLOPs/clock per SM
    const int fma_per_cycle_per_sm = 128;
    double theoretical_flops =
        static_cast<double>(sm_count) * fma_per_cycle_per_sm * 2.0 * clock_rate_hz;
    double theoretical_gflops = theoretical_flops / 1e9;

    // Memory bandwidth (approximate)
    double mem_bandwidth_gb = 2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) / 1e9;

    // Additional architectural info
    int warp_size = prop.warpSize;
    size_t shared_mem_per_block = prop.sharedMemPerBlock;
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int regs_per_block = prop.regsPerBlock;
    int major_cc = prop.major;
    int minor_cc = prop.minor;

    // Write to CSV
    std::ofstream out("benchmarks/device_info.csv");
    out << "name,sm_count,clock_rate_hz,theoretical_gflops,memory_bandwidth_gb,"
        << "warp_size,shared_mem_per_block,max_threads_per_block,regs_per_block,cc\n";
    out << "\"" << name << "\"," << sm_count << "," << clock_rate_hz << "," << theoretical_gflops
        << "," << mem_bandwidth_gb << "," << warp_size << "," << shared_mem_per_block << ","
        << max_threads_per_block << "," << regs_per_block << "," << major_cc << "." << minor_cc
        << "\n";
    out.close();

    // Console summary
    std::cout << "=================================================================\n";
    std::cout << "Device info written to device_info.csv\n";
    std::cout << "Name: " << name << "\n";
    std::cout << "Compute Capability: " << major_cc << "." << minor_cc << "\n";
    std::cout << "SM count: " << sm_count << "\n";
    std::cout << "Clock: " << clock_rate_hz / 1e9 << " GHz\n";
    std::cout << "Theoretical FP32 peak: " << theoretical_gflops << " GFLOPs\n";
    std::cout << "Memory Bandwidth: " << mem_bandwidth_gb << " GB/s\n";
    std::cout << "Warp Size: " << warp_size << "\n";
    std::cout << "Shared Mem / Block: " << shared_mem_per_block / 1024.0 << " KB\n";
    std::cout << "Max Threads / Block: " << max_threads_per_block << "\n";
    std::cout << "Registers / Block: " << regs_per_block << "\n";
    std::cout << "=================================================================\n";
}