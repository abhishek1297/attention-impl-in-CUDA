#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#define CUDA_CHECK() cudaCheck(__FILE__, __LINE__)

inline void cudaCheck(const char *file, int line) {
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

    // Core characteristics
    std::string name(prop.name);
    int sm_count = prop.multiProcessorCount;
    double clock_rate_hz = static_cast<double>(prop.clockRate) * 1e3; // in Hz

    // FLOPs per SM per clock (FP32)
    // Usually 128 FMA units per SM → 256 FLOPs per cycle
    const int fma_per_cycle_per_sm = 128;
    double theoretical_flops =
        static_cast<double>(sm_count) * fma_per_cycle_per_sm * 2.0 * clock_rate_hz;
    double theoretical_gflops = theoretical_flops / 1e9;

    // Memory bandwidth (optional, useful for roofline)
    double mem_bandwidth_gb = 2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) / 1e9;

    // Compute filename
    std::ofstream out("benchmarks/device_info.csv");
    out << "name,sm_count,clock_rate_hz,theoretical_gflops,memory_bandwidth_gb\n";
    out << "\"" << name << "\"," << sm_count << "," << clock_rate_hz << "," << theoretical_gflops
        << "," << mem_bandwidth_gb << "\n";
    out.close();

    std::cout << "=================================================================\n";
    std::cout << "Device info written to device_info.csv\n";
    std::cout << "Name: " << name << "\n";
    std::cout << "SM count: " << sm_count << "\n";
    std::cout << "Clock: " << clock_rate_hz / 1e9 << " GHz\n";
    std::cout << "Theoretical FP32 peak: " << theoretical_gflops << " GFLOPs\n";
    std::cout << "Memory Bandwidth: " << mem_bandwidth_gb << " GB/s\n";
    std::cout << "=================================================================\n";
}