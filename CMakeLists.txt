cmake_minimum_required(VERSION 3.20)
project(attention_implementations_with_CUDA LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75 80)

include_directories(include)

file(
    GLOB ATTENTION_SOURCES
    src/attention/*.cu
)
file(
    GLOB BENCHMARK_SOURCES
    src/allocator.cpp
    src/benchmark.cpp
)

add_library(benchmark_utils STATIC ${BENCHMARK_SOURCES})
target_link_libraries(benchmark_utils PRIVATE CUDA::curand CUDA::cudart)
add_library(attention STATIC ${ATTENTION_SOURCES})
target_link_libraries(attention PRIVATE CUDA::cublas CUDA::cudart)


add_executable(run_entrypoint src/main.cu)
set_target_properties(run_entrypoint PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(run_entrypoint PRIVATE attention benchmark_utils)
add_definitions(-Wno-deprecated-gpu-targets)