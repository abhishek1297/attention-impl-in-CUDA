cmake_minimum_required(VERSION 3.18)
project(attention_implementations_with_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

include_directories(include)

file(
    GLOB ATTENTION_SOURCES
    src/attention/*.cu
    # src/utils.cu
)

add_executable(cuda_attention src/main.cu ${ATTENTION_SOURCES})
set_target_properties(cuda_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
