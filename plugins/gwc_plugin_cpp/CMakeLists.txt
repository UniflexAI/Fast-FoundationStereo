cmake_minimum_required(VERSION 3.10)
project(GwcVolumePlugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

# Detect platform
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CUDA_ARCHITECTURES "87")
    message(STATUS "Building for Jetson (ARM) - CUDA arch 87")
else()
    set(CMAKE_CUDA_ARCHITECTURES "86;89")
    message(STATUS "Building for x86_64 - CUDA arch 86,89")
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA include
if(EXISTS "/usr/local/cuda/include")
    include_directories(/usr/local/cuda/include)
endif()

# TensorRT include
if(EXISTS "/usr/src/tensorrt/include")
    include_directories(/usr/src/tensorrt/include)
endif()
if(EXISTS "/usr/include/aarch64-linux-gnu")
    include_directories(/usr/include/aarch64-linux-gnu)
endif()
if(EXISTS "/usr/include/NvInfer.h")
    include_directories(/usr/include)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(GwcVolumePlugin SHARED
    GwcVolumePlugin.cpp
    build_gwc_kernel.cu
)

target_link_libraries(GwcVolumePlugin nvinfer cudart)

set_property(TARGET GwcVolumePlugin PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GwcVolumePlugin PROPERTY POSITION_INDEPENDENT_CODE ON)
