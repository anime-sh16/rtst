cmake_minimum_required(VERSION 3.19)
project(rtst)

set(CMAKE_CXX_STANDARD 17)

# ExecuTorch source (local clone, v1.3.0)
set(EXECUTORCH_PATH "/Users/animesh/Developer/open-source/executorch")

# ExecuTorch build options
set(EXECUTORCH_BUILD_XNNPACK                ON  CACHE BOOL "XNNPACK delegate backend" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER  ON  CACHE BOOL "FileDataLoader for .pte files" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE       ON  CACHE BOOL "High-level Module API" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "Named data map extension" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED      ON  CACHE BOOL "Portable CPU kernels (fallback for non-delegated ops)" FORCE)
set(EXECUTORCH_XNNPACK_ENABLE_KLEIDI        OFF CACHE BOOL "Disabled — ARM64 path resolution bug" FORCE)

# EXCLUDE_FROM_ALL: only build targets that rtst actually depends on
add_subdirectory(${EXECUTORCH_PATH} executorch_build EXCLUDE_FROM_ALL)

# OpenCV (image I/O for infer mode)
find_package(OpenCV REQUIRED)

# Target: rtst
add_executable(rtst main.cpp)

target_include_directories(rtst PRIVATE
    ${EXECUTORCH_PATH}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rtst PRIVATE
    executorch
    extension_module
    extension_data_loader
    xnnpack_backend
    portable_ops_lib
    ${OpenCV_LIBS}
)
