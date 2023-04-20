cmake_minimum_required(VERSION 3.13)

project(sp_gemm CUDA CXX C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

#set(CMAKE_CXX_STANDARD           17)
#set(CMAKE_CXX_STANDARD_REQUIRED  ON)
#set(CMAKE_CXX_EXTENSIONS         OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")
add_definitions(-DGPU_CC=${CUDA_ARCHS})

set(CUSPARSELT_PATH "/usr/local/cuda-11.5/lib64/")
set(CUDA_TOOLKIT_PATH "/usr/local/cuda-11.5/")

include_directories(include/sputnik)
add_subdirectory(include/sputnik)
add_subdirectory(include/spatha/block_sparse)
add_subdirectory(include/CLASP)
add_subdirectory(src)
