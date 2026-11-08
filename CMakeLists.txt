cmake_minimum_required(VERSION 3.15)
project(rag_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 跑 benchmark 必须开启 Release 与架构级优化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Header-Only 核心库
add_library(rag_core INTERFACE)
target_include_directories(rag_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 1. 功能测试可执行文件
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE rag_core)

# 2. Benchmark 基准测试可执行文件
add_executable(tokenizer_bench benchmarks/tokenizer_bench.cpp)
target_link_libraries(tokenizer_bench PRIVATE rag_core)
