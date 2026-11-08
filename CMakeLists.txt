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

# 3. 内存池对齐与分配测试
add_executable(arena_test tests/arena_test.cpp)
target_link_libraries(arena_test PRIVATE rag_core)

# 4. AVX2 向量算子精度测试
add_executable(simd_test tests/simd_test.cpp)
target_link_libraries(simd_test PRIVATE rag_core)

# 5. AVX2 向量算子吞吐 Benchmark
add_executable(simd_bench benchmarks/simd_bench.cpp)
target_link_libraries(simd_bench PRIVATE rag_core)

# 6. 倒排索引与 BM25 检索功能测试
add_executable(index_test tests/index_test.cpp)
target_link_libraries(index_test PRIVATE rag_core)

# 7. 混合检索端到端引擎测试
add_executable(vrag_test tests/vrag_test.cpp)
target_link_libraries(vrag_test PRIVATE rag_core)

# 8. 全链路端到端性能 Benchmark
add_executable(vrag_bench benchmarks/vrag_bench.cpp)
target_link_libraries(vrag_bench PRIVATE rag_core)

# 9. 真实语料库端到端检索准确率测试
add_executable(real_corpus_test tests/real_corpus_test.cpp)
target_link_libraries(real_corpus_test PRIVATE rag_core)

# 10. 交互式 CLI 搜索工具
add_executable(cli_search src/cli_search.cpp)
target_link_libraries(cli_search PRIVATE rag_core)

add_executable(bench_search src/bench_search.cpp)

find_package(OpenMP REQUIRED)

# 为你的核心库或所有可执行目标链接 OpenMP
# 如果你使用的是 target_link_libraries：
target_link_libraries(bench_search PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(cli_search PRIVATE OpenMP::OpenMP_CXX)

# 确保开启了 O3 优化与 native 指令集
add_compile_options(-O3 -march=native -fopenmp)

add_executable(eval_accuracy tests/eval_accuracy.cpp)
target_link_libraries(eval_accuracy PRIVATE OpenMP::OpenMP_CXX)
