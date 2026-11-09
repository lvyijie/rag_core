cmake_minimum_required(VERSION 3.15)
project(rag_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /utf-8)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
    add_compile_options(-O3 -march=native)
endif()

# Header-Only 核心库（统一继承 OpenMP 与编译优化）
add_library(rag_core INTERFACE)
target_include_directories(rag_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(rag_core INTERFACE OpenMP::OpenMP_CXX)

# 仅在作为顶层项目构建时编译测试、Benchmark 与 CLI 工具
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()

    add_executable(tokenizer_test tests/tokenizer_test.cpp)
    target_link_libraries(tokenizer_test PRIVATE rag_core)
    add_test(NAME tokenizer_test COMMAND tokenizer_test)

    add_executable(arena_test tests/arena_test.cpp)
    target_link_libraries(arena_test PRIVATE rag_core)
    add_test(NAME arena_test COMMAND arena_test)

    add_executable(simd_test tests/simd_test.cpp)
    target_link_libraries(simd_test PRIVATE rag_core)
    add_test(NAME simd_test COMMAND simd_test)

    add_executable(index_test tests/index_test.cpp)
    target_link_libraries(index_test PRIVATE rag_core)
    add_test(NAME index_test COMMAND index_test)

    add_executable(vrag_test tests/vrag_test.cpp)
    target_link_libraries(vrag_test PRIVATE rag_core)
    add_test(NAME vrag_test COMMAND vrag_test)

    add_executable(real_corpus_test tests/real_corpus_test.cpp)
    target_link_libraries(real_corpus_test PRIVATE rag_core)

    add_executable(eval_accuracy tests/eval_accuracy.cpp)
    target_link_libraries(eval_accuracy PRIVATE rag_core)

    add_executable(tokenizer_bench benchmarks/tokenizer_bench.cpp)
    target_link_libraries(tokenizer_bench PRIVATE rag_core)

    add_executable(simd_bench benchmarks/simd_bench.cpp)
    target_link_libraries(simd_bench PRIVATE rag_core)

    add_executable(vrag_bench benchmarks/vrag_bench.cpp)
    target_link_libraries(vrag_bench PRIVATE rag_core)

    add_executable(cli_search src/cli_search.cpp)
    target_link_libraries(cli_search PRIVATE rag_core)

    add_executable(bench_search src/bench_search.cpp)
    target_link_libraries(bench_search PRIVATE rag_core)

    add_executable(beir_bench src/beir_bench.cpp)
    target_link_libraries(beir_bench PRIVATE rag_core)
endif()
