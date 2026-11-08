#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include "../include/arena.hpp"
#include "../include/simd_vector.hpp"

using namespace vrag;

// 编译器防死码消除屏障
static void do_not_optimize(float value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

int main() {
    constexpr std::size_t Dimension = 768;      // 常见 Embedding 维度
    constexpr std::size_t DocCount = 10000;       // 1 万篇文档
    constexpr std::size_t WarmupIters = 50;
    constexpr std::size_t BenchmarkIters = 200;

    std::cout << "====================================================\n";
    std::cout << " AVX2 + FMA Dense Vector Dot Product Benchmark      \n";
    std::cout << " Target: " << DocCount << " docs, " << Dimension << "-D float vectors\n";
    std::cout << " Total memory: ~" << (DocCount * Dimension * sizeof(float)) / (1024 * 1024) << " MiB\n";
    std::cout << "====================================================\n";

    // 1. 在 Arena 中连续分配 10,000 个 32 字节对齐向量
    // 需要空间: 10,000 * 768 * 4 字节 ≈ 29.3 MiB
    Arena arena(32 * 1024 * 1024);

    std::vector<std::span<float>> docs;
    docs.reserve(DocCount);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < DocCount; ++i) {
        auto vec = arena.allocate_vector(Dimension);
        for (std::size_t d = 0; d < Dimension; ++d) {
            vec[d] = dist(rng);
        }
        SimdVector::normalize(vec);
        docs.push_back(vec);
    }

    // 分配一个查询向量 Query
    auto query_vec = arena.allocate_vector(Dimension);
    for (std::size_t d = 0; d < Dimension; ++d) {
        query_vec[d] = dist(rng);
    }
    SimdVector::normalize(query_vec);

    // 2. 预热 Cache
    for (std::size_t iter = 0; iter < WarmupIters; ++iter) {
        for (std::size_t i = 0; i < DocCount; ++i) {
            float score = SimdVector::dot_product_avx2(query_vec, docs[i]);
            do_not_optimize(score);
        }
    }

    // 3. 正式压测
    auto start = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < BenchmarkIters; ++iter) {
        for (std::size_t i = 0; i < DocCount; ++i) {
            float score = SimdVector::dot_product_avx2(query_vec, docs[i]);
            do_not_optimize(score);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    double total_queries = BenchmarkIters;
    double total_dot_products = BenchmarkIters * DocCount;
    double avg_scan_time_ms = elapsed.count() / total_queries;
    double time_per_vector_ns = (elapsed.count() * 1e6) / total_dot_products;
    double vectors_per_sec = total_dot_products / (elapsed.count() / 1000.0);

    // 计算 GFLOPS (每个 768 维点积包含 768 次乘法 + 768 次加法 = 1536 次浮点操作)
    double gflops = (vectors_per_sec * Dimension * 2) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[Results - Single Core (Flat Scan)]:\n";
    std::cout << "10k docs full scan latency : " << avg_scan_time_ms << " ms\n";
    std::cout << "Time per 768-D dot product : " << time_per_vector_ns << " ns\n";
    std::cout << "Throughput                 : " << vectors_per_sec << " vectors/sec\n";
    std::cout << "Compute throughput         : " << gflops << " GFLOPS\n";
    std::cout << "====================================================\n";

    return 0;
}
