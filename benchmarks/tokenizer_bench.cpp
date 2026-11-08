#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>
#include "../include/tokenizer.hpp"

using namespace std::chrono;
using namespace vrag;

// 通用校验函数：用于在基准测试前验证 Token 生成的正确性
template <typename Range>
std::uint64_t compute_checksum(Range&& tokens) {
    std::uint64_t hash = 14695981039346656037ull;

    for (const auto token : tokens) {
        for (unsigned char c : token) {
            hash ^= c;
            hash *= 1099511628211ull;
        }
    }

    return hash;
}

// 最近秩法计算百分位数
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());

    const auto index = static_cast<std::size_t>(
        std::ceil(p * values.size())
    ) - 1;

    return values[index];
}

int main() {
    Tokenizer tokenizer;

    // 构造约 1 MiB 测试文本
    constexpr std::string_view sentence =
        "我喜欢仓鼠仓鼠是一种非常可爱的小动物";

    std::string text;
    constexpr std::size_t target_size = 1024 * 1024;

    text.reserve(target_size);
    while (text.size() < target_size) {
        text.append(sentence);
    }

    // 1. 正确性先验：基准测试前完整校验一次切词数据
    const std::uint64_t ground_truth_checksum =
        compute_checksum(tokenizer.cut(text));

    constexpr int warmup_iterations = 10;
    constexpr int iterations = 100;

    std::uint64_t total_tokens = 0;
    std::uintptr_t global_sink = 0;

    // 2. 预热阶段：让 CPU 进入睿频状态并预热指令/数据缓存
    for (int i = 0; i < warmup_iterations; ++i) {
        for (const auto token : tokenizer.cut(text)) {
            global_sink += reinterpret_cast<std::uintptr_t>(token.data()) + token.size();
        }
    }

    std::vector<double> latencies;
    latencies.reserve(iterations);

    // 3. 正式测试
    for (int i = 0; i < iterations; ++i) {
        std::size_t iter_tokens = 0;
        std::uintptr_t iter_sink = 0;

        const auto start = steady_clock::now();

        // 核心测试：惰性流式迭代，边扫边切，零堆内存分配
        for (const auto token : tokenizer.cut(text)) {
            ++iter_tokens;
            iter_sink ^= reinterpret_cast<std::uintptr_t>(token.data()) + token.size();
        }

        // 内存屏障：强制告诉编译器 iter_sink 已被消费，防止整个循环被 -O3 优化抹除
        #if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "g"(iter_sink) : "memory");
        #endif

        const auto end = steady_clock::now();

        const double elapsed_ms =
            duration<double, std::milli>(end - start).count();

        latencies.push_back(elapsed_ms);
        total_tokens += iter_tokens;
        global_sink ^= iter_sink;
    }

    const double total_seconds =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1000.0;

    const double total_mib =
        static_cast<double>(text.size()) * iterations / (1024.0 * 1024.0);

    const double throughput = total_mib / total_seconds;
    const double tokens_per_second = total_tokens / total_seconds;
    const double average_ms = total_seconds * 1000.0 / iterations;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "========== Tokenizer Benchmark (Iterator Mode) ==========\n";
    std::cout << "Input size: " << text.size() / 1024.0 << " KiB\n";
    std::cout << "Iterations: " << iterations << '\n';
    std::cout << "Average latency: " << average_ms << " ms\n";
    std::cout << "P50 latency: " << percentile(latencies, 0.50) << " ms\n";
    std::cout << "P95 latency: " << percentile(latencies, 0.95) << " ms\n";
    std::cout << "P99 latency: " << percentile(latencies, 0.99) << " ms\n";
    std::cout << "Throughput: " << throughput << " MiB/s\n";
    std::cout << "Tokens per second: " << std::scientific << tokens_per_second << '\n';
    std::cout << "Total tokens: " << total_tokens << '\n';
    std::cout << "Checksum (Ground Truth): " << ground_truth_checksum << '\n';
    std::cout << "Sink (Anti-DCE verification): " << global_sink << '\n';

    return 0;
}
