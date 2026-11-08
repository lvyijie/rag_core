#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../include/vrag.hpp"

using namespace vrag;

// 防死码消除内存屏障
static void do_not_optimize(const std::vector<HitResult>& results) {
    asm volatile("" : : "g"(&results) : "memory");
}

int main() {
    constexpr std::size_t Dimension = 768;
    constexpr std::size_t DocCount = 10000;      // 1 万篇知识库切片
    constexpr std::size_t WarmupQueries = 50;
    constexpr std::size_t BenchmarkQueries = 500; // 测试 500 次端到端检索

    std::cout << "====================================================\n";
    std::cout << " VragEngine Full-Pipeline End-to-End Benchmark       \n";
    std::cout << " Corpus: " << DocCount << " docs | Vector Dim: " << Dimension << "-D\n";
    std::cout << " Queries to run: " << BenchmarkQueries << " (with " << WarmupQueries << " warmup)\n";
    std::cout << "====================================================\n";

    // 1. 初始化引擎 (分配 64MB Arena)
    VragEngine engine(Dimension, 64 * 1024 * 1024);

    // 2. 构造模拟知识库文本与向量
    std::vector<std::string> sample_texts = {
        "三门峡大天鹅国家级自然保护区每年冬季吸引成千上万只大天鹅栖息越冬。",
        "计算机体系结构中乱序执行和分支预测是提升流水线指令吞吐的关键技术。",
        "仓鼠是一种小型啮齿类宠物，夜间活跃度高，主要以种子坚果为食。",
        "现代操作系统通过虚拟内存管理、多级页表以及TLB快表加速地址转换。",
        "开源软件项目依赖社区协作与持续集成，代码质量与安全审计至关重要。"
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "Indexing " << DocCount << " documents into engine...\n";
    auto index_start = std::chrono::steady_clock::now();

    std::vector<float> tmp_vec(Dimension);
    for (std::uint32_t i = 0; i < DocCount; ++i) {
        for (std::size_t d = 0; d < Dimension; ++d) {
            tmp_vec[d] = dist(rng);
        }
        std::string doc_text = sample_texts[i % sample_texts.size()] + " 编号:" + std::to_string(i);
        engine.add_document(i, doc_text, tmp_vec);
    }

    auto index_end = std::chrono::steady_clock::now();
    double index_time_ms = std::chrono::duration<double, std::milli>(index_end - index_start).count();
    std::cout << "Index build completed in: " << index_time_ms << " ms ("
              << (DocCount * 1000.0) / index_time_ms << " docs/sec)\n";

    // 3. 准备多样化的 Query 集合
    std::vector<std::string> queries = {
        "大天鹅栖息越冬",
        "操作系统虚拟内存与页表",
        "啮齿类小仓鼠",
        "计算机体系结构流水线",
        "开源社区代码审计"
    };

    std::vector<std::vector<float>> query_vectors(BenchmarkQueries, std::vector<float>(Dimension));
    for (auto& qv : query_vectors) {
        for (std::size_t d = 0; d < Dimension; ++d) {
            qv[d] = dist(rng);
        }
    }

    // 4. 预热引擎与 Cache
    for (std::size_t i = 0; i < WarmupQueries; ++i) {
        auto results = engine.search(queries[i % queries.size()], query_vectors[i], 10);
        do_not_optimize(results);
    }

    // 5. 记录每次检索延迟
    std::vector<double> latencies_ms;
    latencies_ms.reserve(BenchmarkQueries);

    for (std::size_t i = 0; i < BenchmarkQueries; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        
        auto results = engine.search(queries[i % queries.size()], query_vectors[i], 10);
        do_not_optimize(results);

        auto t1 = std::chrono::steady_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // 6. 统计延迟指标 (Avg, P50, P90, P99)
    std::sort(latencies_ms.begin(), latencies_ms.end());
    double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    double avg = sum / BenchmarkQueries;
    double p50 = latencies_ms[static_cast<std::size_t>(BenchmarkQueries * 0.50)];
    double p90 = latencies_ms[static_cast<std::size_t>(BenchmarkQueries * 0.90)];
    double p99 = latencies_ms[static_cast<std::size_t>(BenchmarkQueries * 0.99)];
    double qps = 1000.0 / avg;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n====================================================\n";
    std::cout << " [E2E Retrieval Latency & Throughput Results]       \n";
    std::cout << " Avg Latency  : " << avg << " ms\n";
    std::cout << " P50 Latency  : " << p50 << " ms\n";
    std::cout << " P90 Latency  : " << p90 << " ms\n";
    std::cout << " P99 Latency  : " << p99 << " ms\n";
    std::cout << " Throughput   : " << qps << " QPS (queries/sec)\n";
    std::cout << "====================================================\n";

    return 0;
}
