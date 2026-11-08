#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#if defined(_WIN32)
#include <windows.h>
#endif
#include "../include/vrag.hpp"

using namespace vrag;

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <corpus.bin> [压测查询轮数(默认100)]\n";
        return 1;
    }

    constexpr std::size_t Dim = 1024;
    std::string bin_path = argv[1];
    int num_queries = (argc >= 3) ? std::stoi(argv[2]) : 100;

    std::cout << "=== VRAG 混合检索内核性能基准测试 ===\n";
    std::cout << "正在载入语料库: " << bin_path << " ...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    VragEngine engine(Dim);
    if (!engine.load_from_binary(bin_path)) {
        std::cerr << "[-] 加载文件失败！\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[+] 加载完成！文档总数: " << engine.total_docs() 
              << "，耗时: " << load_sec << " 秒\n\n";

    // 构造真实的混合测试 Query（覆盖简繁体四大名著经典桥段）
    std::vector<std::string> test_queries = {
        "贾宝玉神游太虚幻境",
        "冷子兴演说荣国府衰败",
        "诸葛亮草船借箭孔明奇计",
        "孙悟空大闹天宫齐天大圣",
        "鲁智深倒拔垂杨柳武松打虎"
    };

    // 构造 1024 维归一化查询向量
    std::vector<float> q_vec(Dim, 0.03125f); // 1 / sqrt(1024) 归一化值

    std::cout << "[*] 开始执行 " << num_queries << " 次双路混合检索压测 (Top 5)...\n";

    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_queries);

    for (int i = 0; i < num_queries; ++i) {
        const auto& q_str = test_queries[i % test_queries.size()];
        
        auto start = std::chrono::high_resolution_clock::now();
        auto hits = engine.search(q_str, q_vec, 5);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        latencies_ms.push_back(elapsed);
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    double avg = sum / latencies_ms.size();
    double p50 = latencies_ms[latencies_ms.size() * 0.50];
    double p95 = latencies_ms[latencies_ms.size() * 0.95];
    double p99 = latencies_ms[latencies_ms.size() * 0.99];

    std::cout << "\n================ 压测报告 ================\n";
    std::cout << "文档总容量   : " << engine.total_docs() << " 篇切片\n";
    std::cout << "总请求次数   : " << num_queries << " 次\n";
    std::cout << "平均延迟     : " << avg << " ms\n";
    std::cout << "P50 延迟     : " << p50 << " ms\n";
    std::cout << "P95 延迟     : " << p95 << " ms\n";
    std::cout << "P99 延迟     : " << p99 << " ms\n";
    std::cout << "吞吐量 (QPS) : " << (1000.0 / avg) << " req/s\n";
    std::cout << "==========================================\n";

    return 0;
}
