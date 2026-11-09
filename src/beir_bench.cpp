#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif

#include "../include/vrag.hpp"

struct BeirQuery {
    std::uint32_t query_id;
    std::string text;
    std::vector<float> vector;
    std::unordered_map<std::uint32_t, std::uint32_t> qrels; // doc_id -> relevance grade (>0)
};

static bool load_beir_queries(const std::string& path, std::size_t expected_dim, std::vector<BeirQuery>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open queries file: " << path << "\n";
        return false;
    }

    std::uint32_t num_queries = 0;
    std::uint32_t dim = 0;
    ifs.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
    ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!ifs || dim != expected_dim) {
        std::cerr << "Query dimension mismatch: expected " << expected_dim << ", got " << dim << "\n";
        return false;
    }

    out.clear();
    out.reserve(num_queries);

    for (std::uint32_t i = 0; i < num_queries; ++i) {
        BeirQuery q;
        std::uint32_t text_len = 0;
        std::uint32_t num_qrels = 0;

        ifs.read(reinterpret_cast<char*>(&q.query_id), sizeof(q.query_id));
        ifs.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
        q.text.resize(text_len);
        if (text_len > 0) {
            ifs.read(q.text.data(), text_len);
        }

        q.vector.resize(dim);
        if (dim > 0) {
            ifs.read(reinterpret_cast<char*>(q.vector.data()), dim * sizeof(float));
        }

        ifs.read(reinterpret_cast<char*>(&num_qrels), sizeof(num_qrels));
        q.qrels.reserve(num_qrels);
        for (std::uint32_t j = 0; j < num_qrels; ++j) {
            std::uint32_t doc_id = 0;
            std::uint32_t rel = 0;
            ifs.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
            ifs.read(reinterpret_cast<char*>(&rel), sizeof(rel));
            if (rel > 0) {
                q.qrels[doc_id] = rel;
            }
        }

        if (!ifs) {
            return false;
        }
        if (!q.qrels.empty()) {
            out.push_back(std::move(q));
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <corpus.bin> <queries.bin> [dataset_name]\n";
        return 1;
    }

    const std::string corpus_path = argv[1];
    const std::string queries_path = argv[2];
    const std::string dataset_name = (argc >= 4) ? argv[3] : "BEIR";

    // 读取 corpus.bin 头获取维度
    std::uint32_t doc_count = 0;
    std::uint32_t dim = 0;
    {
        std::ifstream ifs(corpus_path, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Cannot open corpus file: " << corpus_path << "\n";
            return 1;
        }
        ifs.read(reinterpret_cast<char*>(&doc_count), sizeof(doc_count));
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    }

    vrag::VragEngine engine(dim, 64ULL * 1024 * 1024);

    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!engine.load_from_binary(corpus_path)) {
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    std::vector<BeirQuery> queries;
    if (!load_beir_queries(queries_path, dim, queries) || queries.empty()) {
        std::cerr << "Failed to load queries from " << queries_path << "\n";
        return 1;
    }

    // 预热 20 次
    const std::size_t warmup = std::min<std::size_t>(20, queries.size());
    for (std::size_t i = 0; i < warmup; ++i) {
        auto res = engine.search(queries[i].text, queries[i].vector, 10);
        (void)res;
    }

    // 正式评测 (检索 Top-100 以同时计算 NDCG@10, MRR@10, Recall@10, Recall@100, 以及 Top-10 延迟)
    std::vector<double> latencies_top10_ms;
    latencies_top10_ms.reserve(queries.size());

    double sum_ndcg_10 = 0.0;
    double sum_mrr_10 = 0.0;
    double sum_recall_10 = 0.0;
    double sum_recall_100 = 0.0;

    for (const auto& q : queries) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto hits_10 = engine.search(q.text, q.vector, 10);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_top10_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto hits_100 = engine.search(q.text, q.vector, 100);

        // 1. 计算 DCG@10 与 IDCG@10 (标准 TREC / BEIR 公式: (2^rel - 1) / log2(rank + 1))
        double dcg_10 = 0.0;
        double mrr_10 = 0.0;
        std::size_t rel_found_10 = 0;

        for (std::size_t r = 0; r < hits_10.size(); ++r) {
            auto it = q.qrels.find(hits_10[r].doc_id);
            if (it != q.qrels.end() && it->second > 0) {
                const double gain = std::pow(2.0, static_cast<double>(it->second)) - 1.0;
                dcg_10 += gain / std::log2(static_cast<double>(r) + 2.0);
                ++rel_found_10;
                if (mrr_10 == 0.0) {
                    mrr_10 = 1.0 / static_cast<double>(r + 1);
                }
            }
        }

        std::vector<std::uint32_t> ideal_rels;
        ideal_rels.reserve(q.qrels.size());
        for (const auto& [doc_id, rel] : q.qrels) {
            if (rel > 0) {
                ideal_rels.push_back(rel);
            }
        }
        std::sort(ideal_rels.begin(), ideal_rels.end(), std::greater<std::uint32_t>());

        double idcg_10 = 0.0;
        const std::size_t ideal_k = std::min<std::size_t>(10, ideal_rels.size());
        for (std::size_t r = 0; r < ideal_k; ++r) {
            const double gain = std::pow(2.0, static_cast<double>(ideal_rels[r])) - 1.0;
            idcg_10 += gain / std::log2(static_cast<double>(r) + 2.0);
        }

        if (idcg_10 > 0.0) {
            sum_ndcg_10 += dcg_10 / idcg_10;
        }
        sum_mrr_10 += mrr_10;

        const std::size_t total_rel = ideal_rels.size();
        if (total_rel > 0) {
            sum_recall_10 += static_cast<double>(rel_found_10) / static_cast<double>(total_rel);

            std::size_t rel_found_100 = 0;
            for (const auto& h : hits_100) {
                if (q.qrels.find(h.doc_id) != q.qrels.end()) {
                    ++rel_found_100;
                }
            }
            sum_recall_100 += static_cast<double>(rel_found_100) / static_cast<double>(total_rel);
        }
    }

    std::sort(latencies_top10_ms.begin(), latencies_top10_ms.end());
    const std::size_t nq = queries.size();
    const double avg_ms = std::accumulate(latencies_top10_ms.begin(), latencies_top10_ms.end(), 0.0) /
                          static_cast<double>(nq);
    const double p50_ms = latencies_top10_ms[nq * 50 / 100];
    const double p95_ms = latencies_top10_ms[nq * 95 / 100];
    const double p99_ms = latencies_top10_ms[std::min(nq - 1, nq * 99 / 100)];
    const double qps = (avg_ms > 0.0) ? (1000.0 / avg_ms) : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "========================================================\n";
    std::cout << " Standard BEIR Benchmark Report: " << dataset_name << "\n";
    std::cout << "========================================================\n";
    std::cout << " Corpus Size      : " << engine.document_count() << " docs\n";
    std::cout << " Test Queries     : " << nq << " queries (with ground-truth qrels)\n";
    std::cout << " Index Build Time : " << std::setprecision(2) << load_ms << " ms ("
              << (engine.document_count() * 1000.0 / load_ms) << " docs/s)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::setprecision(4);
    std::cout << " [Accuracy Metrics]\n";
    std::cout << "   NDCG@10        : " << (sum_ndcg_10 / nq) << "\n";
    std::cout << "   MRR@10         : " << (sum_mrr_10 / nq) << "\n";
    std::cout << "   Recall@10      : " << (sum_recall_10 / nq) << "\n";
    std::cout << "   Recall@100     : " << (sum_recall_100 / nq) << "\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::setprecision(3);
    std::cout << " [Latency & Throughput (Top-10)]\n";
    std::cout << "   Avg Latency    : " << avg_ms << " ms\n";
    std::cout << "   P50 Latency    : " << p50_ms << " ms\n";
    std::cout << "   P95 Latency    : " << p95_ms << " ms\n";
    std::cout << "   P99 Latency    : " << p99_ms << " ms\n";
    std::cout << "   Throughput     : " << std::setprecision(1) << qps << " QPS\n";
    std::cout << "========================================================\n";

    return 0;
}
