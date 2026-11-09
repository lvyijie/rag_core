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

struct EvalSummary {
    std::string mode_name;
    double ndcg_10{0.0};
    double mrr_10{0.0};
    double recall_10{0.0};
    double recall_100{0.0};
    double avg_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
    double qps{0.0};
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

static EvalSummary evaluate_mode(
    const vrag::VragEngine& engine,
    const std::vector<BeirQuery>& queries,
    const std::string& mode_name,
    bool use_sparse,
    bool use_dense,
    float rrf_k = 60.0f
) {
    const std::vector<float> empty_vec;
    const std::size_t warmup = std::min<std::size_t>(20, queries.size());
    for (std::size_t i = 0; i < warmup; ++i) {
        std::string_view qt = use_sparse ? std::string_view(queries[i].text) : std::string_view{};
        std::span<const float> qv = use_dense ? std::span<const float>(queries[i].vector) : std::span<const float>(empty_vec);
        auto res = engine.search(qt, qv, 10, rrf_k);
        (void)res;
    }

    std::vector<double> latencies_top10_ms;
    latencies_top10_ms.reserve(queries.size());

    double sum_ndcg_10 = 0.0;
    double sum_mrr_10 = 0.0;
    double sum_recall_10 = 0.0;
    double sum_recall_100 = 0.0;

    for (const auto& q : queries) {
        std::string_view qt = use_sparse ? std::string_view(q.text) : std::string_view{};
        std::span<const float> qv = use_dense ? std::span<const float>(q.vector) : std::span<const float>(empty_vec);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto hits_10 = engine.search(qt, qv, 10, rrf_k);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_top10_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto hits_100 = engine.search(qt, qv, 100, rrf_k);

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
    EvalSummary s;
    s.mode_name = mode_name;
    s.ndcg_10 = sum_ndcg_10 / static_cast<double>(nq);
    s.mrr_10 = sum_mrr_10 / static_cast<double>(nq);
    s.recall_10 = sum_recall_10 / static_cast<double>(nq);
    s.recall_100 = sum_recall_100 / static_cast<double>(nq);
    s.avg_ms = std::accumulate(latencies_top10_ms.begin(), latencies_top10_ms.end(), 0.0) / static_cast<double>(nq);
    s.p50_ms = latencies_top10_ms[nq * 50 / 100];
    s.p95_ms = latencies_top10_ms[nq * 95 / 100];
    s.p99_ms = latencies_top10_ms[std::min(nq - 1, nq * 99 / 100)];
    s.qps = (s.avg_ms > 0.0) ? (1000.0 / s.avg_ms) : 0.0;
    return s;
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

    bool has_dense_vectors = false;
    for (float v : queries[0].vector) {
        if (v != 0.0f) {
            has_dense_vectors = true;
            break;
        }
    }

    std::cout << "====================================================================================================\n";
    std::cout << " Standard BEIR Benchmark & Ablation Report: " << dataset_name << "\n";
    std::cout << " Corpus: " << engine.document_count() << " docs (dim=" << dim << ") | Test Queries: " << queries.size()
              << " | Index Build: " << std::fixed << std::setprecision(2) << load_ms << " ms\n";
    std::cout << "====================================================================================================\n";
    std::cout << std::left << std::setw(28) << "Retrieval Mode"
              << " | " << std::right << std::setw(8) << "NDCG@10"
              << " | " << std::setw(8) << "MRR@10"
              << " | " << std::setw(9) << "Recall@10"
              << " | " << std::setw(10) << "Recall@100"
              << " | " << std::setw(22) << "Latency Avg/P50/P99"
              << " | " << std::setw(10) << "QPS\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    std::vector<EvalSummary> summaries;
    summaries.push_back(evaluate_mode(engine, queries, "1. BM25 Only (Sparse)", true, false));
    if (has_dense_vectors) {
        summaries.push_back(evaluate_mode(engine, queries, "2. Dense SQ8 Only (Vector)", false, true));
        summaries.push_back(evaluate_mode(engine, queries, "3. Hybrid (BM25+SQ8+RRF)", true, true));
    }

    for (const auto& s : summaries) {
        std::ostringstream lat_ss;
        lat_ss << std::fixed << std::setprecision(3) << s.avg_ms << "/" << s.p50_ms << "/" << s.p99_ms << " ms";
        std::cout << std::left << std::setw(28) << s.mode_name
                  << " | " << std::right << std::fixed << std::setprecision(4) << std::setw(8) << s.ndcg_10
                  << " | " << std::setw(8) << s.mrr_10
                  << " | " << std::setw(9) << s.recall_10
                  << " | " << std::setw(10) << s.recall_100
                  << " | " << std::setw(22) << lat_ss.str()
                  << " | " << std::setprecision(1) << std::setw(10) << s.qps << "\n";
    }
    std::cout << "====================================================================================================\n";

    return 0;
}
