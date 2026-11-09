#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

// 用于 P0 实验：同批 Embedding 的 FP32 精确检索参考基线
struct Fp32ReferenceCorpus {
    std::size_t dim{0};
    std::vector<std::uint32_t> doc_ids;
    std::vector<float> norm_vectors; // [num_docs * dim], L2-normalized FP32
    vrag::InvertedIndex bm25_index;

    bool load(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }
        std::uint32_t doc_count = 0;
        std::uint32_t file_dim = 0;
        ifs.read(reinterpret_cast<char*>(&doc_count), sizeof(doc_count));
        ifs.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
        if (!ifs) {
            return false;
        }
        dim = file_dim;
        doc_ids.resize(doc_count);
        norm_vectors.resize(static_cast<std::size_t>(doc_count) * dim);
        bm25_index.reserve_docs(doc_count);

        std::string text;
        for (std::uint32_t i = 0; i < doc_count; ++i) {
            std::uint32_t doc_id = 0;
            std::uint32_t text_len = 0;
            ifs.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
            ifs.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
            text.resize(text_len);
            if (text_len > 0) {
                ifs.read(text.data(), text_len);
            }
            float* vec_dst = norm_vectors.data() + static_cast<std::size_t>(i) * dim;
            if (dim > 0) {
                ifs.read(reinterpret_cast<char*>(vec_dst), dim * sizeof(float));
                vrag::SimdVector::normalize(std::span<float>(vec_dst, dim));
            }
            doc_ids[i] = doc_id;
            bm25_index.add_document(i, text);
        }
        return static_cast<bool>(ifs);
    }

    // 执行 FP32 精确向量检索（或 BM25 + FP32 RRF 混合检索）
    std::vector<vrag::SearchHit> search_fp32(
        std::string_view query_text,
        std::span<const float> query_vec,
        std::size_t top_k,
        bool use_sparse,
        float rrf_k = 60.0f
    ) const {
        const std::size_t n = doc_ids.size();
        std::vector<float> q_norm(query_vec.begin(), query_vec.end());
        vrag::SimdVector::normalize(q_norm);

        auto hit_greater = [](const vrag::SearchHit& a, const vrag::SearchHit& b) noexcept {
            return (a.score != b.score) ? (a.score > b.score) : (a.doc_id < b.doc_id);
        };

        const std::size_t limit = std::min<std::size_t>(n, vrag::VragEngine::RRF_RANK_LIMIT);
        std::vector<vrag::SearchHit> dense_hits;
        dense_hits.reserve(limit);
        float min_top = -1e30f;

        for (std::size_t i = 0; i < n; ++i) {
            const float* d_ptr = norm_vectors.data() + i * dim;
            float sim = vrag::SimdVector::dot_product_avx2(
                q_norm, std::span<const float>(d_ptr, dim));
            const auto slot = static_cast<std::uint32_t>(i);
            if (dense_hits.size() < limit) {
                dense_hits.push_back(vrag::SearchHit{slot, sim});
                if (dense_hits.size() == limit) {
                    std::make_heap(dense_hits.begin(), dense_hits.end(), hit_greater);
                    min_top = dense_hits[0].score;
                }
            } else if (sim > min_top) {
                std::pop_heap(dense_hits.begin(), dense_hits.end(), hit_greater);
                dense_hits.back() = vrag::SearchHit{slot, sim};
                std::push_heap(dense_hits.begin(), dense_hits.end(), hit_greater);
                min_top = dense_hits[0].score;
            }
        }
        if (dense_hits.size() == limit) {
            std::sort_heap(dense_hits.begin(), dense_hits.end(), hit_greater);
        } else {
            std::sort(dense_hits.begin(), dense_hits.end(), hit_greater);
        }

        if (!use_sparse) {
            const std::size_t k_out = std::min(top_k, dense_hits.size());
            std::vector<vrag::SearchHit> out(k_out);
            for (std::size_t i = 0; i < k_out; ++i) {
                out[i] = vrag::SearchHit{doc_ids[dense_hits[i].doc_id], dense_hits[i].score};
            }
            return out;
        }

        std::vector<vrag::SearchHit> bm25_hits;
        bm25_index.search_bm25_into(query_text, vrag::VragEngine::RRF_RANK_LIMIT, bm25_hits);

        std::vector<float> rrf_scores(n, 0.0f);
        std::vector<std::uint32_t> candidates;
        candidates.reserve(vrag::VragEngine::RRF_RANK_LIMIT * 2);

        for (std::size_t r = 0; r < bm25_hits.size(); ++r) {
            std::uint32_t slot = bm25_hits[r].doc_id;
            if (rrf_scores[slot] == 0.0f) candidates.push_back(slot);
            rrf_scores[slot] += 1.0f / (rrf_k + static_cast<float>(r + 1));
        }
        for (std::size_t r = 0; r < dense_hits.size(); ++r) {
            std::uint32_t slot = dense_hits[r].doc_id;
            if (rrf_scores[slot] == 0.0f) candidates.push_back(slot);
            rrf_scores[slot] += 1.0f / (rrf_k + static_cast<float>(r + 1));
        }

        std::vector<vrag::SearchHit> fused;
        fused.reserve(candidates.size());
        for (std::uint32_t slot : candidates) {
            fused.push_back(vrag::SearchHit{doc_ids[slot], rrf_scores[slot]});
        }
        std::stable_sort(fused.begin(), fused.end(), [](const vrag::SearchHit& a, const vrag::SearchHit& b) noexcept {
            return a.score > b.score;
        });
        if (fused.size() > top_k) {
            fused.resize(top_k);
        }
        return fused;
    }
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
    float rrf_k = 60.0f,
    float bm25_weight = 1.0f,
    float dense_weight = 1.0f
) {
    const std::vector<float> empty_vec;
    vrag::SearchOptions opts10;
    opts10.top_k = 10;
    opts10.mode = (use_sparse && use_dense) ? vrag::SearchMode::Hybrid
                : (use_sparse ? vrag::SearchMode::BM25 : vrag::SearchMode::Dense);
    opts10.rrf_k = rrf_k;
    opts10.bm25_weight = bm25_weight;
    opts10.dense_weight = dense_weight;

    vrag::SearchOptions opts100 = opts10;
    opts100.top_k = 100;

    const std::size_t warmup = std::min<std::size_t>(20, queries.size());
    for (std::size_t i = 0; i < warmup; ++i) {
        std::string_view qt = use_sparse ? std::string_view(queries[i].text) : std::string_view{};
        std::span<const float> qv = use_dense ? std::span<const float>(queries[i].vector) : std::span<const float>(empty_vec);
        auto res = engine.search(qt, qv, opts10);
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
        auto hits_10 = engine.search(qt, qv, opts10);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_top10_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto hits_100 = engine.search(qt, qv, opts100);

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

    Fp32ReferenceCorpus fp32_ref;
    double top10_overlap_sum = 0.0;

    if (has_dense_vectors && fp32_ref.load(corpus_path)) {
        // Evaluate FP32 Exact Dense & Hybrid
        auto eval_fp32_mode = [&](const std::string& mode_name, bool use_sparse) -> EvalSummary {
            std::vector<double> latencies;
            latencies.reserve(queries.size());
            double sum_ndcg_10 = 0.0, sum_mrr_10 = 0.0, sum_recall_10 = 0.0, sum_recall_100 = 0.0;

            for (const auto& q : queries) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto hits_10 = fp32_ref.search_fp32(q.text, q.vector, 10, use_sparse);
                auto t1 = std::chrono::high_resolution_clock::now();
                latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

                auto hits_100 = fp32_ref.search_fp32(q.text, q.vector, 100, use_sparse);

                if (!use_sparse) {
                    // Measure Top-10 set overlap between FP32 Exact and SQ8
                    auto sq8_10 = engine.search("", q.vector, 10);
                    std::unordered_set<std::uint32_t> fp32_set;
                    for (const auto& h : hits_10) fp32_set.insert(h.doc_id);
                    std::size_t matched = 0;
                    for (const auto& h : sq8_10) {
                        if (fp32_set.count(h.doc_id)) ++matched;
                    }
                    if (!hits_10.empty()) {
                        top10_overlap_sum += static_cast<double>(matched) / static_cast<double>(hits_10.size());
                    }
                }

                double dcg_10 = 0.0, mrr_10 = 0.0;
                std::size_t rel_found_10 = 0;
                for (std::size_t r = 0; r < hits_10.size(); ++r) {
                    auto it = q.qrels.find(hits_10[r].doc_id);
                    if (it != q.qrels.end() && it->second > 0) {
                        const double gain = std::pow(2.0, static_cast<double>(it->second)) - 1.0;
                        dcg_10 += gain / std::log2(static_cast<double>(r) + 2.0);
                        ++rel_found_10;
                        if (mrr_10 == 0.0) mrr_10 = 1.0 / static_cast<double>(r + 1);
                    }
                }

                std::vector<std::uint32_t> ideal_rels;
                for (const auto& [doc_id, rel] : q.qrels) {
                    if (rel > 0) ideal_rels.push_back(rel);
                }
                std::sort(ideal_rels.begin(), ideal_rels.end(), std::greater<std::uint32_t>());
                double idcg_10 = 0.0;
                for (std::size_t r = 0; r < std::min<std::size_t>(10, ideal_rels.size()); ++r) {
                    idcg_10 += (std::pow(2.0, static_cast<double>(ideal_rels[r])) - 1.0) / std::log2(static_cast<double>(r) + 2.0);
                }
                if (idcg_10 > 0.0) sum_ndcg_10 += dcg_10 / idcg_10;
                sum_mrr_10 += mrr_10;

                if (!ideal_rels.empty()) {
                    sum_recall_10 += static_cast<double>(rel_found_10) / static_cast<double>(ideal_rels.size());
                    std::size_t rel_found_100 = 0;
                    for (const auto& h : hits_100) {
                        if (q.qrels.find(h.doc_id) != q.qrels.end()) ++rel_found_100;
                    }
                    sum_recall_100 += static_cast<double>(rel_found_100) / static_cast<double>(ideal_rels.size());
                }
            }
            std::sort(latencies.begin(), latencies.end());
            const std::size_t nq = queries.size();
            EvalSummary s;
            s.mode_name = mode_name;
            s.ndcg_10 = sum_ndcg_10 / static_cast<double>(nq);
            s.mrr_10 = sum_mrr_10 / static_cast<double>(nq);
            s.recall_10 = sum_recall_10 / static_cast<double>(nq);
            s.recall_100 = sum_recall_100 / static_cast<double>(nq);
            s.avg_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(nq);
            s.p50_ms = latencies[nq * 50 / 100];
            s.p95_ms = latencies[nq * 95 / 100];
            s.p99_ms = latencies[std::min(nq - 1, nq * 99 / 100)];
            s.qps = (s.avg_ms > 0.0) ? (1000.0 / s.avg_ms) : 0.0;
            return s;
        };

        summaries.push_back(eval_fp32_mode("2. Dense FP32 Only (Exact)", false));
        summaries.push_back(evaluate_mode(engine, queries, "3. Dense SQ8 Only (INT8)", false, true));
        summaries.push_back(eval_fp32_mode("4. Hybrid FP32 (BM25+FP32)", true));
        summaries.push_back(evaluate_mode(engine, queries, "5. Hybrid SQ8  (1:1 RRF)", true, true, 60.0f, 1.0f, 1.0f));
        summaries.push_back(evaluate_mode(engine, queries, "6. Hybrid SQ8  (0.3:1 W-RRF)", true, true, 60.0f, 0.3f, 1.0f));
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

    if (summaries.size() >= 5) {
        const double fp32_mb = (static_cast<double>(engine.document_count()) * dim * sizeof(float)) / (1024.0 * 1024.0);
        const double sq8_mb = static_cast<double>(engine.memory_used()) / (1024.0 * 1024.0);
        const double overlap_pct = (top10_overlap_sum / static_cast<double>(queries.size())) * 100.0;
        const double delta_dense_ndcg = summaries[2].ndcg_10 - summaries[1].ndcg_10;
        const double delta_hybrid_ndcg = summaries[4].ndcg_10 - summaries[3].ndcg_10;
        std::cout << std::fixed << std::setprecision(4)
                  << " [P0 FP32 vs SQ8 Fidelity] Top-10 Overlap: " << std::setprecision(2) << overlap_pct << "%"
                  << " | Dense dNDCG@10: " << std::showpos << std::setprecision(4) << delta_dense_ndcg
                  << " | Hybrid dNDCG@10: " << delta_hybrid_ndcg << std::noshowpos
                  << " | Vector Mem: " << std::setprecision(2) << fp32_mb << " MB (FP32) -> " << sq8_mb << " MB (SQ8)\n";
        std::cout << "====================================================================================================\n";
    }

    return 0;
}
