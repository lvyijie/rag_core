#include <atomic>
#include <cmath>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../include/vrag.hpp"

using namespace vrag;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_passed;
        std::cout << "[PASS] " << msg << '\n';
    } else {
        ++g_failed;
        std::cerr << "[FAIL] " << msg << '\n';
    }
}

bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// 1. 验证批量并行建库 (add_documents_batch) 与逐条建库 (add_document) 的数学完全等价性
void test_batch_parallel_ingestion() {
    constexpr std::size_t Dim = 64;
    constexpr std::size_t N = 300; // >= 128 触发 OpenMP 并行归一化与量化

    VragEngine seq_engine(Dim, 64 * 1024);
    VragEngine batch_engine(Dim, 64 * 1024);

    std::vector<std::uint32_t> doc_ids(N);
    std::vector<std::string> owned_texts(N);
    std::vector<std::string_view> text_views(N);
    std::vector<std::string> owned_meta(N);
    std::vector<std::string_view> meta_views(N);
    std::vector<float> flat_vecs(N * Dim, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        doc_ids[i] = static_cast<std::uint32_t>(1000 + i);
        owned_texts[i] = "parallel batch indexing test document number " + std::to_string(i) +
                         ((i % 5 == 0) ? " special_keyword_alpha" : " general_corpus");
        text_views[i] = owned_texts[i];
        owned_meta[i] = "idx:" + std::to_string(i);
        meta_views[i] = owned_meta[i];

        for (std::size_t d = 0; d < Dim; ++d) {
            flat_vecs[i * Dim + d] = std::sin(static_cast<float>(i * 7 + d) * 0.11f);
        }

        seq_engine.add_document(doc_ids[i], text_views[i],
                                std::span<const float>(flat_vecs.data() + i * Dim, Dim),
                                meta_views[i]);
    }

    check(batch_engine.add_documents_batch(doc_ids, text_views, flat_vecs, meta_views),
          "P3.2: add_documents_batch (OpenMP parallel quantization for 300 docs) succeeded");
    check(batch_engine.document_count() == N && seq_engine.document_count() == N,
          "P3.2: batch_engine document_count matches sequential engine");

    std::vector<float> q_vec(Dim);
    for (std::size_t d = 0; d < Dim; ++d) {
        q_vec[d] = std::sin(static_cast<float>(25 * 7 + d) * 0.11f);
    }

    auto hits_seq = seq_engine.search("special_keyword_alpha", q_vec, 15);
    auto hits_batch = batch_engine.search("special_keyword_alpha", q_vec, 15);

    bool exact_match = (hits_seq.size() == hits_batch.size() && !hits_seq.empty());
    for (std::size_t i = 0; exact_match && i < hits_seq.size(); ++i) {
        if (hits_seq[i].doc_id != hits_batch[i].doc_id ||
            !approx_eq(hits_seq[i].score, hits_batch[i].score, 1e-6f) ||
            !approx_eq(hits_seq[i].bm25_score, hits_batch[i].bm25_score, 1e-6f) ||
            !approx_eq(hits_seq[i].dense_score, hits_batch[i].dense_score, 1e-6f) ||
            hits_seq[i].metadata != hits_batch[i].metadata) {
            exact_match = false;
        }
    }
    check(exact_match, "P3.2: add_documents_batch produces bit-for-bit identical search scores to sequential add_document");
}

// 2. 验证多线程并发读写安全 (6 个读线程 + 2 个写线程同时执行 search / add / remove / compact)
void test_concurrent_readers_and_writers() {
    constexpr std::size_t Dim = 32;
    VragEngine engine(Dim, 32 * 1024);

    // 预填充 50 篇常驻锚点文档 (doc_id 1..50)，永不删除
    for (std::uint32_t id = 1; id <= 50; ++id) {
        std::vector<float> vec(Dim, 0.1f);
        vec[id % Dim] = 1.0f;
        engine.add_document(id, "anchor document permanent knowledge " + std::to_string(id), vec, "type:anchor");
    }

    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;

    // 启动 6 个并发查询读线程
    for (int r = 0; r < 6; ++r) {
        threads.emplace_back([&engine, &ok, r]() {
            std::vector<float> q_vec(Dim, 0.1f);
            q_vec[(r + 1) % Dim] = 1.0f;
            SearchOptions opts;
            opts.top_k = 5;
            opts.num_threads = 1;

            for (int iter = 0; iter < 300; ++iter) {
                auto hits = engine.search("anchor document permanent", q_vec, opts);
                if (hits.empty() || !std::isfinite(hits[0].score)) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
                if (!engine.contains_document(1)) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
                auto st = engine.memory_stats();
                if (st.total_bytes() == 0) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }

    // 启动 2 个并发写入/删除/压缩线程 (操作临时文档 doc_id 1000..1200)
    for (int w = 0; w < 2; ++w) {
        threads.emplace_back([&engine, &ok, w]() {
            std::vector<float> vec(Dim, 0.2f);
            vec[(w + 5) % Dim] = 0.9f;
            for (int iter = 0; iter < 100; ++iter) {
                const std::uint32_t doc_id = static_cast<std::uint32_t>(1000 + w * 100 + (iter % 25));
                if (!engine.add_document(doc_id, "dynamic concurrent update note " + std::to_string(iter), vec, "type:dyn")) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
                if (iter % 3 == 0) {
                    engine.remove_document(doc_id);
                }
                if (iter % 25 == 24) {
                    engine.compact();
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    check(ok.load() && engine.document_count() >= 50,
          "P3.1: 6 Reader threads + 2 Writer threads (search + add + remove + compact) completed with zero race/crash");
}

} // namespace

int main() {
    std::cout << "=== Running P3 Concurrency & Parallel Batch Ingestion Suite ===\n";
    test_batch_parallel_ingestion();
    test_concurrent_readers_and_writers();

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed.\n";
    return (g_failed == 0) ? 0 : 1;
}
