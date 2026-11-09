#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <string_view>
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

// 1. 测试统一结果结构 (SearchResult / HitResult) 与单路模式 (SearchMode::BM25 / Dense)
void test_unified_result_and_modes() {
    VragEngine engine(4, 4096);
    engine.add_document(100, "deep learning neural network", std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, "source:arxiv;year:2024");
    engine.add_document(200, "deep learning transformer", std::vector<float>{0.6f, 0.8f, 0.0f, 0.0f}, "source:acl;year:2025");

    std::vector<float> q_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    // Hybrid search observability
    SearchOptions opts;
    opts.top_k = 2;
    opts.mode = SearchMode::Hybrid;
    std::vector<SearchResult> hybrid_hits = engine.search("deep learning transformer", q_vec, opts);

    check(hybrid_hits.size() == 2, "P1.1: Hybrid search returns 2 SearchResult items");
    if (hybrid_hits.size() == 2) {
        // Doc 200 is BM25 rank 1 ("deep learning transformer" matches 3 terms vs 2), Dense rank 2
        // Doc 100 is BM25 rank 2, Dense rank 1
        const auto& h0 = hybrid_hits[0];
        check(h0.bm25_score > 0.0f && h0.dense_score > 0.0f &&
              h0.bm25_rank >= 1 && h0.dense_rank >= 1 &&
              !h0.metadata.empty(),
              "P1.1: SearchResult populates bm25_score, dense_score, bm25_rank, dense_rank, and metadata");
    }

    // Pure BM25 mode via search_bm25
    auto bm25_hits = engine.search_bm25("deep learning transformer", 2);
    check(bm25_hits.size() == 2 && bm25_hits[0].doc_id == 200 &&
          approx_eq(bm25_hits[0].score, bm25_hits[0].bm25_score, 1e-6f) &&
          bm25_hits[0].bm25_rank == 1 && bm25_hits[0].dense_rank == 0,
          "P1.1: search_bm25 returns raw BM25 score as primary score and sets bm25_rank=1");

    // Pure Dense mode via search_dense
    auto dense_hits = engine.search_dense(q_vec, 2);
    check(dense_hits.size() == 2 && dense_hits[0].doc_id == 100 &&
          approx_eq(dense_hits[0].score, dense_hits[0].dense_score, 1e-6f) &&
          dense_hits[0].dense_rank == 1 && dense_hits[0].bm25_rank == 0,
          "P1.1: search_dense returns raw SQ8 cosine score as primary score and sets dense_rank=1");
}

// 2. 测试 Weighted RRF 与 min_score 阈值
void test_weighted_rrf_and_threshold() {
    VragEngine engine(4, 4096);
    // Doc 1: BM25 Rank 1 ("alpha alpha beta"), Dense Rank 2 ([0.6, 0.8, 0, 0])
    // Doc 2: BM25 Rank 2 ("alpha gamma"),      Dense Rank 1 ([1.0, 0.0, 0, 0])
    engine.add_document(1, "alpha alpha beta", std::vector<float>{0.6f, 0.8f, 0.0f, 0.0f});
    engine.add_document(2, "alpha gamma", std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});

    std::vector<float> q_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    // Case A: Heavy BM25 weight (bm25_weight = 3.0, dense_weight = 1.0) -> Doc 1 must win
    SearchOptions opts_bm25_heavy;
    opts_bm25_heavy.top_k = 2;
    opts_bm25_heavy.rrf_k = 60.0f;
    opts_bm25_heavy.bm25_weight = 3.0f;
    opts_bm25_heavy.dense_weight = 1.0f;
    auto hits_a = engine.search("alpha", q_vec, opts_bm25_heavy);
    const float exp_a1 = 3.0f / 61.0f + 1.0f / 62.0f;
    const float exp_a2 = 3.0f / 62.0f + 1.0f / 61.0f;
    check(hits_a.size() == 2 && hits_a[0].doc_id == 1 &&
          approx_eq(hits_a[0].score, exp_a1, 1e-6f) &&
          approx_eq(hits_a[1].score, exp_a2, 1e-6f),
          "P1.3: Weighted RRF (3:1 BM25-heavy) ranks BM25 leader #1 with exact weighted score");

    // Case B: Heavy Dense weight (bm25_weight = 1.0, dense_weight = 3.0) -> Doc 2 must win
    SearchOptions opts_dense_heavy;
    opts_dense_heavy.top_k = 2;
    opts_dense_heavy.rrf_k = 60.0f;
    opts_dense_heavy.bm25_weight = 1.0f;
    opts_dense_heavy.dense_weight = 3.0f;
    auto hits_b = engine.search("alpha", q_vec, opts_dense_heavy);
    check(hits_b.size() == 2 && hits_b[0].doc_id == 2 &&
          approx_eq(hits_b[0].score, exp_a1, 1e-6f),
          "P1.3: Weighted RRF (1:3 Dense-heavy) ranks Dense leader #1 with exact weighted score");

    // Case C: min_score threshold filters out second document
    SearchOptions opts_min = opts_dense_heavy;
    opts_min.min_score = (exp_a1 + exp_a2) * 0.5f;
    auto hits_min = engine.search("alpha", q_vec, opts_min);
    check(hits_min.size() == 1 && hits_min[0].doc_id == 2,
          "P1.2: SearchOptions.min_score threshold accurately filters low-score candidates");
}

// 3. 测试 Metadata Filter 前置过滤与持久化保存
void test_metadata_filter_and_persistence() {
    VragEngine engine(4, 4096);
    engine.add_document(10, "rust systems programming", std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, "lang:en;tag:tech");
    engine.add_document(20, "cpp systems programming",  std::vector<float>{0.9f, 0.1f, 0.0f, 0.0f}, "lang:zh;tag:tech");
    engine.add_document(30, "medical biology programming", std::vector<float>{0.8f, 0.2f, 0.0f, 0.0f}, "lang:zh;tag:bio");

    std::vector<float> q_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchOptions opts;
    opts.top_k = 5;
    opts.filter = [](std::uint32_t /*doc_id*/, std::string_view meta) {
        return meta.find("lang:zh") != std::string_view::npos;
    };

    auto filtered_hits = engine.search("programming", q_vec, opts);
    check(filtered_hits.size() == 2 &&
          filtered_hits[0].doc_id == 20 &&
          filtered_hits[1].doc_id == 30,
          "P1.4: Metadata pre-filter excludes non-matching docs before Top-K selection");

    // Persistence of metadata (VRAGIDX2)
    const std::string snap = "test_p1_meta.vidx";
    check(engine.save_index(snap), "P1.4: save_index with metadata succeeded");
    VragEngine loaded(4, 4096);
    check(loaded.load_index(snap), "P1.4: load_index with metadata succeeded");
    std::remove(snap.c_str());

    check(loaded.get_metadata(20) == "lang:zh;tag:tech" &&
          loaded.get_metadata(30) == "lang:zh;tag:bio",
          "P1.4: metadata preserved accurately after save_index / load_index");
}

// 4. 测试外部 Reranker 回调接口
void test_reranker_interface() {
    VragEngine engine(4, 4096);
    engine.add_document(1, "generic overview of databases", std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    engine.add_document(2, "exact answer: B-tree and LSM-tree storage engine", std::vector<float>{0.7f, 0.7f, 0.0f, 0.0f});
    engine.add_document(3, "other unrelated database note", std::vector<float>{0.8f, 0.6f, 0.0f, 0.0f});

    std::vector<float> q_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    SearchOptions opts;
    opts.top_k = 1; // Only want final Top-1, but reranker inspects all candidates up to candidate_limit
    opts.candidate_limit = 10;
    opts.reranker = [](std::string_view /*query*/, std::vector<HitResult>& candidates) {
        for (auto& c : candidates) {
            if (c.text.find("LSM-tree") != std::string::npos) {
                c.score += 10.0f; // Cross-encoder style semantic boost
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const HitResult& a, const HitResult& b) {
            return a.score > b.score;
        });
    };

    auto hits = engine.search("database", q_vec, opts);
    check(hits.size() == 1 && hits[0].doc_id == 2 && hits[0].score > 10.0f,
          "P1.5: Reranker callback inspects full candidate pool, re-scores, and truncates to top_k=1");
}

} // namespace

int main() {
    std::cout << "=== Running P1 Core Retrieval Capabilities & Unified API Suite ===\n";
    test_unified_result_and_modes();
    test_weighted_rrf_and_threshold();
    test_metadata_filter_and_persistence();
    test_reranker_interface();

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed.\n";
    return (g_failed == 0) ? 0 : 1;
}
