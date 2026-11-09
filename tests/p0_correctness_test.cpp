#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
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

// 1. BM25 数学精确性与边界测试
void test_bm25_correctness() {
    InvertedIndex idx;
    // Doc 0: "alpha beta" (dl = 2, alpha tf = 1)
    // Doc 1: "alpha alpha beta gamma" (dl = 4, alpha tf = 2)
    // Doc 2: "alpha beta gamma delta epsilon" (dl = 5, alpha tf = 1)
    // Doc 3: "zeta eta theta" (dl = 3, alpha tf = 0)
    idx.add_document(0, "alpha beta");
    idx.add_document(1, "alpha alpha beta gamma");
    idx.add_document(2, "alpha beta gamma delta epsilon");
    idx.add_document(3, "zeta eta theta");

    // Analytical BM25 for query "alpha":
    // N = 4, df("alpha") = 3, avgdl = (2 + 4 + 5 + 3) / 4.0 = 3.5
    // idf = ln(1 + (4 - 3 + 0.5) / (3 + 0.5)) = ln(1 + 1.5 / 3.5)
    const float k1 = 1.2f;
    const float b = 0.75f;
    const float avgdl = 3.5f;
    const float idf = std::log(1.0f + 1.5f / 3.5f);

    auto expected_score = [&](float tf, float dl) {
        return idf * (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * (dl / avgdl)));
    };

    const float s0 = expected_score(1.0f, 2.0f);
    const float s1 = expected_score(2.0f, 4.0f);
    const float s2 = expected_score(1.0f, 5.0f);

    auto hits = idx.search_bm25("alpha", 10, k1, b);
    check(hits.size() == 3, "BM25: exact hit count (3 docs containing 'alpha')");
    if (hits.size() == 3) {
        // Expect s1 (tf=2) > s0 (tf=1, short dl=2) > s2 (tf=1, long dl=5)
        check(hits[0].doc_id == 1 && approx_eq(hits[0].score, s1, 1e-5f),
              "BM25: analytical formula match for Doc 1 (tf=2, dl=4)");
        check(hits[1].doc_id == 0 && approx_eq(hits[1].score, s0, 1e-5f),
              "BM25: analytical formula match for Doc 0 (tf=1, dl=2, short doc bonus)");
        check(hits[2].doc_id == 2 && approx_eq(hits[2].score, s2, 1e-5f),
              "BM25: analytical formula match for Doc 2 (tf=1, dl=5, long doc penalty)");
    }

    // When b = 0.0f, document length normalization is disabled: Doc 0 (dl=2, tf=1) and Doc 2 (dl=5, tf=1) must have equal score
    auto hits_b0 = idx.search_bm25("alpha", 10, k1, 0.0f);
    check(hits_b0.size() == 3 && approx_eq(hits_b0[1].score, hits_b0[2].score, 1e-6f),
          "BM25: b=0.0 disables document length normalization");

    // Deletion IDF Invariance Test:
    // Removing Doc 2 from idx must produce the EXACT same BM25 scores as a fresh index built with only Doc 0, 1, 3
    idx.remove_document(2);

    InvertedIndex fresh_idx;
    fresh_idx.add_document(0, "alpha beta");
    fresh_idx.add_document(1, "alpha alpha beta gamma");
    fresh_idx.add_document(3, "zeta eta theta");

    auto hits_after_del = idx.search_bm25("alpha", 10, k1, b);
    auto hits_fresh = fresh_idx.search_bm25("alpha", 10, k1, b);
    check(hits_after_del.size() == hits_fresh.size() && hits_after_del.size() == 2 &&
          approx_eq(hits_after_del[0].score, hits_fresh[0].score, 1e-6f) &&
          approx_eq(hits_after_del[1].score, hits_fresh[1].score, 1e-6f),
          "BM25: deletion IDF & avgdl invariance (matches fresh index bit-for-bit)");
}

// 2. Dense & SQ8 多维度与算子精确性测试
void test_dense_sq8_correctness() {
    // 测试各种非对齐/边界维度：1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 384, 768
    const std::vector<std::size_t> dims = {1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 384, 768};
    bool all_dims_ok = true;

    for (std::size_t dim : dims) {
        const std::size_t padded_dim = (dim + 63) & ~std::size_t(63);
        std::vector<float> q_fp32(dim);
        std::vector<float> d_fp32(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            q_fp32[i] = std::sin(static_cast<float>(i + 1) * 0.13f);
            d_fp32[i] = std::cos(static_cast<float>(i + 1) * 0.07f);
        }

        // FP32 ground truth cosine similarity
        std::vector<float> q_norm = q_fp32;
        std::vector<float> d_norm = d_fp32;
        SimdVector::normalize(q_norm);
        SimdVector::normalize(d_norm);
        float fp32_cos = SimdVector::dot_product_avx2(q_norm, d_norm);

        // SQ8 fused normalize_and_quantize
        std::vector<int8_t> q_i8(padded_dim, 0);
        std::vector<uint8_t> q_abs(padded_dim, 0);
        std::vector<uint8_t> q_u8(padded_dim, 128);
        std::vector<int8_t> d_i8(padded_dim, 0);

        int32_t d_sum_128 = 0;
        float q_scale = SimdInt8::normalize_and_quantize(
            q_fp32, std::span<int8_t>(q_i8.data(), dim), q_abs.data(), q_u8.data(), nullptr);
        float d_scale = SimdInt8::normalize_and_quantize(
            d_fp32, std::span<int8_t>(d_i8.data(), dim), nullptr, nullptr, &d_sum_128);

        // Scalar integer dot product reference
        int32_t scalar_i32 = 0;
        for (std::size_t i = 0; i < padded_dim; ++i) {
            scalar_i32 += static_cast<int32_t>(q_i8[i]) * static_cast<int32_t>(d_i8[i]);
        }
        float scalar_sq8_sim = static_cast<float>(scalar_i32) * (q_scale * d_scale);

        float vnni_sim = SimdInt8::dot_product_u8_s8_fast(
            q_u8.data(), q_i8.data(), q_abs.data(), d_i8.data(), padded_dim, d_sum_128, q_scale * d_scale);
        float avx2_sim = SimdInt8::dot_product_int8_avx2(
            q_i8.data(), q_abs.data(), d_i8.data(), padded_dim, q_scale * d_scale);

        // VNNI and AVX2 integer dot products must match scalar integer dot product exactly
        if (!approx_eq(vnni_sim, scalar_sq8_sim, 1e-6f) || !approx_eq(avx2_sim, scalar_sq8_sim, 1e-6f)) {
            all_dims_ok = false;
            std::cerr << "  Dimension " << dim << " integer dot mismatch: scalar=" << scalar_sq8_sim
                      << " vnni=" << vnni_sim << " avx2=" << avx2_sim << '\n';
        }
        // SQ8 cosine approximation error vs FP32 must be small (< 0.015)
        if (std::fabs(fp32_cos - vnni_sim) > 0.015f) {
            all_dims_ok = false;
            std::cerr << "  Dimension " << dim << " FP32 vs SQ8 error too large: fp32=" << fp32_cos
                      << " sq8=" << vnni_sim << '\n';
        }
    }
    check(all_dims_ok, "Dense & SQ8: exact integer dot match & bounded quantization error across 13 odd/even dimensions");
}

// 3. Hybrid RRF 公式、去重与单路退化验证
void test_hybrid_rrf_correctness() {
    constexpr std::size_t Dim = 4;
    VragEngine engine(Dim, 4096);

    // Doc 10: BM25 Rank 1 ("quantum computing"), Dense Rank 2 ([0.6, 0.8, 0, 0])
    // Doc 20: BM25 Rank 2 ("quantum physics"),   Dense Rank 1 ([1.0, 0.0, 0, 0])
    // Doc 30: BM25 Not matched ("biology cell"), Dense Rank 3 ([0.0, 1.0, 0, 0])
    engine.add_document(10, "quantum computing quantum", std::vector<float>{0.6f, 0.8f, 0.0f, 0.0f});
    engine.add_document(20, "quantum physics", std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    engine.add_document(30, "biology cell", std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});

    const float rrf_k = 60.0f;
    std::vector<float> q_vec = {1.0f, 0.0f, 0.0f, 0.0f};
    auto hits = engine.search("quantum", q_vec, 10, rrf_k);

    // Expected RRF scores:
    // Doc 10: 1/(60+1) [BM25 rank 1] + 1/(60+2) [Dense rank 2]
    // Doc 20: 1/(60+2) [BM25 rank 2] + 1/(60+1) [Dense rank 1]
    // Doc 30: 0        [BM25 miss]   + 1/(60+3) [Dense rank 3]
    const float expected_10_20 = (1.0f / 61.0f) + (1.0f / 62.0f);
    const float expected_30 = 1.0f / 63.0f;

    check(hits.size() == 3, "Hybrid RRF: deduplicated candidate union size == 3");
    if (hits.size() == 3) {
        check(approx_eq(hits[0].score, expected_10_20, 1e-6f) &&
              approx_eq(hits[1].score, expected_10_20, 1e-6f) &&
              approx_eq(hits[2].score, expected_30, 1e-6f) &&
              hits[2].doc_id == 30,
              "Hybrid RRF: exact 1/(k+rank) score accumulation and deduplication");
    }

    // Pure BM25 fallback (zero vector)
    std::vector<float> zero_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    auto bm25_only = engine.search("quantum", zero_vec, 10, rrf_k);
    check(bm25_only.size() == 2 && bm25_only[0].doc_id == 10 && approx_eq(bm25_only[0].score, 1.0f / 61.0f, 1e-6f),
          "Hybrid RRF: zero query vector cleanly degrades to BM25-only ranking");

    // Pure Dense fallback (empty query string)
    auto dense_only = engine.search("", q_vec, 10, rrf_k);
    check(dense_only.size() == 3 && dense_only[0].doc_id == 20 && approx_eq(dense_only[0].score, 1.0f / 61.0f, 1e-6f),
          "Hybrid RRF: empty query text cleanly degrades to Dense-only ranking");
}

// 4. Tokenizer 边界与损坏 UTF-8 序列防护测试
void test_tokenizer_resilience() {
    // 构造包含损坏/截断 UTF-8 字节但紧跟合法英文与中文的字符串：
    // \xE4 (截断的 3 字节 UTF-8 首字节) + "hello" + \xC3 (截断的 2 字节首字节) + "world" + \xF0\x9F (截断的 4 字节) + "测试"
    std::string corrupt_utf8 = "\xE4hello \xC3world \xF0\x9F\x80测试\xE5";
    std::vector<std::string_view> tokens;
    for (auto tk : Tokenizer::cut(corrupt_utf8)) {
        tokens.push_back(tk);
    }
    std::vector<std::string_view> expected = {"hello", "world", "测试"};
    check(tokens == expected, "Tokenizer: truncated/corrupt UTF-8 bytes do not swallow adjacent ASCII/CJK tokens");

    // Case-insensitive hash check
    check(fnv1a_hash("HelloWorld123") == fnv1a_hash("helloworld123"),
          "Tokenizer: fnv1a_hash is strictly case-insensitive for ASCII");
}

// 5. 异常输入与防御性边界测试
void test_exception_and_edge_cases() {
    VragEngine engine(4, 1024);
    std::vector<float> valid_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    // 5.1 空索引检索
    check(engine.search("test", valid_vec, 5).empty(), "Edge case: search on empty engine returns empty");

    // 5.2 维度不匹配拒绝
    std::vector<float> bad_dim_vec = {1.0f, 0.0f};
    check(!engine.add_document(1, "doc", bad_dim_vec), "Edge case: mismatched vector dimension rejected");

    // 5.3 NaN / Inf 向量拒绝
    std::vector<float> nan_vec = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f, 0.0f};
    std::vector<float> inf_vec = {std::numeric_limits<float>::infinity(), 1.0f, 0.0f, 0.0f};
    check(!engine.add_document(1, "nan doc", nan_vec), "Edge case: NaN document vector rejected");
    check(!engine.add_document(1, "inf doc", inf_vec), "Edge case: Inf document vector rejected");

    // 5.4 空文档与全零向量文档可安全插入且不崩溃
    std::vector<float> zero_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    check(engine.add_document(1, "", zero_vec), "Edge case: empty text + zero vector document safely handled");
    check(engine.add_document(2, "normal document text", valid_vec), "Edge case: normal document inserted");

    // 5.5 非法查询参数 (top_k = 0, rrf_k <= 0, NaN query vector)
    check(engine.search("normal", valid_vec, 0).empty(), "Edge case: top_k = 0 returns empty");
    check(engine.search("normal", valid_vec, 5, 0.0f).empty(), "Edge case: rrf_k = 0.0f returns empty");
    check(engine.search("normal", valid_vec, 5, -10.0f).empty(), "Edge case: negative rrf_k returns empty");
    auto nan_q_hits = engine.search("normal", nan_vec, 5);
    check(!nan_q_hits.empty() && nan_q_hits[0].doc_id == 2,
          "Edge case: NaN query vector safely ignored while preserving BM25 results");

    // 5.6 删除不存在的文档 & 损坏索引文件加载
    check(!engine.remove_document(999999), "Edge case: removing non-existent doc_id returns false");
    check(!engine.load_index("non_existent_file_12345.vidx"), "Edge case: loading missing file returns false");

    // 构造一个被截断的损坏索引文件
    {
        std::ofstream ofs("corrupt_test.vidx", std::ios::binary);
        ofs.write("VRAGIDX1", 8);
        std::uint64_t bogus = 4;
        ofs.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
    }
    VragEngine tmp_engine(4, 1024);
    check(!tmp_engine.load_index("corrupt_test.vidx"), "Edge case: truncated snapshot file rejected by load_index");
    std::remove("corrupt_test.vidx");
}

} // namespace

int main() {
    std::cout << "=== Running P0 Foundation Correctness & Edge-Case Suite ===\n";
    test_bm25_correctness();
    test_dense_sq8_correctness();
    test_hybrid_rrf_correctness();
    test_tokenizer_resilience();
    test_exception_and_edge_cases();

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed.\n";
    return (g_failed == 0) ? 0 : 1;
}
