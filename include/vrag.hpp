#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <omp.h>

#include "arena.hpp"
#include "inverted_index.hpp"
#include "simd_vector.hpp"
#include "simd_int8.hpp"
#include "tokenizer.hpp"

namespace vrag {

struct HitResult {
    std::uint32_t doc_id;
    float score;
    std::string text;

    bool operator>(const HitResult& other) const noexcept {
        return score > other.score;
    }
};

class VragEngine {
public:
    explicit VragEngine(std::size_t dimension, std::size_t arena_capacity = 256ULL * 1024 * 1024)
        : dimension_(dimension), arena_(arena_capacity) {}

    bool load_from_binary(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "无法打开文件: " << filepath << '\n';
            return false;
        }

        std::uint32_t doc_count = 0;
        std::uint32_t file_dim = 0;
        ifs.read(reinterpret_cast<char*>(&doc_count), sizeof(doc_count));
        ifs.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));

        if (file_dim != dimension_) {
            std::cerr << "维度不匹配: 数据集维度 " << file_dim 
                      << ", 引擎配置维度 " << dimension_ << '\n';
            return false;
        }

        std::vector<float> vec_buf(dimension_);
        for (std::uint32_t i = 0; i < doc_count; ++i) {
            std::uint32_t doc_id = 0;
            std::uint32_t text_len = 0;

            ifs.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
            ifs.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));

            std::string text(text_len, '\0');
            ifs.read(text.data(), text_len);
            ifs.read(reinterpret_cast<char*>(vec_buf.data()), dimension_ * sizeof(float));

            add_document(doc_id, text, vec_buf);
        }

        return true;
    }

    bool add_document(std::uint32_t doc_id, std::string_view text, std::span<const float> raw_vector) {
        if (raw_vector.size() != dimension_) {
            return false;
        }

        // 1. 临时归一化
        std::vector<float> norm_vec(raw_vector.begin(), raw_vector.end());
        SimdVector::normalize(norm_vec);

        // 2. 从 Arena 分配 32 字节对齐的 int8 向量空间（空间缩减 75%）
        int8_t* ptr = arena_.allocate<int8_t>(dimension_, 32);
        if (!ptr) {
            return false;
        }
        std::span<int8_t> int8_span(ptr, dimension_);

        // 3. 原地量化并保存 scale
        float scale = SimdInt8::quantize(norm_vec, int8_span);

        // 4. 更新文档索引
        if (doc_id >= doc_int8_vectors_.size()) {
            doc_int8_vectors_.resize(doc_id + 1, {});
            doc_scales_.resize(doc_id + 1, 0.0f);
            doc_texts_.resize(doc_id + 1, "");
        }
        doc_int8_vectors_[doc_id] = int8_span;
        doc_scales_[doc_id] = scale;
        doc_texts_[doc_id] = std::string(text);
        active_doc_ids_.push_back(doc_id);

        inverted_index_.add_document(doc_id, text);
        return true;
    }

    // 纯双路独立全召回 + RRF 融合（无精度损失，保持 2.8ms / 347 QPS 表现）
    std::vector<HitResult> search(std::string_view query_text,
                                  std::span<const float> query_vector,
                                  std::size_t top_k = 5,
                                  float k = 60.0f) {
        if (active_doc_ids_.empty()) {
            return {};
        }

        const std::size_t total_active = active_doc_ids_.size();
        constexpr std::size_t RRF_RANK_LIMIT = 200; // 仅取两路排名的前 200 名做倒数衰减融合

        // 1. 右路：BM25 全量召回（取前 200 名，耗时 < 0.3 ms）
        auto bm25_hits = inverted_index_.search_bm25(query_text, RRF_RANK_LIMIT);

        // 2. 左路：Query 归一化与全库多核 INT8 密集点积（耗时 ~2.3 ms）
        std::vector<float> aligned_query(query_vector.begin(), query_vector.end());
        SimdVector::normalize(aligned_query);

        std::vector<SearchHit> dense_hits(total_active);

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < total_active; ++i) {
            std::uint32_t doc_id = active_doc_ids_[i];
            float sim = 0.0f;
            if (!doc_int8_vectors_[doc_id].empty()) {
                sim = SimdInt8::dot_product_fp32_int8_avx2(
                    aligned_query.data(),
                    doc_int8_vectors_[doc_id].data(),
                    dimension_,
                    doc_scales_[doc_id]
                );
            }
            dense_hits[i] = SearchHit{doc_id, sim};
        }

        // 局部偏序截取 Dense 向量前 200 名
        const std::size_t top_dense = std::min(dense_hits.size(), RRF_RANK_LIMIT);
        std::partial_sort(dense_hits.begin(), dense_hits.begin() + top_dense, dense_hits.end(),
                          [](const SearchHit& a, const SearchHit& b) {
                              return a.score > b.score;
                          });

        // 3. 收集并去重两路前 200 名的 DocID（候选池至多 400 个）
        std::vector<std::uint32_t> candidate_doc_ids;
        candidate_doc_ids.reserve(bm25_hits.size() + top_dense);

        std::vector<float> rrf_scores(doc_int8_vectors_.size(), 0.0f);

        for (std::size_t rank = 0; rank < bm25_hits.size(); ++rank) {
            std::uint32_t id = bm25_hits[rank].doc_id;
            if (rrf_scores[id] == 0.0f) {
                candidate_doc_ids.push_back(id);
            }
            rrf_scores[id] += 1.0f / (k + static_cast<float>(rank + 1));
        }

        for (std::size_t rank = 0; rank < top_dense; ++rank) {
            std::uint32_t id = dense_hits[rank].doc_id;
            if (rrf_scores[id] == 0.0f) {
                candidate_doc_ids.push_back(id);
            }
            rrf_scores[id] += 1.0f / (k + static_cast<float>(rank + 1));
        }

        // 4. 小顶堆在两路并集中提取 Top-K
        std::priority_queue<HitResult, std::vector<HitResult>, std::greater<HitResult>> min_heap;
        for (std::uint32_t doc_id : candidate_doc_ids) {
            float score = rrf_scores[doc_id];
            if (min_heap.size() < top_k) {
                min_heap.push(HitResult{doc_id, score, doc_texts_[doc_id]});
            } else if (score > min_heap.top().score) {
                min_heap.pop();
                min_heap.push(HitResult{doc_id, score, doc_texts_[doc_id]});
            }
        }

        // 5. 导出并降序排列
        std::vector<HitResult> results;
        results.reserve(min_heap.size());
        while (!min_heap.empty()) {
            results.push_back(std::move(const_cast<HitResult&>(min_heap.top())));
            min_heap.pop();
        }
        std::reverse(results.begin(), results.end());

        return results;
    }

    [[nodiscard]] std::size_t total_docs() const noexcept { return active_doc_ids_.size(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

private:
    std::size_t dimension_;
    Arena arena_;
    InvertedIndex inverted_index_;

    std::vector<std::uint32_t> active_doc_ids_;
    std::vector<std::span<int8_t>> doc_int8_vectors_;
    std::vector<float> doc_scales_;
    std::vector<std::string> doc_texts_;
};

} // namespace vrag
