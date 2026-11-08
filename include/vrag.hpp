#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
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
    static constexpr std::size_t RRF_RANK_LIMIT = 200;

    explicit VragEngine(std::size_t dimension = 768,
                        std::size_t arena_capacity = 256ULL * 1024 * 1024)
        : dimension_(dimension),
          padded_dim_((dimension + 63) & ~std::size_t(63)),
          arena_(arena_capacity) {}

    // 从原始语料二进制包 (*.bin: UTF-8 文本 + FP32 向量) 加载并构建索引
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

        if (!ifs || file_dim != dimension_) {
            std::cerr << "维度不匹配: 数据集维度 " << file_dim
                      << ", 引擎配置维度 " << dimension_ << '\n';
            return false;
        }

        // 预分配容器与 Arena 容量，避免加载期动态扩容拷贝
        const std::size_t target_slots = slot_to_doc_id_.size() + doc_count;
        arena_.reserve(target_slots * padded_dim_);
        slot_to_doc_id_.reserve(target_slots);
        slot_deleted_.reserve(target_slots);
        active_scales_.reserve(target_slots);
        active_sums_128_.reserve(target_slots);
        doc_texts_.reserve(target_slots);
        doc_id_to_slot_.reserve(target_slots);
        inverted_index_.reserve_docs(target_slots);

        std::vector<float> vec_buf(dimension_);
        std::string text;
        text.reserve(1024);

        for (std::uint32_t i = 0; i < doc_count; ++i) {
            std::uint32_t doc_id = 0;
            std::uint32_t text_len = 0;

            ifs.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
            ifs.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));

            text.resize(text_len);
            ifs.read(text.data(), text_len);
            ifs.read(reinterpret_cast<char*>(vec_buf.data()), dimension_ * sizeof(float));

            if (!ifs) {
                return false;
            }
            add_document(doc_id, text, vec_buf);
        }

        return true;
    }

    // 将已构建完成的索引（已量化 INT8 向量池 + 扁平倒排表 + 原文）直接序列化落盘
    bool save_index(const std::string& filepath) const {
        std::ofstream ofs(filepath, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            return false;
        }

        static constexpr char kMagic[8] = {'V', 'R', 'A', 'G', 'I', 'D', 'X', '1'};
        const std::uint64_t dim = dimension_;
        const std::uint64_t p_dim = padded_dim_;
        const std::uint64_t slots = slot_to_doc_id_.size();
        const std::uint64_t active = active_docs_;
        const std::uint8_t has_del = has_deletions_ ? 1 : 0;

        ofs.write(kMagic, sizeof(kMagic));
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        ofs.write(reinterpret_cast<const char*>(&p_dim), sizeof(p_dim));
        ofs.write(reinterpret_cast<const char*>(&slots), sizeof(slots));
        ofs.write(reinterpret_cast<const char*>(&active), sizeof(active));
        ofs.write(reinterpret_cast<const char*>(&has_del), sizeof(has_del));

        if (slots > 0) {
            ofs.write(reinterpret_cast<const char*>(slot_to_doc_id_.data()), slots * sizeof(std::uint32_t));
            ofs.write(reinterpret_cast<const char*>(slot_deleted_.data()), slots * sizeof(std::uint8_t));
            ofs.write(reinterpret_cast<const char*>(active_scales_.data()), slots * sizeof(float));
            ofs.write(reinterpret_cast<const char*>(active_sums_128_.data()), slots * sizeof(int32_t));
            ofs.write(reinterpret_cast<const char*>(arena_.data()), slots * padded_dim_);

            for (std::size_t i = 0; i < slots; ++i) {
                const std::uint32_t len = static_cast<std::uint32_t>(doc_texts_[i].size());
                ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
                if (len > 0) {
                    ofs.write(doc_texts_[i].data(), len);
                }
            }
        }

        return inverted_index_.save(ofs);
    }

    // 极速加载预构建索引快照（免分词、免归一化、免量化，直接恢复内存映像）
    bool load_index(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }

        char magic[8]{};
        std::uint64_t dim = 0;
        std::uint64_t p_dim = 0;
        std::uint64_t slots = 0;
        std::uint64_t active = 0;
        std::uint8_t has_del = 0;

        if (!ifs.read(magic, sizeof(magic)) ||
            std::memcmp(magic, "VRAGIDX1", 8) != 0 ||
            !ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim)) ||
            !ifs.read(reinterpret_cast<char*>(&p_dim), sizeof(p_dim)) ||
            !ifs.read(reinterpret_cast<char*>(&slots), sizeof(slots)) ||
            !ifs.read(reinterpret_cast<char*>(&active), sizeof(active)) ||
            !ifs.read(reinterpret_cast<char*>(&has_del), sizeof(has_del))) {
            return false;
        }

        if (dim != dimension_ || p_dim != padded_dim_) {
            return false;
        }

        active_docs_ = static_cast<std::size_t>(active);
        has_deletions_ = (has_del != 0);

        const std::size_t n = static_cast<std::size_t>(slots);
        slot_to_doc_id_.resize(n);
        slot_deleted_.resize(n);
        active_scales_.resize(n);
        active_sums_128_.resize(n);
        doc_texts_.resize(n);
        arena_.resize_raw(n * padded_dim_);

        if (n > 0) {
            ifs.read(reinterpret_cast<char*>(slot_to_doc_id_.data()), n * sizeof(std::uint32_t));
            ifs.read(reinterpret_cast<char*>(slot_deleted_.data()), n * sizeof(std::uint8_t));
            ifs.read(reinterpret_cast<char*>(active_scales_.data()), n * sizeof(float));
            ifs.read(reinterpret_cast<char*>(active_sums_128_.data()), n * sizeof(int32_t));
            ifs.read(reinterpret_cast<char*>(arena_.data()), n * padded_dim_);

            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t len = 0;
                ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
                doc_texts_[i].resize(len);
                if (len > 0) {
                    ifs.read(doc_texts_[i].data(), len);
                }
            }
        }

        doc_id_to_slot_.clear();
        doc_id_to_slot_.reserve(active_docs_);
        for (std::size_t i = 0; i < n; ++i) {
            if (!slot_deleted_[i]) {
                doc_id_to_slot_[slot_to_doc_id_[i]] = static_cast<std::uint32_t>(i);
            }
        }

        return inverted_index_.load(ifs);
    }

    // 插入或更新单篇文档 (Upsert：若 doc_id 已存在则自动标记旧槽位失效并写入新版本)
    bool add_document(std::uint32_t doc_id, std::string_view text, std::span<const float> raw_vector) {
        if (raw_vector.size() != dimension_) {
            return false;
        }

        // 若同一外部 doc_id 已存在，先将其旧内部槽位软删除
        auto existing = doc_id_to_slot_.find(doc_id);
        if (existing != doc_id_to_slot_.end()) {
            const std::uint32_t old_slot = existing->second;
            slot_deleted_[old_slot] = 1;
            has_deletions_ = true;
            inverted_index_.remove_document(old_slot);
            doc_texts_[old_slot].clear();
            doc_texts_[old_slot].shrink_to_fit();
            if (active_docs_ > 0) {
                --active_docs_;
            }
        }

        const std::uint32_t slot_id = static_cast<std::uint32_t>(slot_to_doc_id_.size());

        // 1. 从 Arena 分配 32 字节对齐、64 字节倍数补齐的连续 int8 向量空间（容量不足时自动 2 倍扩容）
        int8_t* ptr = arena_.allocate<int8_t>(padded_dim_, 32);
        if (!ptr) {
            return false;
        }
        if (padded_dim_ > dimension_) {
            std::memset(ptr + dimension_, 0, padded_dim_ - dimension_);
        }
        std::span<int8_t> int8_span(ptr, dimension_);

        // 2. 融合单遍归一化、量化与偏移和预计算
        int32_t sum_128 = 0;
        float scale = SimdInt8::normalize_and_quantize(raw_vector, int8_span, nullptr, nullptr, &sum_128);

        // 3. 记录内部连续槽位元数据
        slot_to_doc_id_.push_back(doc_id);
        slot_deleted_.push_back(0);
        active_scales_.push_back(scale);
        active_sums_128_.push_back(sum_128);
        doc_texts_.emplace_back(text);
        doc_id_to_slot_[doc_id] = slot_id;
        ++active_docs_;

        inverted_index_.add_document(slot_id, text);
        return true;
    }

    // 删除指定 doc_id 的文档（O(1) 墓碑标记，立即从稀疏与稠密检索结果中剔除）
    bool remove_document(std::uint32_t doc_id) {
        auto it = doc_id_to_slot_.find(doc_id);
        if (it == doc_id_to_slot_.end()) {
            return false;
        }
        const std::uint32_t slot_id = it->second;
        doc_id_to_slot_.erase(it);
        slot_deleted_[slot_id] = 1;
        has_deletions_ = true;
        inverted_index_.remove_document(slot_id);
        doc_texts_[slot_id].clear();
        doc_texts_[slot_id].shrink_to_fit();
        if (active_docs_ > 0) {
            --active_docs_;
        }
        return true;
    }

    [[nodiscard]] bool contains_document(std::uint32_t doc_id) const noexcept {
        return doc_id_to_slot_.find(doc_id) != doc_id_to_slot_.end();
    }

    // 线程安全只读双路混合检索 + RRF 融合（多线程并发调用零锁竞争，单线程复用零堆分配）
    std::vector<HitResult> search(std::string_view query_text,
                                  std::span<const float> query_vector,
                                  std::size_t top_k = 5,
                                  float k = 60.0f) const {
        if (active_docs_ == 0 || top_k == 0) {
            return {};
        }

        const std::size_t total_slots = slot_to_doc_id_.size();

        // 线程局部查询工作区：保证任意多业务线程并发调用 engine.search() 绝对安全
        thread_local SearchWorkspace ws;
        ws.ensure_dimension(padded_dim_, total_slots);

        // 1. 右路：BM25 零分配全量召回（返回内部 slot_id）
        inverted_index_.search_bm25_into(query_text, RRF_RANK_LIMIT, ws.bm25_hits_buf);

        // 2. 左路：检查查询向量是否有效并执行融合归一化+量化
        bool has_dense_query = false;
        if (query_vector.size() == dimension_) {
            for (float v : query_vector) {
                if (v != 0.0f) {
                    has_dense_query = true;
                    break;
                }
            }
        }

        ws.dense_candidates.clear();

        if (has_dense_query) {
            int32_t q_sum_128 = 0;
            float q_scale = SimdInt8::normalize_and_quantize(
                query_vector,
                std::span<int8_t>(ws.q_int8.data(), dimension_),
                ws.q_abs.data(),
                ws.q_u8.data(),
                &q_sum_128
            );

            const uint8_t* __restrict q_u8_ptr = ws.q_u8.data();
            const int8_t* __restrict q_i8_ptr = ws.q_int8.data();
            const uint8_t* __restrict q_abs_ptr = ws.q_abs.data();
            const int8_t* __restrict arena_base = reinterpret_cast<const int8_t*>(arena_.data());
            const std::uint8_t* __restrict deleted = slot_deleted_.data();
            const float* __restrict scales = active_scales_.data();
            const int32_t* __restrict sums_128 = active_sums_128_.data();
            const std::size_t p_dim = padded_dim_;
            const std::size_t top_dense = std::min(active_docs_, RRF_RANK_LIMIT);
            const bool check_del = has_deletions_;

            auto hit_greater = [](const SearchHit& a, const SearchHit& b) noexcept {
                return (a.score != b.score) ? (a.score > b.score) : (a.doc_id < b.doc_id);
            };

            if (total_slots <= 4096) {
                ws.dense_candidates.reserve(top_dense);
                float min_top_score = -1e30f;

                for (std::size_t i = 0; i < total_slots; ++i) {
                    if (check_del && deleted[i]) {
                        continue;
                    }
                    const int8_t* vec_ptr = arena_base + i * p_dim;
                    prefetch_l1(vec_ptr + p_dim * 2);

                    float sim = SimdInt8::dot_product_u8_s8_fast(
                        q_u8_ptr, q_i8_ptr, q_abs_ptr, vec_ptr, p_dim, sums_128[i], q_scale * scales[i]
                    );
                    const auto slot_u32 = static_cast<std::uint32_t>(i);
                    if (ws.dense_candidates.size() < top_dense) {
                        ws.dense_candidates.push_back(SearchHit{slot_u32, sim});
                        if (ws.dense_candidates.size() == top_dense) {
                            std::make_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                            min_top_score = ws.dense_candidates[0].score;
                        }
                    } else if (sim > min_top_score) {
                        std::pop_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                        ws.dense_candidates.back() = SearchHit{slot_u32, sim};
                        std::push_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                        min_top_score = ws.dense_candidates[0].score;
                    }
                }
                if (ws.dense_candidates.size() == top_dense) {
                    std::sort_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                } else {
                    std::sort(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                }
            } else {
                const int max_threads = omp_get_max_threads();
                auto& shared_heaps = ws.thread_heaps;
                if (shared_heaps.size() < static_cast<std::size_t>(max_threads)) {
                    shared_heaps.resize(max_threads);
                }

                #pragma omp parallel shared(shared_heaps)
                {
                    const int tid = omp_get_thread_num();
                    auto& local_heap = shared_heaps[tid];
                    local_heap.clear();
                    local_heap.reserve(top_dense);
                    float local_min = -1e30f;

                    if (!check_del) {
                        #pragma omp for schedule(dynamic, 1024) nowait
                        for (std::size_t i = 0; i < total_slots; ++i) {
                            const int8_t* vec_ptr = arena_base + i * p_dim;
                            prefetch_l1(vec_ptr + p_dim * 2);

                            float sim = SimdInt8::dot_product_u8_s8_fast(
                                q_u8_ptr, q_i8_ptr, q_abs_ptr, vec_ptr, p_dim, sums_128[i], q_scale * scales[i]
                            );
                            const auto slot_u32 = static_cast<std::uint32_t>(i);
                            if (local_heap.size() < top_dense) {
                                local_heap.push_back(SearchHit{slot_u32, sim});
                                if (local_heap.size() == top_dense) {
                                    std::make_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                    local_min = local_heap[0].score;
                                }
                            } else if (sim > local_min) {
                                std::pop_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                local_heap.back() = SearchHit{slot_u32, sim};
                                std::push_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                local_min = local_heap[0].score;
                            }
                        }
                    } else {
                        #pragma omp for schedule(dynamic, 1024) nowait
                        for (std::size_t i = 0; i < total_slots; ++i) {
                            if (deleted[i]) {
                                continue;
                            }
                            const int8_t* vec_ptr = arena_base + i * p_dim;
                            prefetch_l1(vec_ptr + p_dim * 2);

                            float sim = SimdInt8::dot_product_u8_s8_fast(
                                q_u8_ptr, q_i8_ptr, q_abs_ptr, vec_ptr, p_dim, sums_128[i], q_scale * scales[i]
                            );
                            const auto slot_u32 = static_cast<std::uint32_t>(i);
                            if (local_heap.size() < top_dense) {
                                local_heap.push_back(SearchHit{slot_u32, sim});
                                if (local_heap.size() == top_dense) {
                                    std::make_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                    local_min = local_heap[0].score;
                                }
                            } else if (sim > local_min) {
                                std::pop_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                local_heap.back() = SearchHit{slot_u32, sim};
                                std::push_heap(local_heap.begin(), local_heap.end(), hit_greater);
                                local_min = local_heap[0].score;
                            }
                        }
                    }
                }

                for (const auto& local_heap : ws.thread_heaps) {
                    for (const auto& hit : local_heap) {
                        if (ws.dense_candidates.size() < top_dense) {
                            ws.dense_candidates.push_back(hit);
                            if (ws.dense_candidates.size() == top_dense) {
                                std::make_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                            }
                        } else if (hit.score > ws.dense_candidates[0].score ||
                                   (hit.score == ws.dense_candidates[0].score && hit.doc_id < ws.dense_candidates[0].doc_id)) {
                            std::pop_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                            ws.dense_candidates.back() = hit;
                            std::push_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                        }
                    }
                }
                if (ws.dense_candidates.size() == top_dense) {
                    std::sort_heap(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                } else {
                    std::sort(ws.dense_candidates.begin(), ws.dense_candidates.end(), hit_greater);
                }
            }
        }

        // 3. 收集并去重两路前 200 名的内部 slot_id
        ws.candidate_slots_buf.clear();
        float* __restrict rrf_scores = ws.rrf_scores_buf.data();

        for (std::size_t rank = 0; rank < ws.bm25_hits_buf.size(); ++rank) {
            std::uint32_t slot = ws.bm25_hits_buf[rank].doc_id;
            if (rrf_scores[slot] == 0.0f) {
                ws.candidate_slots_buf.push_back(slot);
            }
            rrf_scores[slot] += 1.0f / (k + static_cast<float>(rank + 1));
        }

        for (std::size_t rank = 0; rank < ws.dense_candidates.size(); ++rank) {
            std::uint32_t slot = ws.dense_candidates[rank].doc_id;
            if (rrf_scores[slot] == 0.0f) {
                ws.candidate_slots_buf.push_back(slot);
            }
            rrf_scores[slot] += 1.0f / (k + static_cast<float>(rank + 1));
        }

        // 4. 在 8 字节 SearchHit 上完成 Top-K 稳定排序，并原地复位 rrf_scores_buf
        ws.candidate_hits_buf.clear();
        for (std::uint32_t slot : ws.candidate_slots_buf) {
            ws.candidate_hits_buf.push_back(SearchHit{slot, rrf_scores[slot]});
            rrf_scores[slot] = 0.0f;
        }

        const std::size_t final_k = std::min(ws.candidate_hits_buf.size(), top_k);
        std::stable_sort(ws.candidate_hits_buf.begin(), ws.candidate_hits_buf.end(),
                         [](const SearchHit& a, const SearchHit& b) noexcept {
                             return a.score > b.score;
                         });

        // 5. 仅对最终胜出的 Top-K 槽位映射外部 doc_id 并拷贝文本
        std::vector<HitResult> results;
        results.reserve(final_k);
        for (std::size_t i = 0; i < final_k; ++i) {
            const auto& hit = ws.candidate_hits_buf[i];
            results.push_back(HitResult{
                slot_to_doc_id_[hit.doc_id],
                hit.score,
                doc_texts_[hit.doc_id]
            });
        }

        return results;
    }

    [[nodiscard]] std::size_t total_docs() const noexcept { return active_docs_; }
    [[nodiscard]] std::size_t document_count() const noexcept { return active_docs_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t memory_used() const noexcept { return arena_.used_bytes(); }

private:
    struct SearchWorkspace {
        std::vector<int8_t> q_int8;
        std::vector<uint8_t> q_abs;
        std::vector<uint8_t> q_u8;
        std::vector<SearchHit> bm25_hits_buf;
        std::vector<SearchHit> dense_candidates;
        std::vector<std::vector<SearchHit>> thread_heaps;
        std::vector<std::uint32_t> candidate_slots_buf;
        std::vector<SearchHit> candidate_hits_buf;
        std::vector<float> rrf_scores_buf;

        SearchWorkspace() {
            bm25_hits_buf.reserve(RRF_RANK_LIMIT);
            dense_candidates.reserve(RRF_RANK_LIMIT);
            candidate_slots_buf.reserve(RRF_RANK_LIMIT * 2);
            candidate_hits_buf.reserve(RRF_RANK_LIMIT * 2);
        }

        void ensure_dimension(std::size_t padded_dim, std::size_t total_slots) {
            if (q_int8.size() < padded_dim) {
                q_int8.assign(padded_dim, 0);
                q_abs.assign(padded_dim, 0);
                q_u8.assign(padded_dim, 128);
            }
            if (rrf_scores_buf.size() < total_slots) {
                rrf_scores_buf.resize(total_slots, 0.0f);
            }
        }
    };

    static void prefetch_l1(const void* addr) noexcept {
#if defined(__AVX2__) || defined(__SSE__)
        _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(addr, 0, 3);
#else
        (void)addr;
#endif
    }

    std::size_t dimension_;
    std::size_t padded_dim_;
    Arena arena_;
    InvertedIndex inverted_index_;

    std::vector<std::uint32_t> slot_to_doc_id_;
    std::vector<std::uint8_t> slot_deleted_;
    std::vector<float> active_scales_;
    std::vector<int32_t> active_sums_128_;
    std::vector<std::string> doc_texts_;
    std::unordered_map<std::uint32_t, std::uint32_t> doc_id_to_slot_;
    std::size_t active_docs_{0};
    bool has_deletions_{false};
};

} // namespace vrag
