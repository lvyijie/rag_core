#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <istream>
#include <memory>
#include <mutex>
#include <ostream>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "tokenizer.hpp"

namespace vrag {

// 紧凑的倒排链表条目
struct PostingEntry {
    std::uint32_t doc_id;
    std::uint16_t tf; // 该词在该文档中出现的频次
};

// 检索结果条目
struct SearchHit {
    std::uint32_t doc_id;
    float score;

    bool operator>(const SearchHit& other) const noexcept {
        return score > other.score;
    }
};

class InvertedIndex {
public:
    InvertedIndex()
        : ws_pool_(std::make_unique<PoolSlot[]>(kWorkspacePoolSize)) {
        index_.reserve(16384);
        scratch_terms_.reserve(512);
    }

    InvertedIndex(const InvertedIndex&) = delete;
    InvertedIndex& operator=(const InvertedIndex&) = delete;

    InvertedIndex(InvertedIndex&& other) noexcept
        : index_(std::move(other.index_)),
          doc_lengths_(std::move(other.doc_lengths_)),
          doc_present_(std::move(other.doc_present_)),
          doc_deleted_(std::move(other.doc_deleted_)),
          total_docs_(other.total_docs_),
          total_tokens_(other.total_tokens_),
          has_deletions_(other.has_deletions_),
          scratch_terms_(std::move(other.scratch_terms_)),
          doc_k_(std::move(other.doc_k_)),
          doc_norm_tf1_(std::move(other.doc_norm_tf1_)),
          cached_k1_(other.cached_k1_),
          cached_b_(other.cached_b_),
          norms_dirty_(other.norms_dirty_.load(std::memory_order_relaxed)),
          ws_pool_(std::move(other.ws_pool_)) {}

    InvertedIndex& operator=(InvertedIndex&& other) noexcept {
        if (this != &other) {
            index_ = std::move(other.index_);
            doc_lengths_ = std::move(other.doc_lengths_);
            doc_present_ = std::move(other.doc_present_);
            doc_deleted_ = std::move(other.doc_deleted_);
            total_docs_ = other.total_docs_;
            total_tokens_ = other.total_tokens_;
            has_deletions_ = other.has_deletions_;
            scratch_terms_ = std::move(other.scratch_terms_);
            doc_k_ = std::move(other.doc_k_);
            doc_norm_tf1_ = std::move(other.doc_norm_tf1_);
            cached_k1_ = other.cached_k1_;
            cached_b_ = other.cached_b_;
            norms_dirty_.store(other.norms_dirty_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ws_pool_ = std::move(other.ws_pool_);
        }
        return *this;
    }

    void reserve_docs(std::size_t doc_count) {
        doc_lengths_.reserve(doc_count);
        doc_present_.reserve(doc_count);
        doc_deleted_.reserve(doc_count);
        doc_k_.reserve(doc_count);
        doc_norm_tf1_.reserve(doc_count);
    }

    // 软删除指定文档（O(1) 墓碑标记，立即对后续查询生效）
    bool remove_document(std::uint32_t doc_id) {
        if (doc_id >= doc_present_.size() || !doc_present_[doc_id] || doc_deleted_[doc_id]) {
            return false;
        }
        doc_deleted_[doc_id] = 1;
        has_deletions_ = true;
        if (total_tokens_ >= doc_lengths_[doc_id]) {
            total_tokens_ -= doc_lengths_[doc_id];
        } else {
            total_tokens_ = 0;
        }
        doc_lengths_[doc_id] = 0;
        if (total_docs_ > 0) {
            --total_docs_;
        }
        norms_dirty_.store(true, std::memory_order_release);
        return true;
    }

    // 索引一篇新文档 (text 为原始内容，doc_id 为内部槽位号或文档唯一标号)
    void add_document(std::uint32_t doc_id, std::string_view text) {
        if (doc_id < doc_present_.size() && doc_present_[doc_id]) {
            if (!doc_deleted_[doc_id]) {
                if (total_tokens_ >= doc_lengths_[doc_id]) {
                    total_tokens_ -= doc_lengths_[doc_id];
                }
                if (total_docs_ > 0) {
                    --total_docs_;
                }
            }
            for (auto& [term_id, plist] : index_) {
                plist.erase(
                    std::remove_if(plist.begin(), plist.end(),
                                   [doc_id](const PostingEntry& e) noexcept { return e.doc_id == doc_id; }),
                    plist.end());
            }
            doc_deleted_[doc_id] = 0;
        }

        // 1. 使用复用缓冲区 + 排序归并统计本地词频
        scratch_terms_.clear();
        for (std::string_view token : Tokenizer::cut(text)) {
            scratch_terms_.push_back(fnv1a_hash(token));
        }

        const std::uint32_t token_count = static_cast<std::uint32_t>(scratch_terms_.size());

        // 2. 记录文档长度元数据
        if (doc_id >= doc_lengths_.size()) {
            doc_lengths_.resize(doc_id + 1, 0);
            doc_present_.resize(doc_id + 1, 0);
            doc_deleted_.resize(doc_id + 1, 0);
        }
        doc_lengths_[doc_id] = token_count;
        doc_present_[doc_id] = 1;
        doc_deleted_[doc_id] = 0;
        total_tokens_ += token_count;
        total_docs_++;
        norms_dirty_.store(true, std::memory_order_release);

        if (scratch_terms_.empty()) {
            return;
        }

        std::sort(scratch_terms_.begin(), scratch_terms_.end());

        // 3. 游程编码统计词频并追加写入全局倒排链表
        std::size_t i = 0;
        const std::size_t n = scratch_terms_.size();
        while (i < n) {
            const std::uint32_t term_id = scratch_terms_[i];
            std::size_t j = i + 1;
            while (j < n && scratch_terms_[j] == term_id) {
                ++j;
            }
            const auto tf = static_cast<std::uint16_t>(
                std::min<std::size_t>(j - i, 65535u)
            );
            index_[term_id].push_back(PostingEntry{doc_id, tf});
            i = j;
        }
    }

    // 线程安全 + 零堆分配热路径：将 BM25 Top-K 结果直接写入调用方提供的复用容器
    // allowed_mask: 可选前置过滤掩码（若非空，则仅允许 allowed_mask[doc_id] != 0 的槽位参与召回）
    void search_bm25_into(std::string_view query, std::size_t top_k,
                          std::vector<SearchHit>& results,
                          float k1 = 1.2f, float b = 0.75f,
                          const std::uint8_t* allowed_mask = nullptr) const {
        results.clear();
        if (total_docs_ == 0 || query.empty() || top_k == 0 ||
            k1 < 0.0f || b < 0.0f || b > 1.0f || !std::isfinite(k1) || !std::isfinite(b)) {
            return;
        }

        // 从无锁对象池租借工作区（避免 Windows MinGW thread_local 析构器堆损坏 Bug）
        ScopedWorkspace guard(ws_pool_.get());
        auto& ws = guard.ws;

        // 1. 提取并去重 Query 的词项
        ws.query_terms.clear();
        for (std::string_view token : Tokenizer::cut(query)) {
            ws.query_terms.push_back(fnv1a_hash(token));
        }
        if (ws.query_terms.empty()) {
            return;
        }
        std::sort(ws.query_terms.begin(), ws.query_terms.end());
        ws.query_terms.erase(std::unique(ws.query_terms.begin(), ws.query_terms.end()),
                             ws.query_terms.end());

        // 2. 双重检查锁定惰性更新 BM25 长度归一化表
        if (norms_dirty_.load(std::memory_order_acquire) || cached_k1_ != k1 || cached_b_ != b) {
            std::lock_guard<std::mutex> lock(norm_mutex_);
            if (norms_dirty_.load(std::memory_order_relaxed) || cached_k1_ != k1 || cached_b_ != b) {
                rebuild_norm_cache_locked(k1, b);
            }
        }

        const std::size_t max_slot = doc_lengths_.size();
        if (ws.score_buf.size() < max_slot) {
            ws.score_buf.resize(max_slot, -1.0f);
        }

        // 3. 使用扁平数组 score_buf + touched_docs 累加 BM25 分数
        ws.touched_docs.clear();
        const float* __restrict doc_k = doc_k_.data();
        const float* __restrict norm_tf1 = doc_norm_tf1_.data();
        const std::uint8_t* __restrict deleted = doc_deleted_.data();
        const bool check_del = has_deletions_;
        float* __restrict scores = ws.score_buf.data();

        // 检查是否存在高区分度词项 (df <= 50% N)；若存在，则自动跳过 df > 50% N 的高频停用词倒排链
        auto active_postings_df = [&](const std::vector<PostingEntry>& plist) noexcept -> std::size_t {
            if (!check_del) {
                return plist.size();
            }
            std::size_t cnt = 0;
            for (const auto& e : plist) {
                if (!deleted[e.doc_id]) {
                    ++cnt;
                }
            }
            return cnt;
        };

        bool has_informative_term = false;
        if (ws.query_terms.size() >= 3 && total_docs_ >= 32) {
            for (std::uint32_t term_id : ws.query_terms) {
                auto it = index_.find(term_id);
                if (it != index_.end() && !it->second.empty()) {
                    const std::size_t df = active_postings_df(it->second);
                    if (df > 0 && df * 2 <= total_docs_) {
                        has_informative_term = true;
                        break;
                    }
                }
            }
        }

        for (std::uint32_t term_id : ws.query_terms) {
            auto it = index_.find(term_id);
            if (it == index_.end()) {
                continue;
            }

            const auto& postings = it->second;
            const std::size_t df = active_postings_df(postings);
            if (df == 0 || (has_informative_term && df * 2 > total_docs_)) {
                continue;
            }

            const std::size_t eff_df = std::min(df, total_docs_);
            float idf = std::log(1.0f + (static_cast<float>(total_docs_ - eff_df) + 0.5f) /
                                           (static_cast<float>(eff_df) + 0.5f));
            if (idf < 0.0f) {
                idf = 0.0f;
            }

            const float idf_k1 = idf * (k1 + 1.0f);
            const PostingEntry* __restrict p_data = postings.data();
            const std::size_t p_size = postings.size();

            if (!check_del && !allowed_mask) {
                for (std::size_t idx = 0; idx < p_size; ++idx) {
                    const std::uint32_t doc_id = p_data[idx].doc_id;
                    const std::uint16_t raw_tf = p_data[idx].tf;

                    float delta;
                    if (raw_tf == 1) {
                        delta = idf * norm_tf1[doc_id];
                    } else {
                        const float tf = static_cast<float>(raw_tf);
                        delta = (idf_k1 * tf) / (tf + doc_k[doc_id]);
                    }

                    float cur = scores[doc_id];
                    if (cur < 0.0f) {
                        ws.touched_docs.push_back(doc_id);
                        cur = 0.0f;
                    }
                    scores[doc_id] = cur + delta;
                }
            } else {
                for (std::size_t idx = 0; idx < p_size; ++idx) {
                    const std::uint32_t doc_id = p_data[idx].doc_id;
                    if ((check_del && deleted[doc_id]) || (allowed_mask && !allowed_mask[doc_id])) {
                        continue;
                    }
                    const std::uint16_t raw_tf = p_data[idx].tf;

                    float delta;
                    if (raw_tf == 1) {
                        delta = idf * norm_tf1[doc_id];
                    } else {
                        const float tf = static_cast<float>(raw_tf);
                        delta = (idf_k1 * tf) / (tf + doc_k[doc_id]);
                    }

                    float cur = scores[doc_id];
                    if (cur < 0.0f) {
                        ws.touched_docs.push_back(doc_id);
                        cur = 0.0f;
                    }
                    scores[doc_id] = cur + delta;
                }
            }
        }

        // 4. 基于小顶堆阈值过滤收集 Top-K 命中的文档分数并原地复位 score_buf
        auto cmp = [](const SearchHit& a, const SearchHit& b) noexcept {
            return (a.score != b.score) ? (a.score > b.score) : (a.doc_id < b.doc_id);
        };
        results.reserve(std::min(ws.touched_docs.size(), top_k));
        float min_top_score = -1e30f;

        for (std::uint32_t doc_id : ws.touched_docs) {
            const float s = scores[doc_id];
            scores[doc_id] = -1.0f;

            if (results.size() < top_k) {
                results.push_back(SearchHit{doc_id, s});
                if (results.size() == top_k) {
                    std::make_heap(results.begin(), results.end(), cmp);
                    min_top_score = results[0].score;
                }
            } else if (s > min_top_score || (s == min_top_score && doc_id < results[0].doc_id)) {
                std::pop_heap(results.begin(), results.end(), cmp);
                results.back() = SearchHit{doc_id, s};
                std::push_heap(results.begin(), results.end(), cmp);
                min_top_score = results[0].score;
            }
        }

        if (results.size() == top_k) {
            std::sort_heap(results.begin(), results.end(), cmp);
        } else {
            std::sort(results.begin(), results.end(), cmp);
        }
    }

    // 基于标准 BM25 算法进行字面检索打分（线程安全 const 方法）
    std::vector<SearchHit> search_bm25(std::string_view query, std::size_t top_k = 10,
                                       float k1 = 1.2f, float b = 0.75f) const {
        std::vector<SearchHit> results;
        search_bm25_into(query, top_k, results, k1, b);
        return results;
    }

    bool save(std::ostream& os) const {
        const std::uint64_t slots = doc_lengths_.size();
        const std::uint64_t docs = total_docs_;
        const std::uint64_t tokens = total_tokens_;
        const std::uint8_t has_del = has_deletions_ ? 1 : 0;
        const std::uint64_t vocab = index_.size();

        os.write(reinterpret_cast<const char*>(&slots), sizeof(slots));
        os.write(reinterpret_cast<const char*>(&docs), sizeof(docs));
        os.write(reinterpret_cast<const char*>(&tokens), sizeof(tokens));
        os.write(reinterpret_cast<const char*>(&has_del), sizeof(has_del));
        os.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));

        if (slots > 0) {
            os.write(reinterpret_cast<const char*>(doc_lengths_.data()), slots * sizeof(std::uint32_t));
            os.write(reinterpret_cast<const char*>(doc_present_.data()), slots * sizeof(std::uint8_t));
            os.write(reinterpret_cast<const char*>(doc_deleted_.data()), slots * sizeof(std::uint8_t));
        }

        for (const auto& [term_id, postings] : index_) {
            const std::uint32_t p_len = static_cast<std::uint32_t>(postings.size());
            os.write(reinterpret_cast<const char*>(&term_id), sizeof(term_id));
            os.write(reinterpret_cast<const char*>(&p_len), sizeof(p_len));
            if (p_len > 0) {
                os.write(reinterpret_cast<const char*>(postings.data()), p_len * sizeof(PostingEntry));
            }
        }
        return static_cast<bool>(os);
    }

    bool load(std::istream& is) {
        std::uint64_t slots = 0;
        std::uint64_t docs = 0;
        std::uint64_t tokens = 0;
        std::uint8_t has_del = 0;
        std::uint64_t vocab = 0;

        if (!is.read(reinterpret_cast<char*>(&slots), sizeof(slots)) ||
            !is.read(reinterpret_cast<char*>(&docs), sizeof(docs)) ||
            !is.read(reinterpret_cast<char*>(&tokens), sizeof(tokens)) ||
            !is.read(reinterpret_cast<char*>(&has_del), sizeof(has_del)) ||
            !is.read(reinterpret_cast<char*>(&vocab), sizeof(vocab))) {
            return false;
        }

        total_docs_ = static_cast<std::size_t>(docs);
        total_tokens_ = static_cast<std::size_t>(tokens);
        has_deletions_ = (has_del != 0);

        doc_lengths_.resize(static_cast<std::size_t>(slots));
        doc_present_.resize(static_cast<std::size_t>(slots));
        doc_deleted_.resize(static_cast<std::size_t>(slots));

        if (slots > 0) {
            is.read(reinterpret_cast<char*>(doc_lengths_.data()), slots * sizeof(std::uint32_t));
            is.read(reinterpret_cast<char*>(doc_present_.data()), slots * sizeof(std::uint8_t));
            is.read(reinterpret_cast<char*>(doc_deleted_.data()), slots * sizeof(std::uint8_t));
        }

        index_.clear();
        index_.reserve(static_cast<std::size_t>(vocab));
        for (std::uint64_t i = 0; i < vocab; ++i) {
            std::uint32_t term_id = 0;
            std::uint32_t p_len = 0;
            is.read(reinterpret_cast<char*>(&term_id), sizeof(term_id));
            is.read(reinterpret_cast<char*>(&p_len), sizeof(p_len));
            auto& vec = index_[term_id];
            vec.resize(p_len);
            if (p_len > 0) {
                is.read(reinterpret_cast<char*>(vec.data()), p_len * sizeof(PostingEntry));
            }
        }

        norms_dirty_.store(true, std::memory_order_release);
        return static_cast<bool>(is);
    }

    void clear() {
        index_.clear();
        doc_lengths_.clear();
        doc_present_.clear();
        doc_deleted_.clear();
        doc_k_.clear();
        doc_norm_tf1_.clear();
        total_docs_ = 0;
        total_tokens_ = 0;
        has_deletions_ = false;
        norms_dirty_.store(true, std::memory_order_release);
    }

    // 根据旧槽位到新槽位的映射表紧凑重建倒排表，物理清除已删除槽位与空词项
    void compact(const std::vector<std::uint32_t>& old_to_new_slot, std::size_t new_slot_count) {
        std::vector<std::uint32_t> new_lengths(new_slot_count, 0);
        std::vector<std::uint8_t> new_present(new_slot_count, 1);
        std::vector<std::uint8_t> new_deleted(new_slot_count, 0);
        std::size_t new_total_tokens = 0;

        for (std::size_t old_s = 0; old_s < doc_lengths_.size() && old_s < old_to_new_slot.size(); ++old_s) {
            const std::uint32_t new_s = old_to_new_slot[old_s];
            if (new_s != UINT32_MAX) {
                new_lengths[new_s] = doc_lengths_[old_s];
                new_total_tokens += doc_lengths_[old_s];
            }
        }

        for (auto it = index_.begin(); it != index_.end(); ) {
            auto& plist = it->second;
            std::size_t write_idx = 0;
            for (std::size_t read_idx = 0; read_idx < plist.size(); ++read_idx) {
                const std::uint32_t old_s = plist[read_idx].doc_id;
                if (old_s < old_to_new_slot.size()) {
                    const std::uint32_t new_s = old_to_new_slot[old_s];
                    if (new_s != UINT32_MAX) {
                        plist[write_idx++] = PostingEntry{new_s, plist[read_idx].tf};
                    }
                }
            }
            if (write_idx == 0) {
                it = index_.erase(it);
            } else {
                plist.resize(write_idx);
                plist.shrink_to_fit();
                ++it;
            }
        }

        doc_lengths_ = std::move(new_lengths);
        doc_present_ = std::move(new_present);
        doc_deleted_ = std::move(new_deleted);
        total_docs_ = new_slot_count;
        total_tokens_ = new_total_tokens;
        has_deletions_ = false;
        norms_dirty_.store(true, std::memory_order_release);
    }

    [[nodiscard]] std::size_t total_docs() const noexcept { return total_docs_; }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return index_.size(); }

    // 估算倒排索引结构占用的内存字节数（含 PostingEntry 数组、词典节点与文档长度表）
    [[nodiscard]] std::size_t memory_bytes() const noexcept {
        std::size_t bytes = (doc_lengths_.capacity() + doc_k_.capacity() + doc_norm_tf1_.capacity()) * sizeof(float)
                          + (doc_present_.capacity() + doc_deleted_.capacity()) * sizeof(std::uint8_t);
        for (const auto& [term_id, plist] : index_) {
            (void)term_id;
            bytes += 32 + plist.capacity() * sizeof(PostingEntry);
        }
        return bytes;
    }

private:
    static constexpr std::size_t kWorkspacePoolSize = 32;

    struct ThreadWorkspace {
        std::vector<std::uint32_t> query_terms;
        std::vector<float> score_buf;
        std::vector<std::uint32_t> touched_docs;

        ThreadWorkspace() {
            query_terms.reserve(64);
            touched_docs.reserve(2048);
        }
    };

    struct PoolSlot {
        std::atomic<bool> in_use{false};
        ThreadWorkspace ws;
    };

    struct ScopedWorkspace {
        PoolSlot* slot{nullptr};
        std::unique_ptr<ThreadWorkspace> fallback;
        ThreadWorkspace& ws;

        explicit ScopedWorkspace(PoolSlot* pool) : ws(acquire(pool)) {}
        ~ScopedWorkspace() noexcept {
            if (slot) {
                slot->in_use.store(false, std::memory_order_release);
            }
        }

        ScopedWorkspace(const ScopedWorkspace&) = delete;
        ScopedWorkspace& operator=(const ScopedWorkspace&) = delete;

    private:
        ThreadWorkspace& acquire(PoolSlot* pool) {
            for (std::size_t i = 0; i < kWorkspacePoolSize; ++i) {
                if (!pool[i].in_use.load(std::memory_order_relaxed) &&
                    !pool[i].in_use.exchange(true, std::memory_order_acquire)) {
                    slot = &pool[i];
                    return slot->ws;
                }
            }
            fallback = std::make_unique<ThreadWorkspace>();
            return *fallback;
        }
    };

    void rebuild_norm_cache_locked(float k1, float b) const {
        const std::size_t n = doc_lengths_.size();
        doc_k_.resize(n);
        doc_norm_tf1_.resize(n);

        const float avgdl = (total_docs_ > 0)
            ? (static_cast<float>(total_tokens_) / static_cast<float>(total_docs_))
            : 1.0f;
        const float c0 = k1 * (1.0f - b);
        const float c1 = (avgdl > 0.0f) ? (k1 * b / avgdl) : 0.0f;
        const float k1_plus_1 = k1 + 1.0f;

        for (std::size_t i = 0; i < n; ++i) {
            const float K = c0 + c1 * static_cast<float>(doc_lengths_[i]);
            doc_k_[i] = K;
            doc_norm_tf1_[i] = k1_plus_1 / (1.0f + K);
        }

        cached_k1_ = k1;
        cached_b_ = b;
        norms_dirty_.store(false, std::memory_order_release);
    }

    std::unordered_map<std::uint32_t, std::vector<PostingEntry>> index_;
    std::vector<std::uint32_t> doc_lengths_;
    std::vector<std::uint8_t> doc_present_;
    std::vector<std::uint8_t> doc_deleted_;
    std::size_t total_docs_{0};
    std::size_t total_tokens_{0};
    bool has_deletions_{false};

    std::vector<std::uint32_t> scratch_terms_;
    mutable std::mutex norm_mutex_;
    mutable std::vector<float> doc_k_;
    mutable std::vector<float> doc_norm_tf1_;
    mutable float cached_k1_{1.2f};
    mutable float cached_b_{0.75f};
    mutable std::atomic<bool> norms_dirty_{true};
    mutable std::unique_ptr<PoolSlot[]> ws_pool_;
};

} // namespace vrag
