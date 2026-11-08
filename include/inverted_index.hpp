#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    InvertedIndex() = default;

    // 索引一篇新文档 (text 为原始内容，doc_id 为文档唯一标号)
    void add_document(std::uint32_t doc_id, std::string_view text) {
        // 1. 统计当前文档中各个 Bi-gram 的本地词频 (Local Term Frequency)
        std::unordered_map<std::uint32_t, std::uint16_t> local_tf;
        std::uint32_t token_count = 0;

        auto tokens = Tokenizer::cut(text);
        for (std::string_view token : tokens) {
            std::uint32_t term_id = fnv1a_hash(token);
            local_tf[term_id]++;
            token_count++;
        }

        // 2. 记录文档长度元数据
        if (doc_id >= doc_lengths_.size()) {
            doc_lengths_.resize(doc_id + 1, 0);
        }
        doc_lengths_[doc_id] = token_count;
        total_tokens_ += token_count;
        total_docs_++;

        // 3. 追加写入全局倒排链表
        for (const auto& [term_id, tf] : local_tf) {
            index_[term_id].push_back(PostingEntry{doc_id, tf});
        }
    }

    // 基于标准 BM25 算法进行字面检索打分
    // k1 = 1.2f, b = 0.75f 为通用信息检索领域最优默认经验值
    std::vector<SearchHit> search_bm25(std::string_view query, std::size_t top_k = 10,
                                       float k1 = 1.2f, float b = 0.75f) const {
        if (total_docs_ == 0 || query.empty()) {
            return {};
        }

        const float avgdl = static_cast<float>(total_tokens_) / static_cast<float>(total_docs_);

        // 1. 提取 Query 的词项与词频
        std::unordered_map<std::uint32_t, std::uint16_t> query_tf;
        auto tokens = Tokenizer::cut(query);
        for (std::string_view token : tokens) {
            std::uint32_t term_id = fnv1a_hash(token);
            query_tf[term_id]++;
        }

        // 2. 累加各匹配文档的 BM25 分数
        std::unordered_map<std::uint32_t, float> doc_scores;

        for (const auto& [term_id, q_tf] : query_tf) {
            auto it = index_.find(term_id);
            if (it == index_.end()) {
                continue; // 当前词未在任何文档中出现
            }

            const auto& postings = it->second;
            std::size_t df = postings.size(); // 包含该词的文档数 (Document Frequency)

            // 计算 IDF (逆文档频率，采用 Lucene 平滑公式)
            float idf = std::log(1.0f + (static_cast<float>(total_docs_ - df) + 0.5f) /
                                           (static_cast<float>(df) + 0.5f));
            if (idf < 0.0f) {
                idf = 0.0f;
            }

            // 遍历包含该词的倒排链表并累加贡献
            for (const auto& entry : postings) {
                float tf = static_cast<float>(entry.tf);
                float doc_len = static_cast<float>(doc_lengths_[entry.doc_id]);

                // BM25 核心词频归一化公式
                float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * (doc_len / avgdl)));
                doc_scores[entry.doc_id] += idf * tf_norm;
            }
        }

        // 3. 收集并排序打分结果
        std::vector<SearchHit> results;
        results.reserve(doc_scores.size());
        for (const auto& [doc_id, score] : doc_scores) {
            results.push_back(SearchHit{doc_id, score});
        }

        // 降序排序
        std::sort(results.begin(), results.end(), [](const SearchHit& a, const SearchHit& b) {
            return a.score > b.score;
        });

        if (results.size() > top_k) {
            results.resize(top_k);
        }

        return results;
    }

    [[nodiscard]] std::size_t total_docs() const noexcept { return total_docs_; }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return index_.size(); }

private:
    // TermID (uint32_t) -> 紧凑倒排链表
    std::unordered_map<std::uint32_t, std::vector<PostingEntry>> index_;
    // 各文档的词元总数 (以 DocID 为下标直接索引)
    std::vector<std::uint32_t> doc_lengths_;
    std::size_t total_docs_{0};
    std::size_t total_tokens_{0};
};

} // namespace vrag
