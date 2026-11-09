#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <span>
#include <streambuf>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <omp.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "arena.hpp"
#include "inverted_index.hpp"
#include "simd_vector.hpp"
#include "simd_int8.hpp"
#include "tokenizer.hpp"

namespace vrag {

namespace detail {

// 单遍流式 64 位 FNV-1a 校验和缓冲器（零额外内存拷贝、零二次文件读取）
class HashingStreamBuf final : public std::streambuf {
public:
    static constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
    static constexpr std::uint64_t kPrime       = 1099511628211ULL;

    explicit HashingStreamBuf(std::streambuf* dest) noexcept : dest_(dest) {}

    [[nodiscard]] std::uint64_t hash() const noexcept { return hash_; }

protected:
    int_type overflow(int_type ch) override {
        if (ch == traits_type::eof()) {
            return traits_type::not_eof(ch);
        }
        const auto b = static_cast<unsigned char>(ch);
        hash_ = (hash_ ^ b) * kPrime;
        return dest_->sputc(static_cast<char>(ch));
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override {
        const std::streamsize written = dest_->sputn(s, count);
        for (std::streamsize i = 0; i < written; ++i) {
            hash_ = (hash_ ^ static_cast<unsigned char>(s[i])) * kPrime;
        }
        return written;
    }

private:
    std::streambuf* dest_;
    std::uint64_t hash_{kOffsetBasis};
};

inline bool verify_file_fnv1a64(std::ifstream& ifs, std::uint64_t payload_bytes, std::uint64_t expected_hash) {
    ifs.seekg(0, std::ios::beg);
    if (!ifs) {
        return false;
    }
    std::uint64_t hash = HashingStreamBuf::kOffsetBasis;
    std::array<char, 65536> buf{};
    std::uint64_t remaining = payload_bytes;
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buf.size()));
        if (!ifs.read(buf.data(), chunk)) {
            return false;
        }
        for (std::streamsize i = 0; i < chunk; ++i) {
            hash = (hash ^ static_cast<unsigned char>(buf[i])) * HashingStreamBuf::kPrime;
        }
        remaining -= static_cast<std::uint64_t>(chunk);
    }
    return hash == expected_hash;
}

inline bool atomic_replace_file(const std::filesystem::path& tmp_path, const std::filesystem::path& target_path) {
#if defined(_WIN32)
    if (MoveFileExW(tmp_path.c_str(), target_path.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0) {
        return true;
    }
#endif
    std::error_code ec;
    std::filesystem::rename(tmp_path, target_path, ec);
    if (!ec) {
        return true;
    }
    std::filesystem::remove(target_path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, target_path, ec);
    return !ec;
}

} // namespace detail

enum class SearchMode : std::uint8_t {
    Hybrid = 0, // 双路召回 + Weighted RRF 融合（某一路输入为空或权重 <= 0 时自动退化为单路）
    BM25   = 1, // 仅执行稀疏 BM25 检索（score 直接为原始 BM25 得分）
    Dense  = 2  // 仅执行稠密 SQ8 向量检索（score 直接为余弦相似度近似得分）
};

struct HitResult {
    std::uint32_t doc_id{0};
    float score{0.0f};             // 最终排序得分（Hybrid 下为加权 RRF 分或 Reranker 分；单路下为对应单路得分）
    std::string text;              // 文档原文（拥有所有权，防止悬垂指针）
    float bm25_score{0.0f};        // 原始 BM25 得分（未命中或未启用时为 0.0f）
    float dense_score{0.0f};       // 原始 Dense SQ8 余弦相似度得分（未命中或未启用时为 0.0f）
    std::uint32_t bm25_rank{0};    // 1-based BM25 排名（0 表示未进入 BM25 候选）
    std::uint32_t dense_rank{0};   // 1-based Dense 排名（0 表示未进入 Dense 候选）
    std::string metadata;          // 文档可选元数据（如来源文件、标签、JSON 属性等）

    bool operator>(const HitResult& other) const noexcept {
        return score > other.score;
    }
};

using SearchResult = HitResult;

// 前置过滤谓词：接收 (doc_id, metadata)，返回 true 表示允许参与召回，false 表示剔除
using FilterPredicate = std::function<bool(std::uint32_t doc_id, std::string_view metadata)>;

// 外部重排序回调：接收 (query_text, candidates)，允许外部模型/规则就地更新 score 并重新排序
using RerankCallback = std::function<void(std::string_view query_text, std::vector<HitResult>& candidates)>;

struct SearchOptions {
    std::size_t top_k = 5;
    SearchMode mode = SearchMode::Hybrid;
    float rrf_k = 60.0f;
    float bm25_weight = 1.0f;
    float dense_weight = 1.0f;
    std::size_t candidate_limit = 200;
    float min_score = -1e30f;
    float bm25_k1 = 1.2f;
    float bm25_b = 0.75f;
    FilterPredicate filter = nullptr;
    RerankCallback reranker = nullptr;
};

class VragEngine {
public:
    static constexpr std::size_t RRF_RANK_LIMIT = 200;

    explicit VragEngine(std::size_t dimension = 768,
                        std::size_t arena_capacity = 256ULL * 1024 * 1024)
        : dimension_(dimension),
          padded_dim_((dimension + 63) & ~std::size_t(63)),
          arena_(arena_capacity),
          ws_pool_(std::make_unique<PoolSlot[]>(kWorkspacePoolSize)) {}

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

        const std::size_t target_slots = slot_to_doc_id_.size() + doc_count;
        arena_.reserve(target_slots * padded_dim_);
        slot_to_doc_id_.reserve(target_slots);
        slot_deleted_.reserve(target_slots);
        active_scales_.reserve(target_slots);
        active_sums_128_.reserve(target_slots);
        doc_texts_.reserve(target_slots);
        doc_metadata_.reserve(target_slots);
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

    // 将已构建完成的索引（已量化 INT8 向量池 + 扁平倒排表 + 原文 + 元数据）直接序列化落盘
    // P2 加固：写入临时文件 (.tmp) + 64 位 FNV-1a 完整性校验和尾部 + 原子替换目标文件
    bool save_index(const std::string& filepath) const {
        if (filepath.empty()) {
            return false;
        }

        const std::filesystem::path target_path(filepath);
        const std::filesystem::path tmp_path(filepath + ".tmp");

        {
            std::ofstream raw_ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!raw_ofs.is_open()) {
                return false;
            }

            detail::HashingStreamBuf hash_buf(raw_ofs.rdbuf());
            std::ostream ofs(&hash_buf);

            static constexpr char kMagic[8] = {'V', 'R', 'A', 'G', 'I', 'D', 'X', '3'};
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
                    const std::uint32_t m_len = static_cast<std::uint32_t>(doc_metadata_[i].size());
                    ofs.write(reinterpret_cast<const char*>(&m_len), sizeof(m_len));
                    if (m_len > 0) {
                        ofs.write(doc_metadata_[i].data(), m_len);
                    }
                }
            }

            if (!inverted_index_.save(ofs) || !ofs.good()) {
                raw_ofs.close();
                std::error_code ec;
                std::filesystem::remove(tmp_path, ec);
                return false;
            }

            // 写入 16 字节尾部：8 字节 Footer Magic + 8 字节 64 位 FNV-1a 校验和
            static constexpr char kFooterMagic[8] = {'V', 'R', 'A', 'G', 'E', 'N', 'D', '3'};
            const std::uint64_t checksum = hash_buf.hash();
            raw_ofs.write(kFooterMagic, sizeof(kFooterMagic));
            raw_ofs.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
            raw_ofs.flush();

            if (!raw_ofs.good()) {
                raw_ofs.close();
                std::error_code ec;
                std::filesystem::remove(tmp_path, ec);
                return false;
            }
        }

        if (!detail::atomic_replace_file(tmp_path, target_path)) {
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        return true;
    }

    // 极速加载预构建索引快照（支持 VRAGIDX1 / VRAGIDX2 / VRAGIDX3，具备 64 位校验和验证与事务性状态保护）
    bool load_index(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            return false;
        }

        const auto end_pos = ifs.tellg();
        if (end_pos < static_cast<std::streamoff>(41)) {
            return false;
        }
        const auto file_size = static_cast<std::uint64_t>(end_pos);
        ifs.seekg(0, std::ios::beg);

        char magic[8]{};
        if (!ifs.read(magic, sizeof(magic))) {
            return false;
        }
        const bool is_v1 = (std::memcmp(magic, "VRAGIDX1", 8) == 0);
        const bool is_v2 = (std::memcmp(magic, "VRAGIDX2", 8) == 0);
        const bool is_v3 = (std::memcmp(magic, "VRAGIDX3", 8) == 0);
        if (!is_v1 && !is_v2 && !is_v3) {
            return false;
        }

        // 对于 VRAGIDX3 格式，在分配任何内存前首先验证尾部 Magic 与全文 64 位 FNV-1a 校验和
        if (is_v3) {
            if (file_size < 41 + 16) {
                return false;
            }
            const std::uint64_t payload_bytes = file_size - 16;
            ifs.seekg(static_cast<std::streamoff>(payload_bytes), std::ios::beg);
            char footer_magic[8]{};
            std::uint64_t expected_checksum = 0;
            if (!ifs.read(footer_magic, sizeof(footer_magic)) ||
                std::memcmp(footer_magic, "VRAGEND3", 8) != 0 ||
                !ifs.read(reinterpret_cast<char*>(&expected_checksum), sizeof(expected_checksum))) {
                return false;
            }
            if (!detail::verify_file_fnv1a64(ifs, payload_bytes, expected_checksum)) {
                return false;
            }
            ifs.seekg(8, std::ios::beg);
        }

        std::uint64_t dim = 0;
        std::uint64_t p_dim = 0;
        std::uint64_t slots = 0;
        std::uint64_t active = 0;
        std::uint8_t has_del = 0;

        if (!ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim)) ||
            !ifs.read(reinterpret_cast<char*>(&p_dim), sizeof(p_dim)) ||
            !ifs.read(reinterpret_cast<char*>(&slots), sizeof(slots)) ||
            !ifs.read(reinterpret_cast<char*>(&active), sizeof(active)) ||
            !ifs.read(reinterpret_cast<char*>(&has_del), sizeof(has_del))) {
            return false;
        }

        if (dim != dimension_ || p_dim != padded_dim_ || active > slots) {
            return false;
        }

        // 防止损坏的头部导致恶意超大内存分配
        const std::uint64_t min_slot_bytes = p_dim + sizeof(std::uint32_t) + sizeof(std::uint8_t) + sizeof(float) + sizeof(int32_t);
        if (slots > 0 && slots > (file_size / min_slot_bytes)) {
            return false;
        }

        // 先加载到临时状态，全部成功后再提交到当前实例（保证异常时原实例数据不受损）
        const std::size_t n = static_cast<std::size_t>(slots);
        std::vector<std::uint32_t> tmp_slot_to_doc_id(n);
        std::vector<std::uint8_t> tmp_slot_deleted(n);
        std::vector<float> tmp_active_scales(n);
        std::vector<int32_t> tmp_active_sums_128(n);
        std::vector<std::string> tmp_doc_texts(n);
        std::vector<std::string> tmp_doc_metadata(n);
        Arena tmp_arena(std::max<std::size_t>(n * padded_dim_, 4096));
        tmp_arena.resize_raw(n * padded_dim_);

        if (n > 0) {
            ifs.read(reinterpret_cast<char*>(tmp_slot_to_doc_id.data()), n * sizeof(std::uint32_t));
            ifs.read(reinterpret_cast<char*>(tmp_slot_deleted.data()), n * sizeof(std::uint8_t));
            ifs.read(reinterpret_cast<char*>(tmp_active_scales.data()), n * sizeof(float));
            ifs.read(reinterpret_cast<char*>(tmp_active_sums_128.data()), n * sizeof(int32_t));
            ifs.read(reinterpret_cast<char*>(tmp_arena.data()), n * padded_dim_);
            if (!ifs) {
                return false;
            }

            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t len = 0;
                if (!ifs.read(reinterpret_cast<char*>(&len), sizeof(len)) || len > file_size) {
                    return false;
                }
                tmp_doc_texts[i].resize(len);
                if (len > 0 && !ifs.read(tmp_doc_texts[i].data(), len)) {
                    return false;
                }
                if (is_v2 || is_v3) {
                    std::uint32_t m_len = 0;
                    if (!ifs.read(reinterpret_cast<char*>(&m_len), sizeof(m_len)) || m_len > file_size) {
                        return false;
                    }
                    tmp_doc_metadata[i].resize(m_len);
                    if (m_len > 0 && !ifs.read(tmp_doc_metadata[i].data(), m_len)) {
                        return false;
                    }
                }
            }
        }

        InvertedIndex tmp_inverted;
        if (!tmp_inverted.load(ifs)) {
            return false;
        }

        std::unordered_map<std::uint32_t, std::uint32_t> tmp_doc_id_to_slot;
        tmp_doc_id_to_slot.reserve(static_cast<std::size_t>(active));
        for (std::size_t i = 0; i < n; ++i) {
            if (!tmp_slot_deleted[i]) {
                tmp_doc_id_to_slot[tmp_slot_to_doc_id[i]] = static_cast<std::uint32_t>(i);
            }
        }

        // 全部校验通过，无异常提交至当前引擎实例
        arena_ = std::move(tmp_arena);
        inverted_index_ = std::move(tmp_inverted);
        slot_to_doc_id_ = std::move(tmp_slot_to_doc_id);
        slot_deleted_ = std::move(tmp_slot_deleted);
        active_scales_ = std::move(tmp_active_scales);
        active_sums_128_ = std::move(tmp_active_sums_128);
        doc_texts_ = std::move(tmp_doc_texts);
        doc_metadata_ = std::move(tmp_doc_metadata);
        doc_id_to_slot_ = std::move(tmp_doc_id_to_slot);
        active_docs_ = static_cast<std::size_t>(active);
        has_deletions_ = (has_del != 0);
        return true;
    }

    // 插入或更新单篇文档 (Upsert：支持可选 metadata 属性)
    bool add_document(std::uint32_t doc_id,
                      std::string_view text,
                      std::span<const float> raw_vector,
                      std::string_view metadata = "") {
        if (raw_vector.size() != dimension_) {
            return false;
        }
        for (float v : raw_vector) {
            if (!std::isfinite(v)) {
                return false;
            }
        }

        auto existing = doc_id_to_slot_.find(doc_id);
        if (existing != doc_id_to_slot_.end()) {
            const std::uint32_t old_slot = existing->second;
            slot_deleted_[old_slot] = 1;
            has_deletions_ = true;
            inverted_index_.remove_document(old_slot);
            doc_texts_[old_slot].clear();
            doc_texts_[old_slot].shrink_to_fit();
            doc_metadata_[old_slot].clear();
            doc_metadata_[old_slot].shrink_to_fit();
            if (active_docs_ > 0) {
                --active_docs_;
            }
        }

        const std::uint32_t slot_id = static_cast<std::uint32_t>(slot_to_doc_id_.size());

        int8_t* ptr = arena_.allocate<int8_t>(padded_dim_, 32);
        if (!ptr) {
            return false;
        }
        if (padded_dim_ > dimension_) {
            std::memset(ptr + dimension_, 0, padded_dim_ - dimension_);
        }
        std::span<int8_t> int8_span(ptr, dimension_);

        int32_t sum_128 = 0;
        float scale = SimdInt8::normalize_and_quantize(raw_vector, int8_span, nullptr, nullptr, &sum_128);

        slot_to_doc_id_.push_back(doc_id);
        slot_deleted_.push_back(0);
        active_scales_.push_back(scale);
        active_sums_128_.push_back(sum_128);
        doc_texts_.emplace_back(text);
        doc_metadata_.emplace_back(metadata);
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
        doc_metadata_[slot_id].clear();
        doc_metadata_[slot_id].shrink_to_fit();
        if (active_docs_ > 0) {
            --active_docs_;
        }
        return true;
    }

    [[nodiscard]] bool contains_document(std::uint32_t doc_id) const noexcept {
        return doc_id_to_slot_.find(doc_id) != doc_id_to_slot_.end();
    }

    [[nodiscard]] std::string get_metadata(std::uint32_t doc_id) const {
        auto it = doc_id_to_slot_.find(doc_id);
        return (it != doc_id_to_slot_.end()) ? doc_metadata_[it->second] : std::string{};
    }

    // 统一检索主接口：支持 SearchMode (Hybrid / BM25 / Dense)、Weighted RRF、Metadata Filter 与 Reranker
    std::vector<HitResult> search(std::string_view query_text,
                                  std::span<const float> query_vector,
                                  const SearchOptions& options) const {
        if (active_docs_ == 0 || options.top_k == 0) {
            return {};
        }
        if (options.mode == SearchMode::Hybrid &&
            (options.rrf_k <= 0.0f || !std::isfinite(options.rrf_k))) {
            return {};
        }
        if (!std::isfinite(options.bm25_weight) || !std::isfinite(options.dense_weight)) {
            return {};
        }

        const std::size_t total_slots = slot_to_doc_id_.size();
        const std::size_t cand_limit = std::max(options.top_k,
            options.candidate_limit > 0 ? options.candidate_limit : RRF_RANK_LIMIT);

        ScopedWorkspace guard(ws_pool_.get());
        auto& ws = guard.ws;
        ws.ensure_dimension(padded_dim_, total_slots);

        // 0. 若用户配置了前置 Metadata Filter，在主线程一次性评估所有活跃槽位掩码
        const std::uint8_t* allowed_mask = nullptr;
        if (options.filter) {
            ws.filter_mask_buf.resize(total_slots);
            std::size_t allowed_count = 0;
            for (std::size_t i = 0; i < total_slots; ++i) {
                if (slot_deleted_[i]) {
                    ws.filter_mask_buf[i] = 0;
                } else {
                    const bool pass = options.filter(slot_to_doc_id_[i], doc_metadata_[i]);
                    ws.filter_mask_buf[i] = pass ? 1 : 0;
                    if (pass) ++allowed_count;
                }
            }
            if (allowed_count == 0) {
                return {};
            }
            allowed_mask = ws.filter_mask_buf.data();
        }

        const bool want_bm25 = (options.mode == SearchMode::BM25) ||
                               (options.mode == SearchMode::Hybrid && options.bm25_weight > 0.0f);
        const bool want_dense = (options.mode == SearchMode::Dense) ||
                                (options.mode == SearchMode::Hybrid && options.dense_weight > 0.0f);

        // 1. 右路：BM25 零分配召回
        ws.bm25_hits_buf.clear();
        if (want_bm25 && !query_text.empty()) {
            inverted_index_.search_bm25_into(
                query_text, cand_limit, ws.bm25_hits_buf,
                options.bm25_k1, options.bm25_b, allowed_mask);
        }

        // 2. 左路：检查查询向量是否有效并执行融合归一化+量化 + SIMD 扫描
        bool has_dense_query = false;
        if (want_dense && query_vector.size() == dimension_) {
            bool all_finite = true;
            for (float v : query_vector) {
                if (!std::isfinite(v)) {
                    all_finite = false;
                    break;
                }
                if (v != 0.0f) {
                    has_dense_query = true;
                }
            }
            if (!all_finite) {
                has_dense_query = false;
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
            const std::size_t top_dense = std::min(active_docs_, cand_limit);
            const bool check_del = has_deletions_;

            auto hit_greater = [](const SearchHit& a, const SearchHit& b) noexcept {
                return (a.score != b.score) ? (a.score > b.score) : (a.doc_id < b.doc_id);
            };

            if (total_slots <= 4096) {
                ws.dense_candidates.reserve(top_dense);
                float min_top_score = -1e30f;

                for (std::size_t i = 0; i < total_slots; ++i) {
                    if ((check_del && deleted[i]) || (allowed_mask && !allowed_mask[i])) {
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

                    const auto slots_signed = static_cast<std::ptrdiff_t>(total_slots);
                    if (!check_del && !allowed_mask) {
                        #pragma omp for schedule(dynamic, 1024) nowait
                        for (std::ptrdiff_t i = 0; i < slots_signed; ++i) {
                            const int8_t* vec_ptr = arena_base + static_cast<std::size_t>(i) * p_dim;
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
                        for (std::ptrdiff_t i = 0; i < slots_signed; ++i) {
                            if ((check_del && deleted[i]) || (allowed_mask && !allowed_mask[i])) {
                                continue;
                            }
                            const int8_t* vec_ptr = arena_base + static_cast<std::size_t>(i) * p_dim;
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

                for (const auto& local_heap : shared_heaps) {
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

        // 3. 收集并融合分数（记录每篇候选文档的 bm25_score, dense_score, bm25_rank, dense_rank）
        ws.candidate_slots_buf.clear();
        float* __restrict rrf_scores = ws.rrf_scores_buf.data();
        float* __restrict bm25_raw = ws.bm25_raw_score_buf.data();
        float* __restrict dense_raw = ws.dense_raw_score_buf.data();
        std::uint32_t* __restrict bm25_rk = ws.bm25_rank_buf.data();
        std::uint32_t* __restrict dense_rk = ws.dense_rank_buf.data();

        const float k = options.rrf_k;
        const float w_bm25 = options.bm25_weight;
        const float w_dense = options.dense_weight;

        for (std::size_t rank = 0; rank < ws.bm25_hits_buf.size(); ++rank) {
            const std::uint32_t slot = ws.bm25_hits_buf[rank].doc_id;
            const float raw_s = ws.bm25_hits_buf[rank].score;
            if (bm25_rk[slot] == 0 && dense_rk[slot] == 0) {
                ws.candidate_slots_buf.push_back(slot);
            }
            bm25_raw[slot] = raw_s;
            bm25_rk[slot] = static_cast<std::uint32_t>(rank + 1);
            if (options.mode == SearchMode::BM25) {
                rrf_scores[slot] = raw_s;
            } else {
                rrf_scores[slot] += w_bm25 / (k + static_cast<float>(rank + 1));
            }
        }

        for (std::size_t rank = 0; rank < ws.dense_candidates.size(); ++rank) {
            const std::uint32_t slot = ws.dense_candidates[rank].doc_id;
            const float raw_s = ws.dense_candidates[rank].score;
            if (bm25_rk[slot] == 0 && dense_rk[slot] == 0) {
                ws.candidate_slots_buf.push_back(slot);
            }
            dense_raw[slot] = raw_s;
            dense_rk[slot] = static_cast<std::uint32_t>(rank + 1);
            if (options.mode == SearchMode::Dense) {
                rrf_scores[slot] = raw_s;
            } else {
                rrf_scores[slot] += w_dense / (k + static_cast<float>(rank + 1));
            }
        }

        // 4. 在 8 字节 SearchHit 上完成阈值过滤与稳定排序
        ws.candidate_hits_buf.clear();
        for (std::uint32_t slot : ws.candidate_slots_buf) {
            const float final_s = rrf_scores[slot];
            rrf_scores[slot] = 0.0f;
            if (final_s >= options.min_score) {
                ws.candidate_hits_buf.push_back(SearchHit{slot, final_s});
            } else {
                bm25_raw[slot] = 0.0f;
                dense_raw[slot] = 0.0f;
                bm25_rk[slot] = 0;
                dense_rk[slot] = 0;
            }
        }

        std::stable_sort(ws.candidate_hits_buf.begin(), ws.candidate_hits_buf.end(),
                         [](const SearchHit& a, const SearchHit& b) noexcept {
                             return a.score > b.score;
                         });

        // 5. 构造返回结果（若配置了 reranker，则将候选池交给 reranker 重排后再截断至 top_k）
        const std::size_t build_k = options.reranker
            ? std::min(ws.candidate_hits_buf.size(), cand_limit)
            : std::min(ws.candidate_hits_buf.size(), options.top_k);

        std::vector<HitResult> results;
        results.reserve(build_k);
        for (std::size_t i = 0; i < build_k; ++i) {
            const auto& hit = ws.candidate_hits_buf[i];
            const std::uint32_t slot = hit.doc_id;
            results.push_back(HitResult{
                slot_to_doc_id_[slot],
                hit.score,
                doc_texts_[slot],
                bm25_raw[slot],
                dense_raw[slot],
                bm25_rk[slot],
                dense_rk[slot],
                doc_metadata_[slot]
            });
            bm25_raw[slot] = 0.0f;
            dense_raw[slot] = 0.0f;
            bm25_rk[slot] = 0;
            dense_rk[slot] = 0;
        }
        // 清理未进入 build_k 的剩余槽位状态
        for (std::size_t i = build_k; i < ws.candidate_hits_buf.size(); ++i) {
            const std::uint32_t slot = ws.candidate_hits_buf[i].doc_id;
            bm25_raw[slot] = 0.0f;
            dense_raw[slot] = 0.0f;
            bm25_rk[slot] = 0;
            dense_rk[slot] = 0;
        }

        if (options.reranker && !results.empty()) {
            options.reranker(query_text, results);
            if (results.size() > options.top_k) {
                results.resize(options.top_k);
            }
        }

        return results;
    }

    // 简写兼容接口（保持与旧版签名 100% 兼容）
    std::vector<HitResult> search(std::string_view query_text,
                                  std::span<const float> query_vector,
                                  std::size_t top_k = 5,
                                  float k = 60.0f) const {
        SearchOptions opts;
        opts.top_k = top_k;
        opts.mode = SearchMode::Hybrid;
        opts.rrf_k = k;
        return search(query_text, query_vector, opts);
    }

    // 便捷纯 BM25 检索接口（直接返回 BM25 原始得分）
    std::vector<HitResult> search_bm25(std::string_view query_text,
                                       std::size_t top_k = 10,
                                       FilterPredicate filter = nullptr) const {
        SearchOptions opts;
        opts.top_k = top_k;
        opts.mode = SearchMode::BM25;
        opts.filter = std::move(filter);
        return search(query_text, std::span<const float>{}, opts);
    }

    // 便捷纯 Dense SQ8 向量检索接口（直接返回余弦相似度得分）
    std::vector<HitResult> search_dense(std::span<const float> query_vector,
                                        std::size_t top_k = 10,
                                        FilterPredicate filter = nullptr) const {
        SearchOptions opts;
        opts.top_k = top_k;
        opts.mode = SearchMode::Dense;
        opts.filter = std::move(filter);
        return search("", query_vector, opts);
    }

    // 墓碑空间压缩：物理清理所有已删除/已覆盖的旧槽位，紧凑重建 Arena 向量池与倒排表
    void compact() {
        if (!has_deletions_) {
            return;
        }

        const std::size_t old_slots = slot_to_doc_id_.size();
        const std::size_t new_slots = active_docs_;

        std::vector<std::uint32_t> old_to_new(old_slots, UINT32_MAX);
        std::vector<std::uint32_t> new_slot_to_doc_id;
        std::vector<std::uint8_t> new_slot_deleted(new_slots, 0);
        std::vector<float> new_scales;
        std::vector<int32_t> new_sums_128;
        std::vector<std::string> new_texts;
        std::vector<std::string> new_metadata;
        std::unordered_map<std::uint32_t, std::uint32_t> new_doc_id_to_slot;

        new_slot_to_doc_id.reserve(new_slots);
        new_scales.reserve(new_slots);
        new_sums_128.reserve(new_slots);
        new_texts.reserve(new_slots);
        new_metadata.reserve(new_slots);
        new_doc_id_to_slot.reserve(new_slots);

        Arena new_arena(std::max<std::size_t>(new_slots * padded_dim_, 4096));
        const int8_t* old_arena_base = reinterpret_cast<const int8_t*>(arena_.data());

        for (std::size_t i = 0; i < old_slots; ++i) {
            if (slot_deleted_[i]) {
                continue;
            }
            const auto new_s = static_cast<std::uint32_t>(new_slot_to_doc_id.size());
            old_to_new[i] = new_s;

            int8_t* dst = new_arena.allocate<int8_t>(padded_dim_, 32);
            std::memcpy(dst, old_arena_base + i * padded_dim_, padded_dim_);

            const std::uint32_t doc_id = slot_to_doc_id_[i];
            new_slot_to_doc_id.push_back(doc_id);
            new_scales.push_back(active_scales_[i]);
            new_sums_128.push_back(active_sums_128_[i]);
            new_texts.push_back(std::move(doc_texts_[i]));
            new_metadata.push_back(std::move(doc_metadata_[i]));
            new_doc_id_to_slot[doc_id] = new_s;
        }

        inverted_index_.compact(old_to_new, new_slots);

        arena_ = std::move(new_arena);
        slot_to_doc_id_ = std::move(new_slot_to_doc_id);
        slot_deleted_ = std::move(new_slot_deleted);
        active_scales_ = std::move(new_scales);
        active_sums_128_ = std::move(new_sums_128);
        doc_texts_ = std::move(new_texts);
        doc_metadata_ = std::move(new_metadata);
        doc_id_to_slot_ = std::move(new_doc_id_to_slot);
        has_deletions_ = false;
    }

    [[nodiscard]] std::size_t total_docs() const noexcept { return active_docs_; }
    [[nodiscard]] std::size_t document_count() const noexcept { return active_docs_; }
    [[nodiscard]] std::size_t total_slots() const noexcept { return slot_to_doc_id_.size(); }
    [[nodiscard]] std::size_t deleted_docs() const noexcept {
        return slot_to_doc_id_.size() >= active_docs_ ? (slot_to_doc_id_.size() - active_docs_) : 0;
    }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t memory_used() const noexcept { return arena_.used_bytes(); }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return inverted_index_.vocab_size(); }

private:
    static constexpr std::size_t kWorkspacePoolSize = 32;

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
        std::vector<float> bm25_raw_score_buf;
        std::vector<float> dense_raw_score_buf;
        std::vector<std::uint32_t> bm25_rank_buf;
        std::vector<std::uint32_t> dense_rank_buf;
        std::vector<std::uint8_t> filter_mask_buf;

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
                bm25_raw_score_buf.resize(total_slots, 0.0f);
                dense_raw_score_buf.resize(total_slots, 0.0f);
                bm25_rank_buf.resize(total_slots, 0);
                dense_rank_buf.resize(total_slots, 0);
            }
        }
    };

    struct PoolSlot {
        std::atomic<bool> in_use{false};
        SearchWorkspace ws;
    };

    struct ScopedWorkspace {
        PoolSlot* slot{nullptr};
        std::unique_ptr<SearchWorkspace> fallback;
        SearchWorkspace& ws;

        explicit ScopedWorkspace(PoolSlot* pool) : ws(acquire(pool)) {}
        ~ScopedWorkspace() noexcept {
            if (slot) {
                slot->in_use.store(false, std::memory_order_release);
            }
        }

        ScopedWorkspace(const ScopedWorkspace&) = delete;
        ScopedWorkspace& operator=(const ScopedWorkspace&) = delete;

    private:
        SearchWorkspace& acquire(PoolSlot* pool) {
            for (std::size_t i = 0; i < kWorkspacePoolSize; ++i) {
                if (!pool[i].in_use.load(std::memory_order_relaxed) &&
                    !pool[i].in_use.exchange(true, std::memory_order_acquire)) {
                    slot = &pool[i];
                    return slot->ws;
                }
            }
            fallback = std::make_unique<SearchWorkspace>();
            return *fallback;
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
    std::vector<std::string> doc_metadata_;
    std::unordered_map<std::uint32_t, std::uint32_t> doc_id_to_slot_;
    std::size_t active_docs_{0};
    bool has_deletions_{false};
    mutable std::unique_ptr<PoolSlot[]> ws_pool_;
};

} // namespace vrag
