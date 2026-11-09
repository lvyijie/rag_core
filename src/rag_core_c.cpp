#define RAG_CORE_BUILD_DLL
#include "../include/rag_core_c.h"
#include "../include/vrag.hpp"

#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct vrag_engine_s {
    vrag::VragEngine impl;

    vrag_engine_s(std::size_t dim, std::size_t cap) : impl(dim, cap) {}
};

struct vrag_result_set_s {
    std::vector<vrag::HitResult> hits;
};

extern "C" {

VRAG_API void vrag_default_search_options(vrag_search_options_t* out_opts) {
    if (!out_opts) return;
    out_opts->top_k = 5;
    out_opts->mode = VRAG_MODE_HYBRID;
    out_opts->rrf_k = 60.0f;
    out_opts->bm25_weight = 1.0f;
    out_opts->dense_weight = 1.0f;
    out_opts->candidate_limit = 200;
    out_opts->min_score = -1e30f;
    out_opts->bm25_k1 = 1.2f;
    out_opts->bm25_b = 0.75f;
    out_opts->num_threads = 0;
}

VRAG_API vrag_engine_t* vrag_engine_create(size_t dimension, size_t arena_capacity_bytes) {
    if (dimension == 0) return nullptr;
    try {
        const std::size_t cap = (arena_capacity_bytes > 0) ? arena_capacity_bytes : (64ULL * 1024 * 1024);
        return new vrag_engine_s(dimension, cap);
    } catch (...) {
        return nullptr;
    }
}

VRAG_API void vrag_engine_destroy(vrag_engine_t* engine) {
    delete engine;
}

VRAG_API int vrag_add_document(
    vrag_engine_t* engine,
    uint32_t doc_id,
    const char* text,
    size_t text_len,
    const float* vector,
    size_t vector_dim,
    const char* metadata,
    size_t metadata_len
) {
    if (!engine || !vector) return 0;
    try {
        std::string_view tv = (text && text_len > 0) ? std::string_view(text, text_len) : std::string_view{};
        std::string_view mv = (metadata && metadata_len > 0) ? std::string_view(metadata, metadata_len) : std::string_view{};
        return engine->impl.add_document(
            doc_id, tv, std::span<const float>(vector, vector_dim), mv
        ) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API int vrag_add_documents_batch(
    vrag_engine_t* engine,
    size_t count,
    const uint32_t* doc_ids,
    const char* const* texts,
    const size_t* text_lens,
    const float* vectors_flat,
    size_t vector_dim,
    const char* const* metadatas,
    const size_t* metadata_lens
) {
    if (!engine) return 0;
    if (count == 0) return 1;
    if (!doc_ids || !texts || !text_lens || !vectors_flat) return 0;
    try {
        std::vector<std::string_view> text_views(count);
        for (size_t i = 0; i < count; ++i) {
            text_views[i] = (texts[i] && text_lens[i] > 0)
                ? std::string_view(texts[i], text_lens[i])
                : std::string_view{};
        }
        std::vector<std::string_view> meta_views;
        if (metadatas && metadata_lens) {
            meta_views.resize(count);
            for (size_t i = 0; i < count; ++i) {
                meta_views[i] = (metadatas[i] && metadata_lens[i] > 0)
                    ? std::string_view(metadatas[i], metadata_lens[i])
                    : std::string_view{};
            }
        }
        return engine->impl.add_documents_batch(
            std::span<const uint32_t>(doc_ids, count),
            text_views,
            std::span<const float>(vectors_flat, count * vector_dim),
            meta_views
        ) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API int vrag_remove_document(vrag_engine_t* engine, uint32_t doc_id) {
    if (!engine) return 0;
    try {
        return engine->impl.remove_document(doc_id) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API int vrag_contains_document(const vrag_engine_t* engine, uint32_t doc_id) {
    if (!engine) return 0;
    return engine->impl.contains_document(doc_id) ? 1 : 0;
}

VRAG_API void vrag_compact(vrag_engine_t* engine) {
    if (!engine) return;
    try {
        engine->impl.compact();
    } catch (...) {}
}

VRAG_API int vrag_save_index(const vrag_engine_t* engine, const char* filepath) {
    if (!engine || !filepath) return 0;
    try {
        return engine->impl.save_index(filepath) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API int vrag_load_index(vrag_engine_t* engine, const char* filepath) {
    if (!engine || !filepath) return 0;
    try {
        return engine->impl.load_index(filepath) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API int vrag_load_from_binary(vrag_engine_t* engine, const char* filepath) {
    if (!engine || !filepath) return 0;
    try {
        return engine->impl.load_from_binary(filepath) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

VRAG_API size_t vrag_document_count(const vrag_engine_t* engine) {
    return engine ? engine->impl.document_count() : 0;
}

VRAG_API size_t vrag_total_slots(const vrag_engine_t* engine) {
    return engine ? engine->impl.total_slots() : 0;
}

VRAG_API size_t vrag_deleted_docs(const vrag_engine_t* engine) {
    return engine ? engine->impl.deleted_docs() : 0;
}

VRAG_API size_t vrag_dimension(const vrag_engine_t* engine) {
    return engine ? engine->impl.dimension() : 0;
}

VRAG_API size_t vrag_vocab_size(const vrag_engine_t* engine) {
    return engine ? engine->impl.vocab_size() : 0;
}

VRAG_API void vrag_memory_stats(const vrag_engine_t* engine, vrag_memory_stats_t* out_stats) {
    if (!engine || !out_stats) return;
    const auto st = engine->impl.memory_stats();
    out_stats->vector_arena_bytes = st.vector_arena_bytes;
    out_stats->vector_meta_bytes = st.vector_meta_bytes;
    out_stats->inverted_index_bytes = st.inverted_index_bytes;
    out_stats->doc_store_bytes = st.doc_store_bytes;
    out_stats->total_bytes = st.total_bytes();
}

VRAG_API vrag_result_set_t* vrag_search(
    const vrag_engine_t* engine,
    const char* query_text,
    size_t query_len,
    const float* query_vector,
    size_t vector_dim,
    const vrag_search_options_t* options,
    vrag_filter_fn filter_cb,
    void* filter_user_data
) {
    if (!engine) return nullptr;
    try {
        vrag::SearchOptions cpp_opts;
        if (options) {
            cpp_opts.top_k = options->top_k;
            cpp_opts.mode = static_cast<vrag::SearchMode>(options->mode);
            cpp_opts.rrf_k = options->rrf_k;
            cpp_opts.bm25_weight = options->bm25_weight;
            cpp_opts.dense_weight = options->dense_weight;
            cpp_opts.candidate_limit = options->candidate_limit;
            cpp_opts.min_score = options->min_score;
            cpp_opts.bm25_k1 = options->bm25_k1;
            cpp_opts.bm25_b = options->bm25_b;
            cpp_opts.num_threads = options->num_threads;
        }
        if (filter_cb) {
            cpp_opts.filter = [filter_cb, filter_user_data](std::uint32_t doc_id, std::string_view meta) -> bool {
                return filter_cb(doc_id, meta.data(), meta.size(), filter_user_data) != 0;
            };
        }

        std::string_view qt = (query_text && query_len > 0) ? std::string_view(query_text, query_len) : std::string_view{};
        std::span<const float> qv = (query_vector && vector_dim > 0) ? std::span<const float>(query_vector, vector_dim) : std::span<const float>{};

        auto* rs = new vrag_result_set_s();
        rs->hits = engine->impl.search(qt, qv, cpp_opts);
        return rs;
    } catch (...) {
        return nullptr;
    }
}

VRAG_API size_t vrag_result_count(const vrag_result_set_t* results) {
    return results ? results->hits.size() : 0;
}

VRAG_API int vrag_result_get(const vrag_result_set_t* results, size_t index, vrag_hit_t* out_hit) {
    if (!results || !out_hit || index >= results->hits.size()) return 0;
    const auto& h = results->hits[index];
    out_hit->doc_id = h.doc_id;
    out_hit->score = h.score;
    out_hit->bm25_score = h.bm25_score;
    out_hit->dense_score = h.dense_score;
    out_hit->bm25_rank = h.bm25_rank;
    out_hit->dense_rank = h.dense_rank;
    out_hit->text = h.text.c_str();
    out_hit->text_len = h.text.size();
    out_hit->metadata = h.metadata.c_str();
    out_hit->metadata_len = h.metadata.size();
    return 1;
}

VRAG_API void vrag_result_free(vrag_result_set_t* results) {
    delete results;
}

} // extern "C"
