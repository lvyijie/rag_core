#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
    #if defined(RAG_CORE_BUILD_DLL)
        #define VRAG_API __declspec(dllexport)
    #else
        #define VRAG_API __declspec(dllimport)
    #endif
#else
    #define VRAG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vrag_engine_s vrag_engine_t;
typedef struct vrag_result_set_s vrag_result_set_t;

typedef enum vrag_search_mode_e {
    VRAG_MODE_HYBRID = 0,
    VRAG_MODE_BM25   = 1,
    VRAG_MODE_DENSE  = 2
} vrag_search_mode_t;

typedef int (*vrag_filter_fn)(uint32_t doc_id, const char* metadata, size_t metadata_len, void* user_data);

typedef struct vrag_search_options_s {
    size_t top_k;
    uint8_t mode;
    float rrf_k;
    float bm25_weight;
    float dense_weight;
    size_t candidate_limit;
    float min_score;
    float bm25_k1;
    float bm25_b;
    int num_threads;
} vrag_search_options_t;

typedef struct vrag_hit_s {
    uint32_t doc_id;
    float score;
    float bm25_score;
    float dense_score;
    uint32_t bm25_rank;
    uint32_t dense_rank;
    const char* text;
    size_t text_len;
    const char* metadata;
    size_t metadata_len;
} vrag_hit_t;

typedef struct vrag_memory_stats_s {
    size_t vector_arena_bytes;
    size_t vector_meta_bytes;
    size_t inverted_index_bytes;
    size_t doc_store_bytes;
    size_t total_bytes;
} vrag_memory_stats_t;

VRAG_API void vrag_default_search_options(vrag_search_options_t* out_opts);

VRAG_API vrag_engine_t* vrag_engine_create(size_t dimension, size_t arena_capacity_bytes);
VRAG_API void vrag_engine_destroy(vrag_engine_t* engine);

VRAG_API int vrag_add_document(
    vrag_engine_t* engine,
    uint32_t doc_id,
    const char* text,
    size_t text_len,
    const float* vector,
    size_t vector_dim,
    const char* metadata,
    size_t metadata_len
);

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
);

VRAG_API int vrag_remove_document(vrag_engine_t* engine, uint32_t doc_id);
VRAG_API int vrag_contains_document(const vrag_engine_t* engine, uint32_t doc_id);
VRAG_API void vrag_compact(vrag_engine_t* engine);

VRAG_API int vrag_save_index(const vrag_engine_t* engine, const char* filepath);
VRAG_API int vrag_load_index(vrag_engine_t* engine, const char* filepath);
VRAG_API int vrag_load_from_binary(vrag_engine_t* engine, const char* filepath);

VRAG_API size_t vrag_document_count(const vrag_engine_t* engine);
VRAG_API size_t vrag_total_slots(const vrag_engine_t* engine);
VRAG_API size_t vrag_deleted_docs(const vrag_engine_t* engine);
VRAG_API size_t vrag_dimension(const vrag_engine_t* engine);
VRAG_API size_t vrag_vocab_size(const vrag_engine_t* engine);
VRAG_API void vrag_memory_stats(const vrag_engine_t* engine, vrag_memory_stats_t* out_stats);

VRAG_API vrag_result_set_t* vrag_search(
    const vrag_engine_t* engine,
    const char* query_text,
    size_t query_len,
    const float* query_vector,
    size_t vector_dim,
    const vrag_search_options_t* options,
    vrag_filter_fn filter_cb,
    void* filter_user_data
);

VRAG_API size_t vrag_result_count(const vrag_result_set_t* results);
VRAG_API int vrag_result_get(const vrag_result_set_t* results, size_t index, vrag_hit_t* out_hit);
VRAG_API void vrag_result_free(vrag_result_set_t* results);

#ifdef __cplusplus
}
#endif
