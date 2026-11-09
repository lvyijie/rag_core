#!/usr/bin/env python3
"""
rag_core — Local AI Agent Hybrid Knowledge Base Demo
====================================================

This example demonstrates how a Python AI Agent integrates `rag_core` as an
in-process C++20 hybrid retrieval engine (BM25 + SQ8 Dense Vector + Weighted RRF):

  1. Batch-ingesting agent documentation chunks + 2D NumPy float32 embeddings
     with zero-copy pointer passing into the C++ SQ8 arena.
  2. Running Hybrid Search (Weighted RRF), Pure BM25, and Pure Dense SQ8 queries.
  3. Filtering results by metadata (e.g. restricting to `module=persistence` or `tier=core`).
  4. Plugging in a custom Python reranker hook before assembling the final LLM context.
  5. Persisting the pre-quantized index (`VRAGIDX3` format with 64-bit FNV-1a checksum)
     and inspecting live memory breakdown.
"""

import os
import sys
import tempfile
from typing import List

import numpy as np

# Allow running directly from repository root without `pip install -e .`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from rag_core import Index, SearchMode, SearchOptions, SearchResult


def build_demo_embeddings(num_docs: int, dim: int = 384) -> np.ndarray:
    """
    Generates deterministic 384-D unit-normalized embeddings for each knowledge base entry.
    In production, replace with `sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5').encode(...)`.
    """
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((num_docs, dim)).astype(np.float32)
    # Create distinct semantic clusters along basis dimensions
    for i in range(num_docs):
        mat[i, i * 8 : (i + 1) * 8] += 6.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / norms).astype(np.float32)


def build_rag_prompt(user_query: str, contexts: List[SearchResult]) -> str:
    """Constructs a grounded RAG prompt for an LLM Agent from retrieved `SearchResult` items."""
    ctx_blocks = []
    for rank, hit in enumerate(contexts, start=1):
        ctx_blocks.append(
            f"[{rank}] (doc_id={hit.doc_id}, metadata={hit.metadata}, "
            f"rrf={hit.score:.4f}, bm25={hit.bm25_score:.2f}#{hit.bm25_rank}, "
            f"cosine={hit.dense_score:.4f}#{hit.dense_rank})\n"
            f"    {hit.text}"
        )
    joined_ctx = "\n\n".join(ctx_blocks)
    return (
        "You are an engineering assistant. Answer the user query strictly using the retrieved context.\n\n"
        f"=== RETRIEVED CONTEXT ===\n{joined_ctx}\n\n"
        f"=== USER QUERY ===\n{user_query}\n"
    )


def main() -> None:
    DIM = 384
    kb_docs = [
        (
            101,
            "rag_core quantizes 384-D FP32 vectors into INT8 (SQ8) inside a 64-byte aligned ArenaAllocator, "
            "reducing vector memory by 4.0x while preserving 97.3% Top-10 overlap on BEIR.",
            "module=quantization;tier=core",
        ),
        (
            102,
            "AVX2 and AVX-512 VNNI intrinsics accelerate int8 dot-product scoring with zero heap allocations "
            "along the hot scan path.",
            "module=simd;tier=core",
        ),
        (
            103,
            "The BM25 inverted index uses a single-pass bilingual UTF-8 tokenizer (ASCII words + CJK unigrams/bigrams) "
            "and maintains exact active document frequency (DF) across O(1) tombstone deletions.",
            "module=bm25;tier=retrieval",
        ),
        (
            104,
            "Weighted Reciprocal Rank Fusion (Weighted RRF) combines BM25 lexical ranks and SQ8 dense cosine ranks: "
            "score(d) = w_bm25 / (k + r_bm25) + w_dense / (k + r_dense).",
            "module=hybrid;tier=retrieval",
        ),
        (
            105,
            "VRAGIDX3 index persistence writes to a temporary file with a streaming 64-bit FNV-1a checksum and "
            "VRAGEND3 footer before atomically renaming over the target snapshot.",
            "module=persistence;tier=storage",
        ),
        (
            106,
            "Reader-Writer concurrency is guarded by std::shared_mutex: up to 16 concurrent reader threads "
            "achieve 66k-107k QPS on BEIR while writers acquire an exclusive lock.",
            "module=concurrency;tier=storage",
        ),
    ]

    doc_ids = [item[0] for item in kb_docs]
    texts = [item[1] for item in kb_docs]
    metadatas = [item[2] for item in kb_docs]
    embeddings = build_demo_embeddings(len(kb_docs), dim=DIM)

    # 1. Initialize index & batch-ingest via zero-copy 2D NumPy matrix
    index = Index(dimension=DIM)
    index.add_documents(doc_ids, texts, embeddings, metadatas)
    print(f"[1] Indexed {len(index)} documents (dim={index.dimension}, vocab_size={index.vocab_size})")

    stats = index.memory_stats()
    print(
        f"    Memory Breakdown: total={stats.total_bytes:,} B "
        f"(vector_arena={stats.vector_arena_bytes:,} B, "
        f"inverted_index={stats.inverted_index_bytes:,} B, "
        f"doc_store={stats.doc_store_bytes:,} B)"
    )

    # 2. Query 1: Weighted Hybrid Search (0.3 BM25 : 1.0 Dense, matching BEIR SciFact optimal config)
    query_text = "How does VRAGIDX3 atomic index persistence prevent corrupted snapshots?"
    # Simulate query embedding close to doc 105 (index 4)
    query_vec = embeddings[4].copy()

    opts = SearchOptions(
        top_k=3,
        mode=SearchMode.HYBRID,
        bm25_weight=0.3,
        dense_weight=1.0,
        rrf_k=60.0,
    )
    hits = index.search(query_text, query_vec, options=opts)

    print("\n[2] Weighted Hybrid Search Results (bm25_weight=0.3, dense_weight=1.0):")
    print(build_rag_prompt(query_text, hits))

    # 3. Query 2: Metadata-filtered search + Python reranker hook
    filtered_opts = SearchOptions(
        top_k=2,
        mode=SearchMode.HYBRID,
        filter=lambda doc_id, meta: "tier=core" in meta,
        reranker=lambda q, results: sorted(
            results,
            key=lambda r: r.bm25_score + r.dense_score,
            reverse=True,
        ),
    )
    core_hits = index.search("SQ8 quantization and AVX2 SIMD acceleration", embeddings[0], options=filtered_opts)
    print("[3] Filtered (`tier=core`) + Custom Reranked Hits:")
    for h in core_hits:
        print(f"    -> doc_id={h.doc_id} | meta={h.metadata} | bm25={h.bm25_score:.2f} | cosine={h.dense_score:.4f}")

    # 4. Atomic Persistence & Reload
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = os.path.join(tmpdir, "agent_memory.vidx")
        assert index.save_index(snap_path)
        reloaded = Index(dimension=DIM)
        assert reloaded.load_index(snap_path)
        print(f"\n[4] Snapshot saved & reloaded cleanly ({os.path.getsize(snap_path):,} bytes on disk, "
              f"{len(reloaded)} active docs).")


if __name__ == "__main__":
    main()
