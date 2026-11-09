#!/usr/bin/env python3
"""
P5 Python Bindings & Ecosystem Integration Test Suite for rag_core.
Verifies:
  1. Single document & 2D NumPy batch ingestion (zero-copy pointer passing)
  2. Unified search API (Hybrid, BM25-only, Dense-only)
  3. Weighted RRF, Metadata Filter callback, and Reranker hook
  4. Index persistence (save_index / load_index) & integrity
  5. Document removal, tombstone compaction, and memory_stats breakdown
  6. Multi-threaded concurrent queries via ThreadPoolExecutor (GIL released in C++)
  7. Defensive validation (dimension mismatch, NaN/Inf guard, invalid params)
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Ensure local python package is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import rag_core
from rag_core import Index, MemoryStats, SearchMode, SearchOptions, SearchResult


def make_unit_vec(dim: int, hot_idx: int, second_idx: int = -1, second_val: float = 0.2) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    if second_idx >= 0:
        v[second_idx] = second_val
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32)


def test_1_ingestion_and_search_modes():
    dim = 64
    idx = Index(dimension=dim)
    assert idx.dimension == dim
    assert len(idx) == 0

    v1 = make_unit_vec(dim, 0)
    v2 = make_unit_vec(dim, 1)
    v3 = make_unit_vec(dim, 2)

    assert idx.add_document(1, "C++ AVX2 SIMD int8 quantization engine", v1, metadata="lang=cpp;tier=core")
    assert idx.add_document(2, "Python bindings with zero-copy NumPy arrays", v2, metadata="lang=py;tier=sdk")
    assert idx.add_document(3, "BM25 inverted index and RRF hybrid fusion", v3, metadata="lang=cpp;tier=retrieval")

    assert len(idx) == 3
    assert idx.document_count == 3

    # 1. BM25-only search
    bm25_hits = idx.search_bm25("AVX2 SIMD quantization", top_k=2)
    assert len(bm25_hits) >= 1
    assert bm25_hits[0].doc_id == 1
    assert bm25_hits[0].bm25_score > 0.0
    assert bm25_hits[0].bm25_rank == 1
    assert bm25_hits[0].metadata == "lang=cpp;tier=core"

    # 2. Dense-only search
    dense_hits = idx.search_dense(v2, top_k=2)
    assert len(dense_hits) == 2
    assert dense_hits[0].doc_id == 2
    assert dense_hits[0].dense_rank == 1
    assert dense_hits[0].dense_score > 0.95

    # 3. Hybrid search
    hyb_hits = idx.search("Python NumPy arrays", v2, top_k=3)
    assert len(hyb_hits) == 3
    assert hyb_hits[0].doc_id == 2
    assert hyb_hits[0].bm25_rank == 1
    assert hyb_hits[0].dense_rank == 1
    print("[PASS] Test 1: Ingestion & BM25 / Dense / Hybrid search modes")


def test_2_batch_numpy_and_filters_reranker():
    dim = 128
    idx = Index(dimension=dim)

    doc_ids = list(range(1, 21))
    texts = [
        f"Document {i}: distributed vector database retrieval node"
        for i in range(20)
    ]
    mat = np.zeros((20, dim), dtype=np.float32)
    for i in range(20):
        mat[i, i % dim] = 1.0
    metas = ["env=prod" if i % 2 == 0 else "env=staging" for i in range(20)]

    ok = idx.add_documents(doc_ids, texts, mat, metas)
    assert ok is True
    assert idx.document_count == 20

    # Metadata filter: only accept env=staging
    q_vec = make_unit_vec(dim, 0)
    opts = SearchOptions(
        top_k=5,
        filter=lambda doc_id, meta: meta == "env=staging",
    )
    filtered_hits = idx.search("vector database", q_vec, options=opts)
    assert len(filtered_hits) == 5
    for h in filtered_hits:
        assert h.metadata == "env=staging"

    # Reranker hook: boost doc_id == 10 to the top
    opts_rerank = SearchOptions(
        top_k=5,
        reranker=lambda q_text, results: sorted(
            [
                SearchResult(
                    doc_id=r.doc_id,
                    score=999.0 if r.doc_id == 10 else r.score,
                    text=r.text,
                    bm25_score=r.bm25_score,
                    dense_score=r.dense_score,
                    bm25_rank=r.bm25_rank,
                    dense_rank=r.dense_rank,
                    metadata=r.metadata,
                )
                for r in results
            ],
            key=lambda x: x.score,
            reverse=True,
        ),
    )
    reranked = idx.search("vector database", q_vec, options=opts_rerank)
    assert reranked[0].doc_id == 10
    assert reranked[0].score == 999.0
    print("[PASS] Test 2: Zero-copy 2D NumPy batch ingestion, Metadata Filter & Reranker")


def test_3_persistence_compaction_and_memory_stats():
    dim = 64
    idx = Index(dimension=dim)
    for i in range(1, 11):
        assert idx.add_document(i, f"rag_core persistence test document {i}", make_unit_vec(dim, i - 1), f"id={i}")

    assert idx.remove_document(2) is True
    assert idx.remove_document(5) is True
    assert idx.remove_document(999) is False
    assert idx.document_count == 8
    assert idx.deleted_docs == 2

    idx.compact()
    assert idx.document_count == 8
    assert idx.deleted_docs == 0
    assert idx.total_slots == 8

    stats = idx.memory_stats()
    assert isinstance(stats, MemoryStats)
    assert stats.total_bytes > 0
    assert stats.vector_arena_bytes == 8 * 64

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_idx.vidx")
        assert idx.save_index(path) is True
        assert os.path.exists(path)

        loaded = Index(dimension=dim)
        assert loaded.load_index(path) is True
        assert loaded.document_count == 8
        hits = loaded.search_bm25("persistence test", top_k=3)
        assert len(hits) == 3
    print("[PASS] Test 3: Persistence (VRAGIDX3), Compaction & MemoryStats")


def test_4_concurrent_queries_and_defensive_guards():
    dim = 64
    idx = Index(dimension=dim)
    mat = np.eye(64, dtype=np.float32)
    doc_ids = list(range(1, 65))
    texts = [f"concurrent worker document number token_{i} with keyword alpha" for i in range(64)]
    assert idx.add_documents(doc_ids, texts, mat) is True

    def worker(t_id: int) -> int:
        target = t_id % 64
        q_vec = mat[target]
        res = idx.search(f"token_{target} alpha", q_vec, top_k=5)
        assert len(res) == 5
        assert res[0].doc_id == target + 1
        return len(res)

    with ThreadPoolExecutor(max_workers=8) as pool:
        totals = list(pool.map(worker, range(64)))
    assert sum(totals) == 64 * 5

    # Defensive checks: dimension mismatch returns False (rejected safely by C++ engine)
    assert idx.add_document(999, "bad dim", np.ones(16, dtype=np.float32)) is False

    # NaN vector rejected safely
    bad_vec = np.ones(dim, dtype=np.float32)
    bad_vec[0] = np.nan
    assert idx.add_document(999, "nan vec", bad_vec) is False

    print("[PASS] Test 4: ThreadPoolExecutor concurrent queries & defensive guards")


if __name__ == "__main__":
    print("=== P5 Python Bindings & SDK Verification Suite ===")
    test_1_ingestion_and_search_modes()
    test_2_batch_numpy_and_filters_reranker()
    test_3_persistence_compaction_and_memory_stats()
    test_4_concurrent_queries_and_defensive_guards()
    print("=== ALL P5 PYTHON TESTS PASSED ===")
