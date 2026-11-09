"""
rag_core: Lightweight In-Process C++20 Hybrid Retrieval Engine (BM25 + SQ8 Dense Vector + Weighted RRF)
Python SDK with zero-copy NumPy integration via stable C ABI.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


class SearchMode(IntEnum):
    HYBRID = 0
    BM25 = 1
    DENSE = 2

    @classmethod
    def from_any(cls, val: Union["SearchMode", str, int]) -> "SearchMode":
        if isinstance(val, cls):
            return val
        if isinstance(val, int):
            return cls(val)
        if isinstance(val, str):
            norm = val.strip().lower()
            if norm == "hybrid":
                return cls.HYBRID
            if norm in ("bm25", "sparse"):
                return cls.BM25
            if norm in ("dense", "vector", "sq8"):
                return cls.DENSE
        raise ValueError(f"Unsupported SearchMode: {val!r}")


@dataclass
class SearchResult:
    doc_id: int
    score: float
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    bm25_rank: int = 0
    dense_rank: int = 0
    metadata: str = ""


HitResult = SearchResult

FilterCallback = Callable[[int, str], bool]
RerankCallback = Callable[[str, List[SearchResult]], Optional[List[SearchResult]]]


@dataclass
class SearchOptions:
    top_k: int = 5
    mode: Union[SearchMode, str] = SearchMode.HYBRID
    rrf_k: float = 60.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    candidate_limit: int = 200
    min_score: float = -1e30
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    num_threads: int = 0
    filter: Optional[FilterCallback] = None
    reranker: Optional[RerankCallback] = None


@dataclass
class MemoryStats:
    vector_arena_bytes: int
    vector_meta_bytes: int
    inverted_index_bytes: int
    doc_store_bytes: int
    total_bytes: int


# --- Ctypes Struct & Function Definitions ---

class _CSearchOptions(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_size_t),
        ("mode", ctypes.c_uint8),
        ("rrf_k", ctypes.c_float),
        ("bm25_weight", ctypes.c_float),
        ("dense_weight", ctypes.c_float),
        ("candidate_limit", ctypes.c_size_t),
        ("min_score", ctypes.c_float),
        ("bm25_k1", ctypes.c_float),
        ("bm25_b", ctypes.c_float),
        ("num_threads", ctypes.c_int),
    ]


class _CHit(ctypes.Structure):
    _fields_ = [
        ("doc_id", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("bm25_score", ctypes.c_float),
        ("dense_score", ctypes.c_float),
        ("bm25_rank", ctypes.c_uint32),
        ("dense_rank", ctypes.c_uint32),
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("metadata", ctypes.c_char_p),
        ("metadata_len", ctypes.c_size_t),
    ]


class _CMemoryStats(ctypes.Structure):
    _fields_ = [
        ("vector_arena_bytes", ctypes.c_size_t),
        ("vector_meta_bytes", ctypes.c_size_t),
        ("inverted_index_bytes", ctypes.c_size_t),
        ("doc_store_bytes", ctypes.c_size_t),
        ("total_bytes", ctypes.c_size_t),
    ]


_C_FILTER_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


def _load_shared_library() -> ctypes.CDLL:
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent

    if sys.platform.startswith("win"):
        # Ensure MinGW / compiler runtime DLLs (e.g. libgomp-1.dll) are discoverable on Windows
        gpp_path = shutil.which("g++")
        if gpp_path:
            mingw_bin = str(Path(gpp_path).resolve().parent)
            if hasattr(os, "add_dll_directory") and os.path.isdir(mingw_bin):
                try:
                    os.add_dll_directory(mingw_bin)
                except OSError:
                    pass
        lib_names = ["rag_core_c.dll", "librag_core_c.dll"]
    elif sys.platform == "darwin":
        lib_names = ["librag_core_c.dylib", "rag_core_c.dylib"]
    else:
        lib_names = ["librag_core_c.so", "rag_core_c.so"]

    search_dirs = [
        pkg_dir,
        repo_root / "build",
        repo_root / "build" / "Release",
        repo_root / "build" / "Debug",
    ]

    for d in search_dirs:
        for name in lib_names:
            candidate = d / name
            if candidate.is_file():
                lib = ctypes.CDLL(str(candidate))
                _bind_signatures(lib)
                return lib

    raise RuntimeError(
        f"Could not find rag_core_c shared library ({lib_names}) in {search_dirs}. "
        "Please build the C++ project first with: cmake -B build && cmake --build build"
    )


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.vrag_default_search_options.argtypes = [ctypes.POINTER(_CSearchOptions)]
    lib.vrag_default_search_options.restype = None

    lib.vrag_engine_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    lib.vrag_engine_create.restype = ctypes.c_void_p

    lib.vrag_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.vrag_engine_destroy.restype = None

    lib.vrag_add_document.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.vrag_add_document.restype = ctypes.c_int

    lib.vrag_add_documents_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.vrag_add_documents_batch.restype = ctypes.c_int

    lib.vrag_remove_document.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.vrag_remove_document.restype = ctypes.c_int

    lib.vrag_contains_document.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.vrag_contains_document.restype = ctypes.c_int

    lib.vrag_compact.argtypes = [ctypes.c_void_p]
    lib.vrag_compact.restype = None

    lib.vrag_save_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vrag_save_index.restype = ctypes.c_int

    lib.vrag_load_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vrag_load_index.restype = ctypes.c_int

    lib.vrag_load_from_binary.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vrag_load_from_binary.restype = ctypes.c_int

    lib.vrag_document_count.argtypes = [ctypes.c_void_p]
    lib.vrag_document_count.restype = ctypes.c_size_t

    lib.vrag_total_slots.argtypes = [ctypes.c_void_p]
    lib.vrag_total_slots.restype = ctypes.c_size_t

    lib.vrag_deleted_docs.argtypes = [ctypes.c_void_p]
    lib.vrag_deleted_docs.restype = ctypes.c_size_t

    lib.vrag_dimension.argtypes = [ctypes.c_void_p]
    lib.vrag_dimension.restype = ctypes.c_size_t

    lib.vrag_vocab_size.argtypes = [ctypes.c_void_p]
    lib.vrag_vocab_size.restype = ctypes.c_size_t

    lib.vrag_memory_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CMemoryStats)]
    lib.vrag_memory_stats.restype = None

    lib.vrag_search.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(_CSearchOptions),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.vrag_search.restype = ctypes.c_void_p

    lib.vrag_result_count.argtypes = [ctypes.c_void_p]
    lib.vrag_result_count.restype = ctypes.c_size_t

    lib.vrag_result_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CHit)]
    lib.vrag_result_get.restype = ctypes.c_int

    lib.vrag_result_free.argtypes = [ctypes.c_void_p]
    lib.vrag_result_free.restype = None


_LIB = _load_shared_library()


def _as_float32_ptr(vec: Union[Sequence[float], "np.ndarray"]):
    """Returns (ctypes_float_ptr, length, keepalive_buffer) with zero-copy for C-contiguous float32 ndarray."""
    if np is not None:
        if isinstance(vec, np.ndarray) and vec.dtype == np.float32 and vec.flags["C_CONTIGUOUS"]:
            flat = vec.ravel()
            return flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), flat.size, flat
        arr = np.ascontiguousarray(vec, dtype=np.float32).ravel()
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size, arr
    n = len(vec)
    c_arr = (ctypes.c_float * n)(*vec)
    return ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_float)), n, c_arr


class Index:
    """
    In-process C++20 hybrid retrieval index (BM25 + SQ8 Dense Vector + Weighted RRF).
    Thread-safe for concurrent multi-reader queries and reader-writer synchronization.
    """

    def __init__(
        self,
        dimension: int = 768,
        arena_capacity_bytes: int = 64 * 1024 * 1024,
        *,
        dim: Optional[int] = None,
    ) -> None:
        if dim is not None:
            dimension = dim
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._handle = _LIB.vrag_engine_create(int(dimension), int(arena_capacity_bytes))
        if not self._handle:
            raise MemoryError("Failed to allocate native VragEngine instance")

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            _LIB.vrag_engine_destroy(handle)
            self._handle = None

    def close(self) -> None:
        if self._handle:
            _LIB.vrag_engine_destroy(self._handle)
            self._handle = None

    def _check_open(self) -> None:
        if not self._handle:
            raise RuntimeError("Index is closed")

    def add_document(
        self,
        doc_id: int,
        text: str,
        vector: Union[Sequence[float], "np.ndarray"],
        metadata: str = "",
    ) -> bool:
        """Inserts or updates (Upsert) a single document with UTF-8 text, FP32 vector, and optional metadata."""
        self._check_open()
        text_bytes = text.encode("utf-8")
        meta_bytes = metadata.encode("utf-8") if metadata else b""
        vec_ptr, vec_len, _keep = _as_float32_ptr(vector)
        ok = _LIB.vrag_add_document(
            self._handle,
            ctypes.c_uint32(doc_id),
            text_bytes,
            len(text_bytes),
            vec_ptr,
            vec_len,
            meta_bytes,
            len(meta_bytes),
        )
        return bool(ok)

    def add_documents(
        self,
        doc_ids: Sequence[int],
        texts: Sequence[str],
        vectors: Union[Sequence[Sequence[float]], "np.ndarray"],
        metadatas: Optional[Sequence[str]] = None,
    ) -> bool:
        """
        Batch-inserts documents using OpenMP parallel L2 normalization and SQ8 quantization.
        Zero-copy when `vectors` is a 2D C-contiguous numpy.ndarray(dtype=np.float32) of shape (N, D).
        """
        self._check_open()
        n = len(doc_ids)
        if n == 0:
            return True
        if len(texts) != n:
            raise ValueError(f"len(texts)={len(texts)} does not match len(doc_ids)={n}")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(f"len(metadatas)={len(metadatas)} does not match len(doc_ids)={n}")

        dim = self.dimension
        vec_ptr, total_floats, _keep_vec = _as_float32_ptr(vectors)
        if total_floats != n * dim:
            raise ValueError(f"vectors total float count ({total_floats}) != {n} * {dim}")

        if np is not None and isinstance(doc_ids, np.ndarray) and doc_ids.dtype == np.uint32 and doc_ids.flags["C_CONTIGUOUS"]:
            ids_ptr = doc_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            _keep_ids = doc_ids
        elif np is not None:
            ids_arr = np.ascontiguousarray(doc_ids, dtype=np.uint32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            _keep_ids = ids_arr
        else:
            ids_arr = (ctypes.c_uint32 * n)(*doc_ids)
            ids_ptr = ctypes.cast(ids_arr, ctypes.POINTER(ctypes.c_uint32))
            _keep_ids = ids_arr

        encoded_texts = [t.encode("utf-8") for t in texts]
        c_texts = (ctypes.c_char_p * n)(*encoded_texts)
        c_text_lens = (ctypes.c_size_t * n)(*(len(b) for b in encoded_texts))

        if metadatas is not None:
            encoded_metas = [m.encode("utf-8") for m in metadatas]
            c_metas = (ctypes.c_char_p * n)(*encoded_metas)
            c_meta_lens = (ctypes.c_size_t * n)(*(len(b) for b in encoded_metas))
            meta_ptr = ctypes.cast(c_metas, ctypes.POINTER(ctypes.c_char_p))
            meta_len_ptr = ctypes.cast(c_meta_lens, ctypes.POINTER(ctypes.c_size_t))
        else:
            encoded_metas = []
            c_metas = None
            c_meta_lens = None
            meta_ptr = None
            meta_len_ptr = None

        ok = _LIB.vrag_add_documents_batch(
            self._handle,
            n,
            ids_ptr,
            ctypes.cast(c_texts, ctypes.POINTER(ctypes.c_char_p)),
            ctypes.cast(c_text_lens, ctypes.POINTER(ctypes.c_size_t)),
            vec_ptr,
            dim,
            meta_ptr,
            meta_len_ptr,
        )
        return bool(ok)

    add_documents_batch = add_documents

    def remove_document(self, doc_id: int) -> bool:
        """Removes a document in O(1) via tombstone marking."""
        self._check_open()
        return bool(_LIB.vrag_remove_document(self._handle, ctypes.c_uint32(doc_id)))

    def contains_document(self, doc_id: int) -> bool:
        self._check_open()
        return bool(_LIB.vrag_contains_document(self._handle, ctypes.c_uint32(doc_id)))

    def __contains__(self, doc_id: int) -> bool:
        return self.contains_document(doc_id)

    def __len__(self) -> int:
        return self.document_count

    def compact(self) -> None:
        """Physically reclaims deleted tombstone slots and purges dead vocabulary terms."""
        self._check_open()
        _LIB.vrag_compact(self._handle)

    def save_index(self, filepath: Union[str, os.PathLike]) -> bool:
        """Atomically saves the pre-quantized index with 64-bit FNV-1a checksum (VRAGIDX3)."""
        self._check_open()
        path_bytes = os.fsdecode(filepath).encode("utf-8")
        return bool(_LIB.vrag_save_index(self._handle, path_bytes))

    def load_index(self, filepath: Union[str, os.PathLike]) -> bool:
        """Transactionally loads a pre-quantized index snapshot after verifying its 64-bit checksum."""
        self._check_open()
        path_bytes = os.fsdecode(filepath).encode("utf-8")
        return bool(_LIB.vrag_load_index(self._handle, path_bytes))

    def load_from_binary(self, filepath: Union[str, os.PathLike]) -> bool:
        self._check_open()
        path_bytes = os.fsdecode(filepath).encode("utf-8")
        return bool(_LIB.vrag_load_from_binary(self._handle, path_bytes))

    def search(
        self,
        query_text: str = "",
        query_vector: Optional[Union[Sequence[float], "np.ndarray"]] = None,
        top_k: int = 5,
        *,
        options: Optional[SearchOptions] = None,
        mode: Union[SearchMode, str] = SearchMode.HYBRID,
        rrf_k: float = 60.0,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        candidate_limit: int = 200,
        min_score: float = -1e30,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        num_threads: int = 0,
        filter: Optional[FilterCallback] = None,
        reranker: Optional[RerankCallback] = None,
    ) -> List[SearchResult]:
        """
        Executes Hybrid (BM25 + SQ8 + Weighted RRF), pure BM25, or pure Dense SQ8 search.
        """
        self._check_open()
        if options is not None:
            top_k = options.top_k
            mode = options.mode
            rrf_k = options.rrf_k
            bm25_weight = options.bm25_weight
            dense_weight = options.dense_weight
            candidate_limit = options.candidate_limit
            min_score = options.min_score
            bm25_k1 = options.bm25_k1
            bm25_b = options.bm25_b
            num_threads = options.num_threads
            filter = options.filter
            reranker = options.reranker

        resolved_mode = SearchMode.from_any(mode)
        effective_top_k = max(top_k, candidate_limit) if reranker is not None else top_k

        c_opts = _CSearchOptions(
            top_k=int(effective_top_k),
            mode=int(resolved_mode),
            rrf_k=float(rrf_k),
            bm25_weight=float(bm25_weight),
            dense_weight=float(dense_weight),
            candidate_limit=int(candidate_limit),
            min_score=float(min_score),
            bm25_k1=float(bm25_k1),
            bm25_b=float(bm25_b),
            num_threads=int(num_threads),
        )

        q_bytes = query_text.encode("utf-8") if query_text else b""
        if query_vector is not None and len(query_vector) > 0:
            vec_ptr, vec_len, _keep = _as_float32_ptr(query_vector)
        else:
            vec_ptr = ctypes.POINTER(ctypes.c_float)()
            vec_len = 0
            _keep = None

        c_filter_ptr = ctypes.c_void_p(0)
        _keep_cb = None
        if filter is not None:
            def _py_filter_trampoline(doc_id: int, meta_ptr, meta_len: int, _user_data) -> int:
                try:
                    raw = ctypes.string_at(meta_ptr, meta_len) if meta_len > 0 else b""
                    meta_str = raw.decode("utf-8", errors="replace")
                    return 1 if filter(int(doc_id), meta_str) else 0
                except Exception:
                    return 0

            _keep_cb = _C_FILTER_FN(_py_filter_trampoline)
            c_filter_ptr = ctypes.cast(_keep_cb, ctypes.c_void_p)

        rs_handle = _LIB.vrag_search(
            self._handle,
            q_bytes,
            len(q_bytes),
            vec_ptr,
            vec_len,
            ctypes.byref(c_opts),
            c_filter_ptr,
            None,
        )
        if not rs_handle:
            return []

        try:
            cnt = _LIB.vrag_result_count(rs_handle)
            results: List[SearchResult] = []
            c_hit = _CHit()
            for i in range(cnt):
                if _LIB.vrag_result_get(rs_handle, i, ctypes.byref(c_hit)):
                    text_str = ctypes.string_at(c_hit.text, c_hit.text_len).decode("utf-8", errors="replace") if c_hit.text_len > 0 else ""
                    meta_str = ctypes.string_at(c_hit.metadata, c_hit.metadata_len).decode("utf-8", errors="replace") if c_hit.metadata_len > 0 else ""
                    results.append(
                        SearchResult(
                            doc_id=int(c_hit.doc_id),
                            score=float(c_hit.score),
                            text=text_str,
                            bm25_score=float(c_hit.bm25_score),
                            dense_score=float(c_hit.dense_score),
                            bm25_rank=int(c_hit.bm25_rank),
                            dense_rank=int(c_hit.dense_rank),
                            metadata=meta_str,
                        )
                    )
        finally:
            _LIB.vrag_result_free(rs_handle)

        if reranker is not None and results:
            maybe_new = reranker(query_text, results)
            if maybe_new is not None:
                results = list(maybe_new)
            if len(results) > top_k:
                results = results[:top_k]

        return results

    def search_bm25(
        self,
        query_text: str,
        top_k: int = 10,
        filter: Optional[FilterCallback] = None,
    ) -> List[SearchResult]:
        """Executes pure BM25 lexical retrieval and returns raw BM25 scores."""
        return self.search(query_text=query_text, query_vector=None, top_k=top_k, mode=SearchMode.BM25, filter=filter)

    def search_dense(
        self,
        query_vector: Union[Sequence[float], "np.ndarray"],
        top_k: int = 10,
        filter: Optional[FilterCallback] = None,
    ) -> List[SearchResult]:
        """Executes pure SQ8 dense vector retrieval and returns raw cosine similarity scores."""
        return self.search(query_text="", query_vector=query_vector, top_k=top_k, mode=SearchMode.DENSE, filter=filter)

    @property
    def document_count(self) -> int:
        self._check_open()
        return int(_LIB.vrag_document_count(self._handle))

    @property
    def total_slots(self) -> int:
        self._check_open()
        return int(_LIB.vrag_total_slots(self._handle))

    @property
    def deleted_docs(self) -> int:
        self._check_open()
        return int(_LIB.vrag_deleted_docs(self._handle))

    @property
    def dimension(self) -> int:
        self._check_open()
        return int(_LIB.vrag_dimension(self._handle))

    @property
    def vocab_size(self) -> int:
        self._check_open()
        return int(_LIB.vrag_vocab_size(self._handle))

    def memory_stats(self) -> MemoryStats:
        self._check_open()
        st = _CMemoryStats()
        _LIB.vrag_memory_stats(self._handle, ctypes.byref(st))
        return MemoryStats(
            vector_arena_bytes=int(st.vector_arena_bytes),
            vector_meta_bytes=int(st.vector_meta_bytes),
            inverted_index_bytes=int(st.inverted_index_bytes),
            doc_store_bytes=int(st.doc_store_bytes),
            total_bytes=int(st.total_bytes),
        )


VragEngine = Index

__all__ = [
    "Index",
    "VragEngine",
    "SearchMode",
    "SearchOptions",
    "SearchResult",
    "HitResult",
    "MemoryStats",
]
