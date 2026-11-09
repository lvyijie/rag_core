import csv
import http.client
import json
import os
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
API_HOST = "api.siliconflow.cn"
API_PATH = "/v1/embeddings"
API_KEY = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("EMBED_API_KEY")
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
DIMENSION = 1024

_tls = threading.local()

def _get_https_conn() -> http.client.HTTPSConnection:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(API_HOST, timeout=45, context=ctx)
        _tls.conn = conn
    return conn

def _reset_https_conn():
    conn = getattr(_tls, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _tls.conn = None

def get_embeddings_batch(texts: list[str], max_retries: int = 4) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    # Truncate overly long texts to avoid token limit errors
    safe_texts = [t[:2000] if len(t) > 2000 else (t if t else " ") for t in texts]
    body = json.dumps({"model": MODEL_NAME, "input": safe_texts}, ensure_ascii=False).encode("utf-8")

    for attempt in range(max_retries):
        try:
            conn = _get_https_conn()
            conn.request("POST", API_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                _reset_https_conn()
                raise RuntimeError(f"HTTP {resp.status}: {raw.decode('utf-8', errors='ignore')[:200]}")
            res_json = json.loads(raw)
            data = sorted(res_json["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except Exception as e:
            _reset_https_conn()
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding API failed: {e}")
            time.sleep(0.8 * (attempt + 1))

def embed_all(texts: list[str], batch_size: int = 64, workers: int = 12) -> list[list[float]]:
    batches = [(i, texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    results = [None] * len(batches)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get_embeddings_batch, b): idx for idx, (_, b) in enumerate(batches)}
        for fut in futures:
            idx = futures[fut]
            results[idx] = fut.result()
            done += len(batches[idx][1])
            elapsed = max(0.001, time.time() - t0)
            print(f"\r  Embedded {done}/{len(texts)} ({done / elapsed:.1f} items/s)", end="", flush=True)
    print()

    flat = []
    for r in results:
        flat.extend(r)
    return flat

def download_beir_dataset(dataset: str, data_root: Path) -> Path:
    ds_dir = data_root / dataset
    if (ds_dir / "corpus.jsonl").exists() and (ds_dir / "queries.jsonl").exists():
        return ds_dir

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{dataset}.zip"
    url = f"{BEIR_BASE_URL}/{dataset}.zip"
    print(f"[*] Downloading official BEIR dataset '{dataset}' from {url} ...")

    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(zip_path, "wb") as out_f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out_f.write(chunk)

    print(f"[*] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    zip_path.unlink(missing_ok=True)
    return ds_dir

def prepare_beir_bins(dataset: str, data_root: Path, mode: str) -> tuple[Path, Path]:
    ds_dir = download_beir_dataset(dataset, data_root)

    # 1. Load corpus.jsonl
    doc_str_to_int = {}
    corpus_texts = []
    with open(ds_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            str_id = str(obj["_id"])
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            full_text = f"{title}\n{text}".strip() if title else text
            int_id = len(corpus_texts)
            doc_str_to_int[str_id] = int_id
            corpus_texts.append(full_text)

    # 2. Load qrels/test.tsv (fallback to dev.tsv if test.tsv not present)
    qrels_path = ds_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        qrels_path = ds_dir / "qrels" / "dev.tsv"

    qrels_map: dict[str, dict[int, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid_str, cid_str, score_str = row[0], row[1], row[2]
            score = int(float(score_str))
            if score <= 0 or cid_str not in doc_str_to_int:
                continue
            qrels_map.setdefault(qid_str, {})[doc_str_to_int[cid_str]] = score

    # 3. Load queries.jsonl (keep only test queries that have ground-truth qrels)
    test_queries = []
    with open(ds_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid_str = str(obj["_id"])
            if qid_str in qrels_map:
                test_queries.append((len(test_queries), obj.get("text", "").strip(), qrels_map[qid_str]))

    corpus_bin = ds_dir / f"corpus_{mode}.bin"
    queries_bin = ds_dir / f"queries_{mode}.bin"

    if corpus_bin.exists() and queries_bin.exists():
        return corpus_bin, queries_bin

    if mode == "local-embed":
        from fastembed import TextEmbedding
        local_model_name = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        print(f"[*] Loading local ONNX embedding model '{local_model_name}' via FastEmbed...")
        model = TextEmbedding(model_name=local_model_name)
        doc_vecs = [v.tolist() for v in model.embed(corpus_texts, batch_size=64)]
        q_vecs = [v.tolist() for v in model.embed([q[1] for q in test_queries], batch_size=64)]
        dim = len(doc_vecs[0])
    elif mode == "hybrid":
        dim = DIMENSION
        print(f"[*] Embedding {len(corpus_texts)} corpus docs & {len(test_queries)} queries with {MODEL_NAME}...")
        doc_vecs = embed_all(corpus_texts)
        q_vecs = embed_all([q[1] for q in test_queries])
    elif mode == "hybrid-sim":
        dim = DIMENSION
        import math, random
        rng = random.Random(42)
        pool_size = 256
        vec_pool = []
        for _ in range(pool_size):
            v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vec_pool.append(struct.pack(f"<{dim}f", *[x / norm for x in v]))
    else:
        dim = 64
        zero_bytes = struct.pack(f"<{dim}f", *([0.0] * dim))

    print(f"[*] Packing {corpus_bin.name} ({len(corpus_texts)} docs, dim={dim})...")
    with open(corpus_bin, "wb", buffering=8 * 1024 * 1024) as f:
        f.write(struct.pack("<II", len(corpus_texts), dim))
        for doc_id, text in enumerate(corpus_texts):
            tb = text.encode("utf-8")
            f.write(struct.pack("<II", doc_id, len(tb)))
            f.write(tb)
            if mode in ("hybrid", "local-embed"):
                f.write(struct.pack(f"<{dim}f", *doc_vecs[doc_id]))
            elif mode == "hybrid-sim":
                f.write(vec_pool[doc_id & (pool_size - 1)])
            else:
                f.write(zero_bytes)

    print(f"[*] Packing {queries_bin.name} ({len(test_queries)} test queries)...")
    with open(queries_bin, "wb", buffering=4 * 1024 * 1024) as f:
        f.write(struct.pack("<II", len(test_queries), dim))
        for idx, (qid, qtext, qrel) in enumerate(test_queries):
            tb = qtext.encode("utf-8")
            f.write(struct.pack("<II", qid, len(tb)))
            f.write(tb)
            if mode in ("hybrid", "local-embed"):
                f.write(struct.pack(f"<{dim}f", *q_vecs[idx]))
            elif mode == "hybrid-sim":
                f.write(vec_pool[idx & (pool_size - 1)])
            else:
                f.write(zero_bytes)
            f.write(struct.pack("<I", len(qrel)))
            for cid, rel in qrel.items():
                f.write(struct.pack("<II", cid, rel))

    return corpus_bin, queries_bin

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if "--local-embed" in args:
        mode = "local-embed"
    elif "--hybrid-sim" in args:
        mode = "hybrid-sim"
    elif "--bm25" in args or not API_KEY:
        mode = "bm25"
    else:
        mode = "hybrid"

    datasets = [d for d in args if not d.startswith("--")]
    if not datasets:
        datasets = ["scifact", "nfcorpus", "arguana", "fiqa"]

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "beir_data"
    exe_name = "beir_bench.exe" if os.name == "nt" else "beir_bench"
    bench_exe = repo_root / "build" / exe_name

    if not bench_exe.exists():
        print(f"[-] Benchmark binary not found at {bench_exe}. Please build the project first.")
        sys.exit(1)

    local_model_name = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    mode_desc = {
        "bm25": "Pure BM25",
        "local-embed": f"Real Embeddings ({local_model_name})",
        "hybrid": f"Real Embeddings ({MODEL_NAME})",
        "hybrid-sim": "Synthetic 1024d Vectors (Pipeline Latency Test)",
    }[mode]

    for ds in datasets:
        corpus_bin, queries_bin = prepare_beir_bins(ds, data_root, mode)
        label = f"BEIR/{ds} [{mode_desc}]"
        subprocess.run([str(bench_exe), str(corpus_bin), str(queries_bin), label], check=True)

if __name__ == "__main__":
    main()
