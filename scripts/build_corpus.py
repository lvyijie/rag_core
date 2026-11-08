import os
import sys
import time
import struct
import json
import threading
import http.client
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 配置硅基流动参数 (BGE-M3 1024 维)
API_HOST = "api.siliconflow.cn"
API_PATH = "/v1/embeddings"
API_KEY = os.getenv("EMBED_API_KEY", "")
MODEL_NAME = "BAAI/bge-m3"
DIMENSION = 1024
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))   # 提升单次批大小至 64，减少一半请求次数
MAX_WORKERS = int(os.getenv("EMBED_WORKERS", "12"))     # 默认 12 线程并发

_thread_local = threading.local()

def _get_https_conn() -> http.client.HTTPSConnection:
    """线程局部复用的 HTTPS Keep-Alive 长连接，免除每次请求的 TCP+TLS 握手耗时"""
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = http.client.HTTPSConnection(API_HOST, timeout=60)
        _thread_local.conn = conn
    return conn

def _reset_https_conn():
    conn = getattr(_thread_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _thread_local.conn = None

def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """针对长篇小说/文档的高速段落聚合切片（默认 500 字一块，减少冗余碎片请求）"""
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = max(1, max_chunk_size - overlap)
    # 兼容单换行与双换行分段，优先将短自然段合并至接近 max_chunk_size
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    buf = ""

    for line in lines:
        if not buf:
            buf = line
        elif len(buf) + 1 + len(line) <= max_chunk_size:
            buf += "\n" + line
        else:
            if len(buf) >= 20:
                chunks.append(buf)
            if len(line) <= max_chunk_size:
                buf = line
            else:
                # 超长自然段走滑动窗口
                start = 0
                n = len(line)
                while start < n:
                    end = start + max_chunk_size
                    piece = line[start:end].strip()
                    if len(piece) >= 20:
                        chunks.append(piece)
                    if end >= n:
                        break
                    start += step
                buf = ""

    if len(buf) >= 20:
        chunks.append(buf)

    return chunks

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """使用标准库 http.client Keep-Alive 长连接发起批量请求，零第三方依赖"""
    if not API_KEY:
        raise ValueError("请先设置环境变量 EMBED_API_KEY，例如: $env:EMBED_API_KEY='sk-...'")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "User-Agent": "VRAG-Core/2.0"
    }
    body = json.dumps({"model": MODEL_NAME, "input": texts}, ensure_ascii=False).encode("utf-8")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = _get_https_conn()
            conn.request("POST", API_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {raw.decode('utf-8', errors='ignore')[:200]}")
            res_json = json.loads(raw.decode("utf-8"))
            data = res_json["data"]
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except Exception as e:
            _reset_https_conn()
            if attempt == max_retries - 1:
                raise RuntimeError(f"请求 API 失败: {e}")
            time.sleep(0.8 * (attempt + 1))

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 3:
        print(f"用法: python {sys.argv[0]} <文件或文件夹路径> <输出文件.bin> [最大切片数(可选)] [单块字数(默认500)]")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    output_bin = sys.argv[2]
    max_chunks = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    chunk_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 500
    overlap = min(50, max(10, chunk_size // 10))

    # 1. 扫描所有文本文件并极速切片
    t0 = time.time()
    all_chunks = []
    files = [target_path] if target_path.is_file() else sorted(target_path.rglob("*"))
    valid_files = [f for f in files if f.is_file() and f.suffix.lower() in [".txt", ".md", ".cpp", ".hpp", ".c", ".h"]]

    print(f"[*] 扫描到 {len(valid_files)} 个文件，开始按 {chunk_size} 字/块聚合切片...")
    for f in valid_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
                all_chunks.extend(chunk_text(content, max_chunk_size=chunk_size, overlap=overlap))
            if max_chunks > 0 and len(all_chunks) >= max_chunks:
                all_chunks = all_chunks[:max_chunks]
                break
        except Exception as e:
            print(f"[!] 读取文件失败 {f}: {e}")

    total_chunks = len(all_chunks)
    print(f"[+] 本地文本切片完成，耗时 {time.time() - t0:.2f}s，共生成 {total_chunks} 个切片。")
    if total_chunks == 0:
        print("[-] 未提取到有效切片，退出。")
        sys.exit(1)

    # 2. 长连接并发批量调用 API 生成 1024 维向量
    batches = [(i, all_chunks[i : i + BATCH_SIZE]) for i in range(0, total_chunks, BATCH_SIZE)]
    print(
        f"[*] 开始请求远端 Embedding 模型: {MODEL_NAME} "
        f"(共 {len(batches)} 批, 批大小: {BATCH_SIZE}, Keep-Alive 并发线程: {MAX_WORKERS})..."
    )
    start_time = time.time()

    batch_results = {}
    completed_chunks = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_offset = {
            executor.submit(get_embeddings_batch, b_texts): (offset, b_texts)
            for offset, b_texts in batches
        }
        for future in as_completed(future_to_offset):
            offset, b_texts = future_to_offset[future]
            vectors = future.result()
            batch_results[offset] = (b_texts, vectors)
            completed_chunks += len(b_texts)
            elapsed_now = max(time.time() - start_time, 1e-6)
            speed = completed_chunks / elapsed_now
            eta = (total_chunks - completed_chunks) / max(speed, 1e-6)
            print(
                f"    进度: [{completed_chunks}/{total_chunks}] "
                f"({completed_chunks / total_chunks * 100:.1f}%) | "
                f"速度: {speed:.1f} 篇/s | 预计剩余: {eta:.1f}s   ",
                end="\r"
            )

    elapsed = time.time() - start_time
    print(f"\n[+] 全部向量化完成！耗时: {elapsed:.2f}s，平均每秒处理 {total_chunks / max(elapsed, 1e-6):.1f} 篇切片。")

    # 3. 按原始顺序序列化导出为二进制文件
    print(f"[*] 正在打包写入二进制格式: {output_bin} ...")
    fmt = f"<{DIMENSION}f"
    with open(output_bin, "wb") as f:
        f.write(struct.pack("<II", total_chunks, DIMENSION))
        for offset in sorted(batch_results.keys()):
            b_texts, vectors = batch_results[offset]
            for j, (text, vec) in enumerate(zip(b_texts, vectors)):
                doc_id = offset + j
                text_bytes = text.encode("utf-8")
                f.write(struct.pack("<II", doc_id, len(text_bytes)))
                f.write(text_bytes)
                f.write(struct.pack(fmt, *vec))

    file_size_mb = Path(output_bin).stat().st_size / (1024 * 1024)
    print(f"[+] 写入成功！生成文件大小: {file_size_mb:.2f} MB。")

if __name__ == "__main__":
    main()
