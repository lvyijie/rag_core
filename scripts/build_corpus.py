import os
import sys
import time
import struct
from pathlib import Path

# 配置硅基流动参数 (BGE-M3 1024 维)
API_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = os.getenv("EMBED_API_KEY", "")
MODEL_NAME = "BAAI/bge-m3"
DIMENSION = 1024
BATCH_SIZE = 16  # 单次请求并发数，防止接口超时

def chunk_text(text: str, max_chunk_size: int = 300, overlap: int = 40) -> list[str]:
    """滑动窗口切片：针对大文本进行重叠切块，避免语义割裂"""
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    # 优先按双换行切大段
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            if len(para) >= 20:  # 过滤无意义的超短碎片
                chunks.append(para)
        else:
            # 单段过长时走滑动窗口切分
            start = 0
            while start < len(para):
                end = start + max_chunk_size
                chunk = para[start:end].strip()
                if len(chunk) >= 20:
                    chunks.append(chunk)
                if end >= len(para):
                    break
                start += (max_chunk_size - overlap)
                
    return chunks

import json
import urllib.request

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """使用标准库 urllib 发起批量请求，零第三方依赖"""
    if not API_KEY:
        raise ValueError("请先设置环境变量 EMBED_API_KEY，例如: export EMBED_API_KEY='sk-...'")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "model": MODEL_NAME,
        "input": texts
    }

    req = urllib.request.Request(
        API_URL, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                data = res_json["data"]
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"请求 API 失败: {e}")
            time.sleep(2)

def main():
    if len(sys.argv) < 3:
        print(f"用法: python {sys.argv[0]} <文件或文件夹路径> <输出文件.bin>")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    output_bin = sys.argv[2]

    # 1. 扫描所有文本文件
    all_chunks = []
    files = [target_path] if target_path.is_file() else list(target_path.rglob("*"))
    valid_files = [f for f in files if f.is_file() and f.suffix in [".txt", ".md", ".cpp", ".hpp", ".c", ".h"]]

    print(f"[*] 扫描到 {len(valid_files)} 个文本/代码文件，开始切片...")
    for f in valid_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
                all_chunks.extend(chunk_text(content))
        except Exception as e:
            print(f"[!] 读取文件失败 {f}: {e}")

    total_chunks = len(all_chunks)
    print(f"[+] 文本解析完成，共生成 {total_chunks} 个切片。")
    if total_chunks == 0:
        print("[-] 未提取到有效切片，退出。")
        sys.exit(1)

    # 2. 批量调用 API 生成 1024 维向量
    print(f"[*] 开始请求远端 Embedding 模型: {MODEL_NAME} (批大小: {BATCH_SIZE})...")
    records = []
    start_time = time.time()

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_texts = all_chunks[i : i + BATCH_SIZE]
        print(f"    处理进度: [{i + len(batch_texts)}/{total_chunks}] ({(i + len(batch_texts)) / total_chunks * 100:.1f}%)", end="\r")
        
        vectors = get_embeddings_batch(batch_texts)
        for j, (text, vec) in enumerate(zip(batch_texts, vectors)):
            doc_id = i + j
            records.append((doc_id, text, vec))

    elapsed = time.time() - start_time
    print(f"\n[+] 全部向量化完成！耗时: {elapsed:.2f}s，平均每秒处理 {total_chunks / elapsed:.1f} 篇切片。")

    # 3. 序列化导出为二进制文件
    print(f"[*] 正在打包写入二进制格式: {output_bin} ...")
    with open(output_bin, "wb") as f:
        # Header: total_docs (uint32), dimension (uint32)
        f.write(struct.pack("<II", len(records), DIMENSION))
        
        for doc_id, text, vec in records:
            text_bytes = text.encode("utf-8")
            # Entry: doc_id (uint32), text_length (uint32), text_bytes, 1024 * float32
            f.write(struct.pack("<II", doc_id, len(text_bytes)))
            f.write(text_bytes)
            f.write(struct.pack(f"<{DIMENSION}f", *vec))

    file_size_mb = Path(output_bin).stat().st_size / (1024 * 1024)
    print(f"[✔] 写入成功！生成文件大小: {file_size_mb:.2f} MB。")

if __name__ == "__main__":
    main()
