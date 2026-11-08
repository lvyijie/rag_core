import sys
import struct
import random
import math
from pathlib import Path

def generate_normalized_vector(dim: int) -> list[float]:
    """生成一个服从高斯分布并归一化的单位向量（纯标准库，无需 numpy）"""
    # Box-Muller 变换生成标准正态分布随机数
    vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def perturb_vector(base_vec: list[float], noise_scale: float = 0.15) -> list[float]:
    """对基准向量添加扰动并重新归一化，模拟真实流形空间分布"""
    vec = [b + random.gauss(0.0, noise_scale) for b in base_vec]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def main():
    if len(sys.argv) < 4:
        print(f"用法: python {sys.argv[0]} <源文本文件> <目标切片总数> <输出.bin>")
        print("例如生成 10 万切片: python scripts/generate_benchmark_corpus.py hongloumeng.txt 100000 bench_30m.bin")
        sys.exit(1)

    src_file = sys.argv[1]
    target_chunks = int(sys.argv[2])
    output_bin = sys.argv[3]
    dimension = 1024

    print(f"[*] 正在读取源文本: {src_file}...")
    with open(src_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    lines = [p.strip() for p in raw_text.split("\n") if len(p.strip()) > 30]
    if not lines:
        print("[-] 源文件有效内容不足！")
        sys.exit(1)

    print(f"[*] 种子池包含 {len(lines)} 个真实段落。开始合成 {target_chunks} 个文档切片...")

    base_vec = generate_normalized_vector(dimension)
    fmt = f"<{dimension}f"

    with open(output_bin, "wb") as f:
        # Header: total_docs, dimension
        f.write(struct.pack("<II", target_chunks, dimension))

        for doc_id in range(target_chunks):
            if (doc_id + 1) % 10000 == 0:
                print(f"    生成进度: [{doc_id + 1}/{target_chunks}] ({(doc_id + 1) / target_chunks * 100:.1f}%)", end="\r")

            para_seed = lines[doc_id % len(lines)]
            chunk_text = f"【测试切片_{doc_id}】{para_seed}"
            vec = perturb_vector(base_vec)

            text_bytes = chunk_text.encode("utf-8")
            f.write(struct.pack("<II", doc_id, len(text_bytes)))
            f.write(text_bytes)
            f.write(struct.pack(fmt, *vec))

    size_mb = Path(output_bin).stat().st_size / (1024 * 1024)
    print(f"\n[✔] 生成完毕！文件: {output_bin}，切片数: {target_chunks}，体积: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
