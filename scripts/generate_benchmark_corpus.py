import sys
import struct
import random
import math
from pathlib import Path

def generate_normalized_vector(dim: int) -> list[float]:
    """生成一个服从高斯分布并归一化的单位向量（纯标准库，无需 numpy）"""
    vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 4:
        print(f"用法: python {sys.argv[0]} <源文本文件或文件夹> <目标切片总数> <输出.bin>")
        print("例如生成 10 万切片: python scripts/generate_benchmark_corpus.py test_txt 100000 bench_30m.bin")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    target_chunks = int(sys.argv[2])
    output_bin = sys.argv[3]
    dimension = 1024

    print(f"[*] 正在读取源文本: {src_path}...")
    files = [src_path] if src_path.is_file() else sorted(src_path.rglob("*.txt"))
    lines = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for p in f:
                s = p.strip()
                if len(s) > 30:
                    lines.append(s[:300])

    if not lines:
        print("[-] 源文件有效内容不足！")
        sys.exit(1)

    print(f"[*] 种子池包含 {len(lines)} 个真实段落。开始极速合成 {target_chunks} 个文档切片...")
    t0 = random.Random(42)

    # 预生成 256 个已打包的归一化向量二进制块（将 10 万切片合成耗时从 ~35 秒压降至 ~1 秒）
    pool_size = 256
    fmt = f"<{dimension}f"
    packed_vectors = [
        struct.pack(fmt, *generate_normalized_vector(dimension))
        for _ in range(pool_size)
    ]

    with open(output_bin, "wb", buffering=8 * 1024 * 1024) as f:
        f.write(struct.pack("<II", target_chunks, dimension))

        for doc_id in range(target_chunks):
            if (doc_id + 1) % 10000 == 0:
                print(f"    生成进度: [{doc_id + 1}/{target_chunks}] ({(doc_id + 1) / target_chunks * 100:.1f}%)", end="\r")

            para_seed = lines[doc_id % len(lines)]
            chunk_text = f"【测试切片_{doc_id}】{para_seed}"
            text_bytes = chunk_text.encode("utf-8")

            f.write(struct.pack("<II", doc_id, len(text_bytes)))
            f.write(text_bytes)
            f.write(packed_vectors[doc_id & (pool_size - 1)])

    size_mb = Path(output_bin).stat().st_size / (1024 * 1024)
    print(f"\n[+] 生成完毕！文件: {output_bin}，切片数: {target_chunks}，体积: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
