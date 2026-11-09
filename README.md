# vrag-core

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus)
![SIMD](https://img.shields.io/badge/SIMD-AVX--VNNI%20%7C%20AVX2-0071C5?style=flat-square&logo=intel)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-3FB950?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **说明**：本项目是一个**个人学习项目**，旨在从零探索现代 C++20、SIMD 硬件指令集优化（AVX2 / AVX-VNNI）、内存对齐管理以及本地 RAG 双路混合检索的底层实现。

**vrag-core** 是一个零外部依赖、纯头文件（Header-Only）的轻量级内存混合检索内核，将 **BM25 稀疏倒排检索** 与 **SQ8 量化稠密向量检索** 结合，通过 **RRF（倒数排名融合）** 输出重排结果。

---

## 核心特性

- **双路混合检索**：结合 BM25 倒排索引与 INT8 量化向量扫描，利用 RRF 融合字面精确匹配与语义相似度（无向量输入时自动退化为纯 BM25）。
- **零拷贝流式分词**：支持 CJK UTF-8 二元滑窗（Bi-gram）与大小写不敏感的 ASCII 字母数字（`[a-zA-Z0-9]+`）切分，单核吞吐约 `3.4 GB/s`。
- **SIMD 向量量化算子**：支持 `AVX-VNNI` (`vpdpbusd`) 与 `AVX2 + FMA` 点积加速，对称 INT8 量化节省 75% 向量内存，非 AVX2 平台提供标量回退。
- **工程与并发完备**：内置 64B 对齐自动扩容 `Arena` 内存池与无锁查询工作区池，支持多线程 `const` 并发查询、幂等更新（Upsert）、$O(1)$ 墓碑软删除及预量化索引快照（`save_index` / `load_index`）。

---

## 目录结构

```text
rag_core/
├── include/
│   ├── arena.hpp           # 64B 对齐自动扩容线性内存池
│   ├── inverted_index.hpp  # 扁平化内存倒排索引与 BM25 打分 (含高频停用词剪枝)
│   ├── simd_int8.hpp       # SQ8 量化与 AVX-VNNI / AVX2 点积算子
│   ├── simd_vector.hpp     # FP32 AVX2 向量点积与 L2 归一化
│   ├── tokenizer.hpp       # UTF-8 CJK Bi-gram 与 ASCII 字母数字分词器
│   └── vrag.hpp            # 混合检索主引擎、CRUD 生命周期与 RRF 融合
├── src/                    # CLI (cli_search)、压测 (bench_search) 与 BEIR 评测器 (beir_bench)
├── tests/                  # 单元测试与召回率评测
├── benchmarks/             # 分词、SIMD 与端到端微基准测试
└── scripts/                # 语料切块、API 向量化与官方 BEIR 自动评测脚本
```

---

## 快速开始

### 1. 编译与测试

**Linux / macOS**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

**Windows (MinGW-w64)**
```powershell
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

### 2. 代码调用示例

```cpp
#include "vrag.hpp"
#include <iostream>
#include <vector>

int main() {
    vrag::VragEngine engine(/*dimension=*/1024);

    std::vector<float> vec(1024, 0.02f);
    engine.add_document(1, "AVX2 intrinsics accelerate int8 dot-product kernels.", vec);

    auto hits = engine.search("AVX2 dot-product", vec, /*top_k=*/5);
    for (const auto& hit : hits) {
        std::cout << "doc_id=" << hit.doc_id << " score=" << hit.score
                  << " text=" << hit.text << "\n";
    }

    // 保存/加载预量化索引快照 (免分词与量化极速恢复)
    engine.save_index("snapshot.vidx");
    return 0;
}
```

### 3. 运行官方 BEIR 标准基准评测

```powershell
# 1. 纯 BM25 基线评测 (自动下载官方数据集并输出标准 TREC NDCG@10 / Recall@100 与延迟)
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --bm25

# 2. 双路混合检索 (BM25 + 1024d SQ8 + RRF) 全链路延迟评测
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --hybrid-sim
```

---

## 基准测试 (Benchmarks)

测试环境：Intel Core i7-12700H (14C/20T, AVX2 + AVX-VNNI), GCC 16.1.0 (`-O3 -march=native -fopenmp`), Dual-Channel DDR5

### 1. 国际标准 BEIR 评测集实测（真实语料 + 真实长短 Query）

| BEIR 数据集 | 领域与 Query 特征 | 文档数 / 测试 Query 数 | vrag-core `NDCG@10` | Lucene (BEIR 论文参考) | 纯 BM25 延迟 (Avg / P99) | 纯 BM25 QPS | 混合检索 1024d 延迟 (Avg / P99) | 混合检索 QPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 科学文献摘要 | 5,183 / 300 | **0.6622** | `0.665` | `0.058 ms` / `0.146 ms` | `17,214` | `0.193 ms` / `0.316 ms` | **5,178** |
| **NFCorpus** | 生物医学检索 | 3,633 / 323 | **0.3083** | `0.325` | `0.017 ms` / `0.066 ms` | `59,233` | `0.115 ms` / `0.169 ms` | **8,681** |
| **ArguAna** | 200 词超长段落 Query | 8,674 / 1,401 | **0.3149** | `0.315` | `0.338 ms` / `0.608 ms` | `2,957` | `0.572 ms` / `0.846 ms` | **1,749** |
| **FiQA** | 金融问答真实长句 | 57,638 / 648 | **0.2361** | `0.236` | `0.373 ms` / `0.923 ms` | `2,683` | `1.405 ms` / `2.063 ms` | **712** |

> **架构取舍说明**：在 $\le 10$ 万文档规模下，`vrag-core` 采用纯内存未压缩连续倒排表与 64B 对齐 `Arena` 暴力扫描（100% 精确向量召回），省去了磁盘段压缩解码与图索引跳转开销，因此在中小规模语料上具备极低的检索延迟与高建库吞吐（`FiQA` 5.7 万篇建库耗时 `0.63s`，约 `90,900 docs/s`）。

### 2. 核心算子与中文语料微基准

| 测试项 | 规模 / 维度 | 平均延迟 | P50 | P99 | 吞吐量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tokenizer_bench` | 1 MiB 中英混合文本 | `0.291 ms` | `0.289 ms` | `0.324 ms` | `3,431 MiB/s` |
| `simd_bench` | 10,000 × 768d (单核 FP32) | `1.210 ms` | - | - | `8.27M vec/s` |
| `bench_search` | 17,557 中文切片 × 1024d | `0.244 ms` | `0.237 ms` | `0.420 ms` | `4,102 QPS` |
| `bench_search` | 100,000 中文切片 × 1024d | `2.470 ms` | `2.410 ms` | `4.100 ms` | `405 QPS` |

---

## License

[MIT License](LICENSE)
