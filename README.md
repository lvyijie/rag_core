# vrag-core

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus)
![SIMD](https://img.shields.io/badge/SIMD-AVX--VNNI%20%7C%20AVX2-0071C5?style=flat-square&logo=intel)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-3FB950?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **说明**：本项目是一个**个人学习项目**，用于探索现代 C++20、SIMD 指令集优化（AVX2 / AVX-VNNI）、内存对齐管理以及本地 RAG 混合检索的底层实现。

**vrag-core** 是一个零外部依赖、纯头文件（Header-Only）的轻量级内存混合检索内核，将 **BM25 稀疏倒排检索** 与 **SQ8 标量量化稠密向量扫描** 结合，通过 **RRF（倒数排名融合）** 输出重排结果。

---

## 核心特性

- **双路混合检索**：结合 BM25 倒排索引与 INT8 量化向量扫描，利用 RRF 融合字面匹配与向量相似度（支持仅稀疏、仅稠密、双路融合三种模式）。
- **零拷贝流式分词**：支持 CJK UTF-8 二元滑窗（Bi-gram）与大小写不敏感的 ASCII 字母数字（`[a-zA-Z0-9]+`）切分，不含词干还原（Stemmer）。
- **SIMD 向量算子**：支持 `AVX-VNNI` (`vpdpbusd`) 与 `AVX2 + FMA` 点积计算，对称 INT8 量化减少 75% 向量内存占用，非 AVX2 环境提供标量回退。
- **并发与持久化支持**：内置 64B 对齐自动扩容 `Arena` 内存池与无锁查询工作区池，支持多线程 `const` 查询、文档幂等更新（Upsert）、墓碑软删除及预量化索引文件存取（`save_index` / `load_index`）。

---

## 目录结构

```text
rag_core/
├── include/
│   ├── arena.hpp           # 64B 对齐自动扩容线性内存池
│   ├── inverted_index.hpp  # 内存倒排索引与 BM25 打分
│   ├── simd_int8.hpp       # SQ8 量化与 AVX-VNNI / AVX2 点积算子
│   ├── simd_vector.hpp     # FP32 AVX2 向量点积与 L2 归一化
│   ├── tokenizer.hpp       # UTF-8 CJK Bi-gram 与 ASCII 字母数字分词器
│   └── vrag.hpp            # 混合检索主引擎、CRUD 与 RRF 融合
├── src/                    # CLI (cli_search)、压测 (bench_search) 与 BEIR 消融评测 (beir_bench)
├── tests/                  # 单元测试与召回率测试
├── benchmarks/             # 分词、SIMD 与端到端微基准测试
└── scripts/                # 语料构建与 BEIR 自动评测脚本
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

    engine.save_index("snapshot.vidx");
    return 0;
}
```

### 3. 运行 BEIR 基准与消融评测

```powershell
# 1. 使用本地开源模型 (FastEmbed BAAI/bge-small-en-v1.5) 运行 BM25 vs Dense vs Hybrid 三路消融评测
python scripts/run_beir_benchmark.py nfcorpus scifact --local-embed

# 2. 纯 BM25 模式评测 (自动下载数据集并输出 NDCG@10、Recall 与延迟)
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --bm25

# 3. 1024 维混合检索管线延迟评测
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --hybrid-sim
```

---

## 基准测试与消融实验 (Benchmarks & Ablation)

在 Intel Core i7-12700H 平台上，使用下述测试配置对 `vrag-core` 进行了本机内存检索测试。以下数据仅反映单机纯内存检索耗时與对应配置下的检索质量，不代表生产环境下的多客户端并发服务容量或端到端 Embedding 推理耗时。

### 测试环境与口径说明

- **硬件与系统**：Intel Core i7-12700H (14C/20T, 6P+8E), 双通道 DDR5, Windows 11 x86_64
- **编译器与编译选项**：MinGW-w64 GCC 16.1.0, `-O3 -march=native -fopenmp`
- **线程策略**：
  - 纯 BM25 检索路径：单线程执行
  - 稠密向量扫描路径：当文档槽位数 `<= 4096` 时单线程执行；`> 4096` 时启用 OpenMP 多线程动态调度（`schedule(dynamic, 1024)`）
- **计时范围与预热**：数据预先加载并常驻内存，经 20 次查询预热后，单客户端串行逐条测量 `engine.search(..., top_k=10)` 的完整耗时（含 Query 分词、BM25 倒排打分、Query 向量归一化与量化、INT8 点积扫描、Top-200 候选堆维护、RRF 融合及 Top-10 结果构造），不含外部 Embedding 模型推理时间。表中 `QPS` 按 `1000 / Avg_ms` 折算，代表单流顺序吞吐率。
- **分词与检索参数**：UTF-8 CJK Bi-gram + ASCII `[a-zA-Z0-9]+` 小写折叠（无词干提取 Stemmer），BM25 参数 `k1 = 1.2, b = 0.75`，双路等权 RRF 参数 `rrf_k = 60.0`（各取 Top-200 候选融合）；当查询词项数 `>= 3` 且存在 `df <= 50% N` 的词项时，跳过 `df > 50% N` 的高频词倒排链。

### 1. 真实语义向量下的三路消融实验 (`BAAI/bge-small-en-v1.5`, 384d)

在 `BEIR/NFCorpus`（3,633 篇医学文献，323 条测试 Query）与 `BEIR/SciFact`（5,183 篇科学文献，300 条测试 Query）上，使用同一批 `BAAI/bge-small-en-v1.5` 真实语义向量对比单路稀疏、单路稠密（SQ8 量化）与双路 RRF 混合检索：

| 数据集 | 检索模式 | `NDCG@10` | `MRR@10` | `Recall@10` | `Recall@100` | 延迟 (Avg / P50 / P99) | 单流吞吐 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus**<br>*(3,633 docs, 384d)* | 1. BM25 Only (Sparse) | `0.3083` | `0.5140` | `0.1492` | `0.2342` | `0.018 / 0.013 / 0.088 ms` | `54,258 QPS` |
| | 2. Dense SQ8 Only (Vector) | `0.3391` | `0.5253` | `0.1597` | `0.3056` | `0.071 / 0.068 / 0.133 ms` | `14,154 QPS` |
| | **3. Hybrid (BM25 + SQ8 + RRF)** | **`0.3575`** | **`0.5735`** | **`0.1704`** | **`0.3142`** | `0.089 / 0.084 / 0.201 ms` | `11,176 QPS` |
| **SciFact**<br>*(5,183 docs, 384d)* | 1. BM25 Only (Sparse) | `0.6622` | `0.6288` | `0.7889` | `0.8892` | `0.056 / 0.053 / 0.139 ms` | `17,764 QPS` |
| | 2. Dense SQ8 Only (Vector) | **`0.7246`** | **`0.6897`** | **`0.8486`** | `0.9533` | `0.120 / 0.117 / 0.211 ms` | `8,303 QPS` |
| | **3. Hybrid (BM25 + SQ8 + RRF)** | `0.7160` | `0.6832` | `0.8379` | **`0.9617`** | `0.165 / 0.162 / 0.239 ms` | `6,072 QPS` |

> **消融结果观察**：
> - 在 `NFCorpus` 上，稀疏检索（`0.3083`）与稠密检索（`0.3391`）能力相近且互补，等权 RRF 融合将 `NDCG@10` 提升至 **`0.3575`**（较 BM25 提升 `+0.0492`，较纯向量提升 `+0.0184`），`MRR@10` 提升至 **`0.5735`**。
> - 在 `SciFact` 上，稠密模型（`0.7246`）显著强于无词干提取的 BM25（`0.6622`）；双路融合将候选集总召回率 `Recall@100` 从 `0.8892`（BM25）/ `0.9533`（Dense）提升至 **`0.9617`**，而 Top-10 排序指标 `NDCG@10`（`0.7160`）大幅优于纯 BM25（`+0.0538`），但因无权重的 1:1 RRF 受较弱一路拖累而略低于纯 Dense（`-0.0086`）。

### 2. BEIR 四大数据集纯 BM25 与 1024 维混合管线耗时 (Top-10)

| 数据集 | 文档数 / Query 数 | BM25 `NDCG@10` | BM25 `Recall@100` | 纯 BM25 延迟 (Avg / P50 / P99) | 纯 BM25 单流吞吐 | 混合管线 1024d 延迟 (Avg / P50 / P99) | 混合管线单流吞吐 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 5,183 / 300 | `0.6622` | `0.8892` | `0.058 / 0.056 / 0.146 ms` | `17,214 QPS` | `0.193 / 0.191 / 0.316 ms` | `5,178 QPS` |
| **NFCorpus** | 3,633 / 323 | `0.3083` | `0.2342` | `0.017 / 0.013 / 0.066 ms` | `59,233 QPS` | `0.115 / 0.110 / 0.169 ms` | `8,681 QPS` |
| **ArguAna** | 8,674 / 1,401 | `0.3149` | `0.9208` | `0.338 / 0.323 / 0.608 ms` | `2,957 QPS` | `0.572 / 0.563 / 0.846 ms` | `1,749 QPS` |
| **FiQA** | 57,638 / 648 | `0.2361` | `0.5081` | `0.373 / 0.344 / 0.923 ms` | `2,683 QPS` | `1.405 / 1.381 / 2.063 ms` | `712 QPS` |

### 3. 算子微基准与中文语料测试

| 测试目标 | 工作负载描述 | 线程数 | 平均延迟 | P50 延迟 | P99 延迟 | 吞吐率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tokenizer_bench` | 1 MiB UTF-8 中英数混合文本分词 | 单线程 | `0.291 ms` | `0.289 ms` | `0.324 ms` | `3,431 MiB/s` |
| `simd_bench` | 10,000 × 768d FP32 向量点积扫描 | 单线程 | `1.210 ms` | - | - | `8.27M vec/s` |
| `bench_search` | 17,557 中文切片 × 1024d 混合检索 (Top-5) | BM25 单线程 + 向量 OpenMP | `0.244 ms` | `0.237 ms` | `0.420 ms` | `4,102 QPS` |
| `bench_search` | 100,000 中文切片 × 1024d 混合检索 (Top-5) | BM25 单线程 + 向量 OpenMP | `2.470 ms` | `2.410 ms` | `4.100 ms` | `405 QPS` |

### 评测限制说明

- **单流顺序查询 vs 并发吞吐**：表中测量的是单客户端串行请求延迟与折算的单流吞吐率（`1000 / Avg_ms`），不代表多线程/多客户端并发压测下的服务端吞吐容量。
- **内存常驻 vs 冷启动/磁盘检索**：所有评测均在索引与向量完全加载至物理内存且经过预热后执行，不包含冷启动文件加载、页面缺页中断或磁盘 I/O 开销。
- **混合模式向量说明**：表 1（三路消融实验）使用 `BAAI/bge-small-en-v1.5` 真实语义向量评测检索质量与延迟；表 2 中的“混合管线 1024d 延迟”采用 `--hybrid-sim` 归一化合成向量专门测量 1024 维全量扫描与 RRF 管线的计算耗时。
- **中英文分词行为差异**：内置分词器对英文按连续字母数字（`[a-zA-Z0-9]+`）切分，对中文按 UTF-8 二元滑窗（Bi-gram）切分。BEIR 英文数据集与中文语料触发的是不同的分词分支，两者的词表规模与倒排链分布特征不同。
- **检索质量指标的可比性条件**：`NDCG@10`、`Recall@100` 等指标受分词粒度、是否启用词干提取（Stemmer）、停用词过滤策略及 BM25/RRF 超参数影响；与其他检索引擎对比质量或延迟时，需确保语料、Query、`qrels` 及预处理配置一致。

---

## License

[MIT License](LICENSE)
