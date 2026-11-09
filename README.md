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
    engine.add_document(1, "AVX2 intrinsics accelerate int8 dot-product kernels.", vec, "tag:simd;lang:en");

    vrag::SearchOptions opts;
    opts.top_k = 5;
    opts.mode = vrag::SearchMode::Hybrid;
    opts.bm25_weight = 0.5f;
    opts.dense_weight = 1.0f;
    opts.filter = [](std::uint32_t /*doc_id*/, std::string_view meta) {
        return meta.find("tag:simd") != std::string_view::npos;
    };

    auto hits = engine.search("AVX2 dot-product", vec, opts);
    for (const auto& hit : hits) {
        std::cout << "doc_id=" << hit.doc_id
                  << " score=" << hit.score
                  << " (bm25=" << hit.bm25_score << " #" << hit.bm25_rank
                  << ", dense=" << hit.dense_score << " #" << hit.dense_rank << ")"
                  << " meta=" << hit.metadata << "\n";
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

- **硬件与系统**：Intel Core i7-12700H (14C/20T, 6P+8E, 24 MB L3 Cache), 双通道 DDR5, Windows 11 x86_64
- **编译器与编译选项**：MinGW-w64 GCC 16.1.0, `-O3 -march=native -fopenmp`
- **线程策略（明确区分单线程与 OpenMP）**：
  - 纯 BM25 检索路径：单线程执行
  - 稠密向量扫描路径：支持通过 `SearchOptions::num_threads` 显式指定单线程（`1`）或 OpenMP 多线程（`> 1`）。默认自适应策略（`num_threads = 0`）在槽位数 `<= 16,384`（量化向量完全驻留 L3 Cache）时采用单线程扫描以避免 OpenMP 线程屏障开销，在 `> 16,384` 时启用 OpenMP 多线程动态调度（`schedule(dynamic, 1024)`）。下表同时列出 `1-Thread` 与 `OpenMP (20T)` 的实测对照。
- **统计稳定性与计时范围**：数据预先加载并常驻内存，经 20 次查询预热后，对每条测试 Query **重复测量 5 轮并取其中位数**作为该 Query 的稳定耗时，再跨全部 Query 汇总计算 `Avg`、`P50`、`P99`（降低操作系统调度抖动对尾延迟的干扰）。计时区间包含：Query 分词、BM25 倒排打分、Query 向量归一化与量化、INT8 点积扫描、Top-200 候选堆维护、RRF 融合及 Top-10 结果构造（不含外部 Embedding 模型推理时间）。单流 `QPS` 按 `1000 / Avg_ms` 折算；多线程并发读取吞吐另见表 3。
- **分词与检索参数**：UTF-8 CJK Bi-gram + ASCII `[a-zA-Z0-9]+` 小写折叠（无词干提取 Stemmer），BM25 参数 `k1 = 1.2, b = 0.75`，RRF 参数 `rrf_k = 60.0`（各取 Top-200 候选融合）；当查询词项数 `>= 3` 且存在 `df <= 50% N` 的词项时，跳过 `df > 50% N` 的高频词倒排链。

### 1. 实验一：384 维真实语义向量消融实验与 FP32 vs SQ8 精度对比 (`BAAI/bge-small-en-v1.5`)

在 `BEIR/NFCorpus`（3,633 篇医学文献，323 条测试 Query）与 `BEIR/SciFact`（5,183 篇科学文献，300 条测试 Query）上，使用本地 ONNX 模型 `BAAI/bge-small-en-v1.5` 生成的同一批 384 维真实语义向量，对比单路稀疏、**FP32 精确向量基线**、**SQ8 量化向量检索（单线程 vs OpenMP）**与双路 RRF 混合检索：

| 数据集 | 检索模式与线程配置 | `NDCG@10` | `MRR@10` | `Recall@10` | `Recall@100` | 延迟 (Avg / P50 / P99) | 单流吞吐 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus**<br>*(3,633 docs, 384d)* | 1. BM25 Only (1-Thread) | `0.3083` | `0.5140` | `0.1492` | `0.2342` | `0.013 / 0.009 / 0.053 ms` | `78,114 QPS` |
| | 2. Dense FP32 (1-Thread) | `0.3401` | `0.5305` | `0.1612` | `0.3059` | `0.137 / 0.130 / 0.233 ms` | `7,295 QPS` |
| | 3a. Dense SQ8 (1-Thread) | `0.3391` | `0.5253` | `0.1597` | `0.3056` | `0.064 / 0.062 / 0.103 ms` | `15,627 QPS` |
| | 3b. Dense SQ8 (OpenMP 20T) | `0.3391` | `0.5253` | `0.1597` | `0.3056` | `0.111 / 0.107 / 0.166 ms` | `8,973 QPS` |
| | 4. Hybrid FP32 (1:1, 1-Thread) | **`0.3586`** | `0.5730` | **`0.1746`** | **`0.3161`** | `0.145 / 0.140 / 0.245 ms` | `6,873 QPS` |
| | **5a. Hybrid SQ8 (1:1, 1-Thread)** | `0.3575` | **`0.5735`** | `0.1704` | `0.3142` | `0.076 / 0.073 / 0.129 ms` | `13,074 QPS` |
| | 5b. Hybrid SQ8 (1:1, OpenMP 20T) | `0.3575` | **`0.5735`** | `0.1704` | `0.3142` | `0.131 / 0.128 / 0.221 ms` | `7,646 QPS` |
| | 6. Hybrid SQ8 (0.3:1 W-RRF, 1T) | `0.3509` | `0.5510` | `0.1675` | `0.3120` | `0.082 / 0.078 / 0.128 ms` | `12,264 QPS` |
| **SciFact**<br>*(5,183 docs, 384d)* | 1. BM25 Only (1-Thread) | `0.6622` | `0.6288` | `0.7889` | `0.8892` | `0.045 / 0.042 / 0.099 ms` | `22,212 QPS` |
| | 2. Dense FP32 (1-Thread) | `0.7203` | `0.6847` | `0.8452` | `0.9533` | `0.202 / 0.193 / 0.351 ms` | `4,963 QPS` |
| | 3a. Dense SQ8 (1-Thread) | `0.7246` | `0.6897` | `0.8486` | `0.9533` | `0.079 / 0.077 / 0.124 ms` | `12,667 QPS` |
| | 3b. Dense SQ8 (OpenMP 20T) | `0.7246` | `0.6897` | `0.8486` | `0.9533` | `0.131 / 0.132 / 0.175 ms` | `7,611 QPS` |
| | 4. Hybrid FP32 (1:1, 1-Thread) | `0.7181` | `0.6850` | `0.8412` | **`0.9617`** | `0.243 / 0.241 / 0.354 ms` | `4,115 QPS` |
| | 5a. Hybrid SQ8 (1:1, 1-Thread) | `0.7160` | `0.6832` | `0.8379` | **`0.9617`** | `0.132 / 0.129 / 0.194 ms` | `7,551 QPS` |
| | 5b. Hybrid SQ8 (1:1, OpenMP 20T) | `0.7160` | `0.6832` | `0.8379` | **`0.9617`** | `0.192 / 0.190 / 0.267 ms` | `5,209 QPS` |
| | **6. Hybrid SQ8 (0.3:1 W-RRF, 1T)** | **`0.7264`** | **`0.6916`** | **`0.8546`** | **`0.9617`** | `0.131 / 0.128 / 0.216 ms` | `7,642 QPS` |

#### SQ8 量化误差分布与全量索引内存明细

- **Top-10 重合率定义**：对每条查询 $q \in Q$，分别取纯向量 FP32 精确检索与 SQ8 检索的前 10 名文档集合，计算交集比例 $\text{Overlap@10}(q) = \frac{|\text{Top10}_{\text{FP32}}(q) \cap \text{Top10}_{\text{SQ8}}(q)|}{10}$，再統計全部查询的均值（Mean）与第 10 百分位数（P10）：

| 数据集 | `Overlap@10` (Mean / P10) | 余弦点积绝对误差 $\|\cos_{\text{FP32}} - \cos_{\text{SQ8}}\|$ (Mean / P95 / Max) | 原始向量内存 (FP32 $\to$ SQ8) | 向量元数据 (`VecMeta`) | 倒排索引 (`InvIdx`) | 文档原文存储 (`DocTexts`) | 索引总内存 (`Total`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus** (3,633 docs, 384d) | `97.34%` / `90.00%` | `0.0009` / `0.0021` / `0.0039` | `5.32 MB -> 1.33 MB` (**4.0x**) | `0.13 MB` | `5.93 MB` | `5.79 MB` | `13.19 MB` |
| **SciFact** (5,183 docs, 384d) | `97.20%` / `90.00%` | `0.0009` / `0.0022` / `0.0041` | `7.59 MB -> 1.90 MB` (**4.0x**) | `0.18 MB` | `8.00 MB` | `7.81 MB` | `17.89 MB` |
| **FiQA** (57,638 docs, 1024d) | `100.00%` / `100.00%` | `0.0002` / `0.0005` / `0.0008` | `225.15 MB -> 56.29 MB` (**4.0x**) | `2.03 MB` | `55.72 MB` | `46.57 MB` | `160.61 MB` |

> **关键现象说明**：
> 1. **向量压缩比 vs 索引总内存**：4.0 倍压缩特指稠密向量池（`Arena` 中每个维度从 4 字节 `float` 降为 1 字节 `int8_t`）。完整引擎内存还包含向量缩放因子/偏移元数据（`VecMeta`）、BM25 倒排表（`InvIdx`）及文档原文字符串（`DocTexts`）。
> 2. **加权融合（`0.3:1 W-RRF`）与过拟合边界说明**：在 `NFCorpus` 上，BM25 与 Dense 实力相当且互补，默认 `1:1` 等权融合即取得最优（`NDCG@10 = 0.3575`）；而在 `SciFact` 上，Dense（`0.7246`）显著强于无词干提取的 BM25（`0.6622`），降低 BM25 权重至 `0.3:1` 可使四项指标（`NDCG@10 = 0.7264`）超过单路基线。注意：`0.3:1` 是针对 `SciFact` 通道强弱差异的参数敏感性观察，并不代表对所有数据集普适最优；实际业务中应通过独立验证集（Validation Set）选择通道权重。
> 3. **小规模数据下的单线程 vs OpenMP 开销**：在 `NFCorpus`（`1.33 MB` 向量）和 `SciFact`（`1.90 MB` 向量）上，全部量化向量均可装入 CPU 的 `24 MB` L3 Cache，单线程扫描仅需 `0.064 ~ 0.079 ms`，而强行开启 20 线程 OpenMP 反而会因线程唤醒与屏障同步增加约 `0.05 ms` 开销（`0.111 ~ 0.131 ms`）。而在 `FiQA`（`56.29 MB` 向量，超出 L3 Cache）上，OpenMP 20 线程（`1.085 ms`）则比单线程（`2.718 ms`）快 **2.51 倍**。

### 2. 实验二：不同文档规模下的纯 BM25 与 1024 维管线耗时压测 (Top-10)

本实验独立于实验一，旨在测试更大文档规模（最高 57,638 篇文档）和更高向量维度（1024 维）下的计算耗时。其中 1024 维向量采用 `--hybrid-sim` 生成的单位球面正态分布合成 FP32 向量（入库量化为 SQ8，各路取 Top-200 候选经 `RRF(k=60)` 融合），因此仅报告管线延迟，不评估合成向量的语义质量指标：

| 数据集 | 文档数 / Query 数 | BM25 `NDCG@10` | BM25 `Recall@100` | 纯 BM25 延迟 (1-Thread, Avg/P50/P99) | 1024d 混合检索延迟 (1-Thread) | 1024d 混合检索延迟 (OpenMP 20T) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus** | 3,633 / 323 | `0.3083` | `0.2342` | `0.017 / 0.013 / 0.066 ms` | `0.092 / 0.089 / 0.145 ms` | `0.115 / 0.110 / 0.169 ms` |
| **SciFact** | 5,183 / 300 | `0.6622` | `0.8892` | `0.058 / 0.056 / 0.146 ms` | `0.148 / 0.144 / 0.228 ms` | `0.193 / 0.191 / 0.316 ms` |
| **ArguAna** | 8,674 / 1,401 | `0.3149` | `0.9208` | `0.338 / 0.323 / 0.608 ms` | `0.512 / 0.501 / 0.780 ms` | `0.572 / 0.563 / 0.846 ms` |
| **FiQA** | 57,638 / 648 | `0.2361` | `0.5081` | `0.352 / 0.324 / 0.901 ms` | `3.058 / 3.044 / 3.927 ms` | **`1.625 / 1.599 / 2.325 ms`** |

### 3. 实验三：多线程并发读取扩展性 (`std::shared_lock` + 无锁工作区池)

区别于根据单次延迟倒数折算的“单流 QPS”，本实验测试多个独立客户端线程同时对同一个 `VragEngine` 实例发起 `Hybrid (BM25 + SQ8 + RRF, Top-10)` 查询（每个查询线程内部设置 `num_threads = 1`，共完成 2,000 次查询）时的真实并发吞吐率：

| 数据集与规模 | `1 线程` 并发吞吐 | `2 线程` 并发吞吐 | `4 线程` 并发吞吐 | `8 线程` 并发吞吐 | `16 线程` 并发吞吐 | 16 线程加速比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus** (3,633 docs, 384d) | `10,270 QPS` | `20,626 QPS` | `38,992 QPS` | `64,311 QPS` | **`107,562 QPS`** | `10.47x` |
| **SciFact** (5,183 docs, 384d) | `6,674 QPS` | `13,519 QPS` | `25,615 QPS` | `43,050 QPS` | **`66,596 QPS`** | `9.98x` |
| **FiQA** (57,638 docs, 1024d) | `318 QPS` | `542 QPS` | `996 QPS` | `1,597 QPS` | **`2,599 QPS`** | `8.17x` |

### 4. 算子微基准与中文语料测试

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
