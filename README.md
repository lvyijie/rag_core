# vrag-core

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus)
![SIMD](https://img.shields.io/badge/SIMD-AVX--VNNI%20%7C%20AVX2-0071C5?style=flat-square&logo=intel)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-3FB950?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **说明**：本项目是一个**个人学习项目**，用于探索现代 C++20、SIMD 指令集优化（AVX2 / AVX-VNNI）、内存对齐管理以及本地 RAG 混合检索的底层实现。

**vrag-core** 是一个零外部依赖、纯头文件（Header-Only）的轻量级内存混合检索内核，将 **BM25 稀疏倒排检索** 与 **SQ8 标量量化稠密向量扫描** 结合，通过 **RRF（倒数排名融合）** 输出重排结果。

---

## 核心特性

- **双路混合检索**：结合 BM25 倒排索引与 INT8 量化向量扫描，利用 RRF 融合字面匹配与向量相似度（无向量输入时退化为纯 BM25）。
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
├── src/                    # CLI (cli_search)、压测 (bench_search) 与 BEIR 评测 (beir_bench)
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

### 3. 运行 BEIR 基准测试

```powershell
# 纯 BM25 模式评测 (自动下载数据集并输出 NDCG@10、Recall 与延迟)
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --bm25

# 混合模式 (BM25 + 1024d SQ8 全扫 + RRF) 延迟评测
python scripts/run_beir_benchmark.py scifact nfcorpus arguana fiqa --hybrid-sim
```

---

## 基准测试 (Benchmarks)

在 Intel Core i7-12700H 平台上，使用下述测试配置对 `vrag-core` 进行了本机内存检索测试。在 BEIR `SciFact`（5,183 篇）和 `FiQA`（57,638 篇）上，纯 BM25 平均查询延迟分别为 `0.058 ms` 和 `0.373 ms`，开启 1024 维 SQ8 向量全扫与 RRF 融合后的平均查询延迟分别为 `0.193 ms` 和 `1.405 ms`。以上数据仅反映单机纯内存检索耗时，不代表生产环境下的多客户端并发服务容量或端到端 Embedding 推理耗时。

### 测试环境与口径说明

- **硬件与系统**：Intel Core i7-12700H (14C/20T, 6P+8E), 双通道 DDR5, Windows 11 x86_64
- **编译器与编译选项**：MinGW-w64 GCC 16.1.0, `-O3 -march=native -fopenmp`
- **线程策略**：
  - 纯 BM25 检索路径：单线程执行
  - 稠密向量扫描路径：当文档槽位数 `<= 4096` 时单线程执行；`> 4096` 时启用 OpenMP 多线程动态调度（`schedule(dynamic, 1024)`）
- **计时范围与预热**：数据预先加载并常驻内存，经 20 次查询预热后，单客户端串行逐条测量 `engine.search(..., top_k=10)` 的完整耗时（含 Query 分词、BM25 倒排打分、Query 向量归一化与量化、INT8 点积扫描、Top-200 候选堆维护、RRF 融合及 Top-10 结果构造），不含外部 Embedding 模型推理时间。表中 `QPS` 按 `1000 / Avg_ms` 折算，代表单流顺序吞吐率。
- **分词与 BM25 参数**：UTF-8 CJK Bi-gram + ASCII `[a-zA-Z0-9]+` 小写折叠（无词干提取 Stemmer），BM25 参数 `k1 = 1.2, b = 0.75, rrf_k = 60.0`；当查询词项数 `>= 3` 且存在 `df <= 50% N` 的词项时，跳过 `df > 50% N` 的高频词倒排链。

### 1. BEIR 数据集本机实测结果 (Top-10)

| 数据集 | 文档数 / Query 数 | BM25 `NDCG@10` | BM25 `Recall@100` | 纯 BM25 延迟 (Avg / P50 / P99) | 纯 BM25 单流吞吐 | 混合检索 1024d 延迟 (Avg / P50 / P99) | 混合检索单流吞吐 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciFact** | 5,183 / 300 | `0.6622` | `0.8892` | `0.058 / 0.056 / 0.146 ms` | `17,214 QPS` | `0.193 / 0.191 / 0.316 ms` | `5,178 QPS` |
| **NFCorpus** | 3,633 / 323 | `0.3083` | `0.2342` | `0.017 / 0.013 / 0.066 ms` | `59,233 QPS` | `0.115 / 0.110 / 0.169 ms` | `8,681 QPS` |
| **ArguAna** | 8,674 / 1,401 | `0.3149` | `0.9208` | `0.338 / 0.323 / 0.608 ms` | `2,957 QPS` | `0.572 / 0.563 / 0.846 ms` | `1,749 QPS` |
| **FiQA** | 57,638 / 648 | `0.2361` | `0.5081` | `0.373 / 0.344 / 0.923 ms` | `2,683 QPS` | `1.405 / 1.381 / 2.063 ms` | `712 QPS` |

### 2. 算子微基准与中文语料测试

| 测试目标 | 工作负载描述 | 线程数 | 平均延迟 | P50 延迟 | P99 延迟 | 吞吐率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tokenizer_bench` | 1 MiB UTF-8 中英数混合文本分词 | 单线程 | `0.291 ms` | `0.289 ms` | `0.324 ms` | `3,431 MiB/s` |
| `simd_bench` | 10,000 × 768d FP32 向量点积扫描 | 单线程 | `1.210 ms` | - | - | `8.27M vec/s` |
| `bench_search` | 17,557 中文切片 × 1024d 混合检索 (Top-5) | BM25 单线程 + 向量 OpenMP | `0.244 ms` | `0.237 ms` | `0.420 ms` | `4,102 QPS` |
| `bench_search` | 100,000 中文切片 × 1024d 混合检索 (Top-5) | BM25 单线程 + 向量 OpenMP | `2.470 ms` | `2.410 ms` | `4.100 ms` | `405 QPS` |

---

## License

[MIT License](LICENSE)
