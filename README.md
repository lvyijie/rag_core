# vrag-core

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus)
![SIMD](https://img.shields.io/badge/SIMD-AVX--VNNI%20%7C%20AVX2-0071C5?style=flat-square&logo=intel)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-3FB950?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **说明**：本项目是一个**个人学习项目**，旨在从零探索现代 C++20、SIMD 硬件指令集优化（AVX2 / AVX-VNNI）、内存对齐管理以及本地 RAG 双路混合检索的底层实现。

**vrag-core** 是一个零外部依赖、纯头文件（Header-Only）的轻量级混合检索内核，将 **BM25 稀疏倒排检索** 与 **SQ8 量化稠密向量检索** 结合，通过 **RRF（倒数排名融合）** 输出重排结果。

---

## 核心特性

- **双路混合检索**：结合 BM25 倒排索引与 INT8 量化向量扫描，利用 RRF 融合字面精确匹配与语义相似度（无向量输入时自动退化为纯 BM25）。
- **零拷贝流式分词**：支持 CJK UTF-8 二元滑窗（Bi-gram）与大小写不敏感的 ASCII 字母数字（`[a-zA-Z0-9]+`）切分，单核吞吐约 `3.3 GB/s`。
- **SIMD 向量量化算子**：支持 `AVX-VNNI` (`vpdpbusd`) 与 `AVX2 + FMA` 点积加速，对称 INT8 量化节省 75% 向量内存，非 AVX2 平台提供标量回退。
- **低开销内存与并发设计**：内置 64B 对齐自动扩容 `Arena` 内存池与无锁查询工作区池，支持多线程 `const` 并发查询、文档更新（Upsert）、墓碑软删除及预量化索引快照（`save_index` / `load_index`）。

---

## 目录结构

```text
rag_core/
├── include/
│   ├── arena.hpp           # 64B 对齐自动扩容线性内存池
│   ├── inverted_index.hpp  # 扁平化内存倒排索引与 BM25 打分
│   ├── simd_int8.hpp       # SQ8 量化与 AVX-VNNI / AVX2 点积算子
│   ├── simd_vector.hpp     # FP32 AVX2 向量点积与 L2 归一化
│   ├── tokenizer.hpp       # UTF-8 CJK Bi-gram 与 ASCII 字母数字分词器
│   └── vrag.hpp            # 混合检索主引擎与 RRF 融合
├── src/                    # 交互式 CLI (cli_search) 与基准测试工具 (bench_search)
├── tests/                  # 单元测试与召回率评测
├── benchmarks/             # 分词、SIMD 与端到端微基准测试
└── scripts/                # 文本切块、API 向量化与离线压测数据生成脚本
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
    return 0;
}
```

### 3. 语料构建与命令行工具

```powershell
# 1. 使用硅基流动 Embedding API 切块并构建真实语料包
$env:SILICONFLOW_API_KEY="sk-..."
python scripts/build_corpus.py ./docs corpus.bin

# 2. 或离线生成压测数据集 (无需 API Key)
python scripts/generate_benchmark_corpus.py ./docs bench_100k.bin 100000

# 3. 运行交互式检索与性能压测
./build/cli_search corpus.bin
./build/bench_search bench_100k.bin 500
```

---

## 基准测试参考

测试环境：Intel Core i7-12700H (14C/20T, AVX2 + AVX-VNNI), GCC 16.1.0 (`-O3 -march=native`), Dual-Channel DDR5

| 测试项 | 规模 / 维度 | 平均延迟 | P50 | P99 | 吞吐量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tokenizer_bench` | 1 MiB 中英混合文本 | `0.303 ms` | `0.294 ms` | `0.359 ms` | `3,303 MiB/s` |
| `simd_bench` | 10,000 × 768d (单核 FP32) | `1.22 ms` | - | - | `8.19M vec/s` |
| `vrag_bench` | 10,000 切片 × 768d | `0.244 ms` | `0.224 ms` | `0.661 ms` | `4,091 QPS` |
| `bench_search` | 17,557 切片 × 1024d | `0.244 ms` | `0.237 ms` | `0.420 ms` | `4,102 QPS` |
| `bench_search` | 100,000 切片 × 1024d | `2.47 ms` | `2.41 ms` | `4.10 ms` | `404.8 QPS` |

---

## License

[MIT License](LICENSE)
