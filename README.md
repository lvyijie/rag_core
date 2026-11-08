# VRAG Core — 高性能 C++20 本地混合检索内核

![C++20](https://img.shields.io/badge/std-C%2B%2B20-blue.svg)
![ISA](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-orange.svg)
![Parallel](https://img.shields.io/badge/Parallel-OpenMP-green.svg)
![Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)

**VRAG Core** 是一个基于现代 C++20 从零构建的超轻量级本地 RAG（检索增强生成）混合检索内核。

在 **100,000 篇文档切片（约 3000 万字中文语料，1024 维向量）** 的规模下，引擎无需依赖重型外部服务，仅凭 ~100 MB 内存即可在 **~2.88 ms** 内完成“双路全量检索 + RRF 重排”，单机吞吐量达 **~347 QPS**，并保持精确扫描召回率。

---

## 核心特性

* **双路混合检索 (Hybrid Search)**：结合针对 CJK 语料优化的 Bi-gram 内存倒排索引（BM25）与稠密向量检索（Dense Vector），兼顾专有名词精确匹配与跨词抽象语义泛化。
* **RRF 倒数排名融合 (Reciprocal Rank Fusion)**：采用无量纲排名融合算法抹平稀疏与稠密分值尺度差异，通过两路共振打分提升 Top-K 命中精度。
* **SQ8 标量量化破除带宽墙**：将 1024 维 `float32` 向量对称量化为 `int8_t` 并配对独立缩放因子（Scale），将向量内存体积压缩 **75%**（从 ~390 MB 降至 ~97 MB），大幅缓解双通道内存总线带宽瓶颈。
* **手写 AVX2 + FMA 硬件算子**：基于 `<immintrin.h>` 实现 32 字节对齐的 SIMD 向量归一化与 `fp32 × int8` 混合精度点积流水线。
* **零碎片 Arena 内存池**：自定义线性分配器（Arena Allocator）统一接管百兆级向量空间，确保 32 字节边界对齐与缓存行（Cache Line）连续性。
* **局部偏序剪枝**：利用 `std::partial_sort` 与小顶堆（Min-Heap）将排序复杂度从 $O(N \log N)$ 压降至 $O(N \log K)$，消除全量排序算力浪费。

---

## 系统架构与检索流水线

```text
Query ──┬─► [Bi-gram Tokenizer] ──► [Inverted Index (BM25)] ──────► Top-200 稀疏候选 ─┐
        │                                                                            ├──► [RRF 融合 + 小顶堆] ──► Top-K 结果
        └─► [Query L2 Normalize] ─► [OpenMP + AVX2 INT8 全库扫描] ─► Top-200 稠密候选 ─┘
```

---

## 性能演进与基准压测

### 测试环境与数据集
* **语料规模**：100,000 篇文档切片（约 3,000 万汉字，基于《红楼梦》文本合成扩充）
* **向量维度**：1024 维
* **压测请求**：500 次连续混合检索（Top-5）

### 优化迭代对照表

以下为同一硬件环境下各架构演进阶段的实测表现：

| 迭代阶段 | 核心架构变更 | 向量内存 | 平均延迟 | P50 延迟 | P99 延迟 | 吞吐量 (QPS) | Top-5 召回率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1 初始基线** | 单线程 FP32 + 字符串哈希表 | 390.6 MB | 41.40 ms | 40.85 ms | 45.12 ms | 24.1 | 100% |
| **V2 多核扁平化** | OpenMP 多核 + 扁平化连续数组 | 390.6 MB | 13.34 ms | 12.75 ms | 16.79 ms | 75.0 | 100% |
| **V3 两阶段漏斗** | BM25 粗筛 (2000) + Dense 精排 | 97.6 MB | 1.12 ms | 0.20 ms | 3.30 ms | 895.5 | 16% (语义截断) |
| **V4 最终正式版** | **SQ8 量化 + 双路全扫 + RRF** | **97.6 MB** | **2.88 ms** | **2.45 ms** | **5.54 ms** | **347.4** | **100%** |

### 核心瓶颈复盘：为什么 V2 开满 20 线程仍卡在 13ms？
1. **内存总线带宽墙 (Memory Bandwidth Wall)**：在 V2（FP32）中，单次查询需扫描 $100,000 \times 1024 \times 4\text{ B} \approx 390.6\text{ MB}$。主流双通道内存的实际并发读取带宽约为 30~40 GB/s，光是将数据从 DRAM 搬运至 CPU 寄存器就需要约 10~13 ms。此时继续增加线程数只会加剧内存控制器排队与大小核同步等待（Barrier Stall）。
2. **SQ8 破局**：V4 引入 INT8 对称标量量化后，单次扫描数据搬运量骤降至 **97.6 MB**，直接将内存传输耗时斩断 75%，配合 AVX2 指令流水线将延迟压入 **2.88 ms**。
3. **两阶段粗排取舍**：V3 曾尝试用 BM25 截断前 2,000 篇文档再做向量重排，虽然 P50 降至 0.2 ms，但在跨词抽象语义查询下出现了严重的“召回漏失”（一致率跌至 16%）。最终 V4 回归双路独立全召回架构，在不损失精度的前提下保持 < 3 ms 的响应速度。

---

## 目录结构

```text
rag_core/
├── CMakeLists.txt
├── include/
│   ├── arena.hpp           # 32字节对齐的零碎片线性内存池
│   ├── inverted_index.hpp  # 扁平化内存倒排索引与 BM25 打分器
│   ├── simd_int8.hpp       # SQ8 对称标量量化与 AVX2+FMA 混合点积算子
│   ├── simd_vector.hpp     # FP32 AVX2 向量算子与归一化
│   ├── tokenizer.hpp       # UTF-8 感知与 CJK Bi-gram 分词器
│   └── vrag.hpp            # 混合检索核心引擎与 RRF 融合调度
├── src/
│   ├── bench_search.cpp    # 多轮吞吐量与分位延迟 (P50/P95/P99) 压测工具
│   └── cli_search.cpp      # 交互式命令行检索客户端
├── tests/
│   ├── eval_accuracy.cpp   # 专有名词与跨词抽象语义召回率评测套件
│   ├── real_corpus_test.cpp# 真实语料端到端测试
│   ├── simd_test.cpp       # SIMD 算子精度单元测试
│   ├── tokenizer_test.cpp  # 分词器单元测试
│   └── vrag_test.cpp       # 引擎基础功能测试
├── benchmarks/             # Google Benchmark / 算子微基准测试
└── scripts/                # 语料切片生成与 Embedding 导出脚本
```

---

## 快速开始

### 1. 环境依赖
* **编译器**：支持 C++20 的 GCC 11+ 或 Clang 14+
* **构建工具**：CMake 3.16+
* **硬件指令集**：支持 `AVX2` 与 `FMA` 的 x86_64 处理器
* **并行库**：OpenMP

### 2. 编译构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 3. 运行测试与基准压测

```bash
# 1. 运行单元测试
./build/simd_test
./build/tokenizer_test
./build/vrag_test

# 2. 运行 10 万切片 (3000万字) 500 次双路混合检索压测
./build/bench_search bench_30m.bin 500

# 3. 运行专有名词与抽象语义准确度评测
./build/eval_accuracy bench_30m.bin
```

---

## C++ API 调用示例

VRAG Core 采用纯头文件（Header-only）核心设计，可直接集成至现有的 C++ 项目中：

```cpp
#include "vrag.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化引擎：指定向量维度为 1024，预分配 256MB Arena 内存池
    vrag::VragEngine engine(1024, 256ULL * 1024 * 1024);

    // 2. 从二进制快照极速加载语料与向量（自动完成归一化与 INT8 量化）
    if (!engine.load_from_binary("bench_30m.bin")) {
        return -1;
    }

    // 3. 构造查询词与查询向量（实际使用中由 Embedding 模型生成）
    std::string query_text = "冷子興演說榮國府衰敗";
    std::vector<float> query_vec(1024, 0.05f);

    // 4. 执行双路混合检索（返回 Top-5，RRF 平滑常数 k = 60.0）
    auto results = engine.search(query_text, query_vec, 5, 60.0f);

    // 5. 遍历输出结果
    for (const auto& hit : results) {
        std::cout << "[DocID: " << hit.doc_id << "] "
                  << "RRF Score: " << hit.score << " | "
                  << hit.text << "\n";
    }

    return 0;
}
```

---

## 二进制语料格式规范 (`*.bin`)

`load_from_binary` 读取的紧凑二进制文件布局如下（小端序）：

| 字段 | 类型 | 字节数 | 说明 |
| :--- | :--- | :--- | :--- |
| `doc_count` | `uint32_t` | 4 B | 文档切片总数 $N$ |
| `dimension` | `uint32_t` | 4 B | 向量维度 $D$（如 1024） |
| **循环 $N$ 次** | | | **每个切片的数据块：** |
| ├─ `doc_id` | `uint32_t` | 4 B | 文档唯一 ID |
| ├─ `text_len` | `uint32_t` | 4 B | UTF-8 文本字节长度 $L$ |
| ├─ `text_data` | `char[L]` | $L$ B | UTF-8 原始文本字节流 |
| └─ `vector_data`| `float[D]` | $4 \times D$ B | 原始 FP32 嵌入向量（载入时自动量化为 INT8） |
