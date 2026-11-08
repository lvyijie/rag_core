<div align="center">

# vrag-core

**High-Performance, Header-Only Hybrid Search Engine for C++20**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?style=flat-square&logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![SIMD](https://img.shields.io/badge/SIMD-AVX--VNNI%20%7C%20AVX2%20%7C%20Scalar-0071C5.svg?style=flat-square&logo=intel)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
[![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-3FB950.svg?style=flat-square)](https://www.openmp.org/)
[![Header-Only](https://img.shields.io/badge/Library-Header--Only-blueviolet.svg?style=flat-square)](#quick-start)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[架构设计](#architecture) •
[快速集成](#quick-start) •
[API 参考](#api-reference) •
[命令行与工具链](#cli--tooling) •
[性能基准](#benchmarks) •
[二进制协议](#binary-format)

</div>

---

**vrag-core** 是一个零外部依赖、纯头文件（Header-Only）的嵌入式 C++20 混合检索库。它将 **BM25 稀疏倒排索引** 与 **SQ8 标量量化稠密向量检索** 融合于同一内存空间，并通过 **Reciprocal Rank Fusion (RRF)** 输出重排结果，专为端侧 AI Agent、本地 RAG 引擎与高吞吐低延迟服务设计。

## Highlights

- **Pure Header-Only & Zero Dependencies** — 仅依赖 C++20 标准库与 OpenMP，直接包含头文件即可接入现有工程。
- **Hybrid Retrieval (BM25 + Dense SQ8 + RRF)** — 单次调用同时执行稀疏词项匹配与稠密语义扫描，支持无向量输入时零开销退化为纯 BM25 检索。
- **Thread-Safe Zero-Allocation Query Path** — `search()` 为无锁 `const` 线程安全方法，内部基于 `thread_local` 平坦工作区与局部小顶堆（Min-Heap）实现多线程并发只读查询零堆分配、零数据竞争。
- **Full Document Lifecycle (Upsert & Tombstone Delete)** — 内部采用稠密槽位映射（Dense Slot Mapping），支持任意稀疏 `uint32_t` 文档 ID、幂等覆盖更新（Upsert）与 $O(1)$ 墓碑软删除（`remove_document`）。
- **Hardware-Accelerated SIMD with Portable Fallback** — 内置 `AVX-VNNI` (`vpdpbusd`) 与 `AVX2 + FMA` 混合精度点积算子（对称 INT8 量化节省 **75%** 向量内存），并在非 AVX2 平台（如 ARM64）自动回退至 4 路展开标量流水线。
- **Zero-Copy Multilingual Tokenizer** — 基于 UTF-8 状态机与编译期查表，同时支持 CJK 二元分词（Bi-gram）与大小写不敏感的 ASCII 字母数字词元（`[a-zA-Z0-9]+`），单核吞吐量达 **3.3 GB/s**。
- **Instant Pre-Quantized Index Snapshot** — 支持将已量化的 `int8` 连续内存池与扁平倒排表直接序列化落盘（`save_index` / `load_index`），冷启动无需重复分词与量化。

---

## Quick Start

### 1. 最小代码示例

```cpp
#include <vrag.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化引擎 (维度: 1024, 初始 Arena 容量: 64 MB，超出时自动按 2 倍平滑扩容)
    vrag::VragEngine engine(/*dimension=*/1024, /*arena_capacity=*/64ULL * 1024 * 1024);

    // 2. 插入或更新文档 (支持任意 uint32_t doc_id 与幂等覆盖更新)
    std::vector<float> vec_a(1024, 0.02f);
    std::vector<float> vec_b(1024, 0.05f);
    engine.add_document(1001, "AVX2 intrinsics accelerate int8 dot-product kernels.", vec_a);
    engine.add_document(1002, "BM25 and dense vector retrieval fused via Reciprocal Rank Fusion.", vec_b);

    // 3. 执行线程安全的双路混合检索 (Top-5, RRF k = 60.0)
    std::vector<float> query_vec(1024, 0.02f);
    auto hits = engine.search("AVX2 dot-product", query_vec, /*top_k=*/5, /*rrf_k=*/60.0f);

    for (const auto& hit : hits) {
        std::cout << "doc_id=" << hit.doc_id
                  << " score=" << hit.score
                  << " text=\"" << hit.text << "\"\n";
    }

    // 4. 持久化预量化索引快照 (下次启动可通过 engine.load_index("snapshot.vidx") 瞬时挂载)
    engine.save_index("snapshot.vidx");
    return 0;
}
```

### 2. CMake 集成

将本仓库作为子目录引入，或直接将 `include/` 目录添加至头文件路径：

```cmake
add_subdirectory(rag_core)
target_link_libraries(your_target PRIVATE rag_core)
```

手动编译单文件时开启 C++20、OpenMP 与本地指令集优化：

```bash
g++ -std=c++20 -O3 -march=native -fopenmp main.cpp -I./include -o app
```

### 3. 编译本项目工具与测试套件

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

---

## Architecture

```text
                        ┌───► vrag::Tokenizer ───────► vrag::InvertedIndex (BM25) ──► Sparse Top-N ─┐
                        │     (UTF-8 Bi-gram / Alnum)  (Thread-Local Score Buffer)                  │
Query (Text + FP32 Vec)─┤                                                                           ├─► RRF Fusion ─► Top-K
                        │                                                                           │
                        └───► vrag::SimdInt8 ────────► OpenMP Dynamic Full Scan ────► Dense Top-N ──┘
                              (L2 Norm + SQ8 Quant)    (64B Arena + AVX-VNNI/AVX2)
```

| 模块头文件 | 核心职责 |
| :--- | :--- |
| `include/vrag.hpp` | `vrag::VragEngine` 主入口，管理内部槽位映射、双路并发调度、RRF 融合与索引快照序列化 |
| `include/inverted_index.hpp` | `vrag::InvertedIndex`，基于扁平化倒排链、IDF/TF 预计算表与 `thread_local` 零分配工作区的 BM25 检索引擎 |
| `include/simd_int8.hpp` | `vrag::SimdInt8`，提供对称 `int8` 标量量化与 `AVX-VNNI` / `AVX2` / 标量回退点积算子 |
| `include/simd_vector.hpp` | `vrag::SimdVector`，提供 `FP32` AVX2+FMA 向量点积与原位 L2 归一化 |
| `include/tokenizer.hpp` | `vrag::Tokenizer`，基于流式迭代器的零拷贝分词器与大小写不敏感 `FNV-1a` 64-bit 哈希 |
| `include/arena.hpp` | `vrag::Arena`，跨平台（POSIX / Win32）64 字节对齐且支持几何级数自动扩容的连续内存池 |

---

## API Reference

### `vrag::VragEngine`

```cpp
namespace vrag {

struct HitResult {
    uint32_t doc_id;
    float score;
    std::string text;
};

class VragEngine {
public:
    // 构造检索实例 (arena_capacity 不足时自动按 2 倍扩容)
    explicit VragEngine(size_t dimension = 768,
                        size_t arena_capacity = 256ULL * 1024 * 1024);

    // 插入或更新单篇文档 (Upsert：若 doc_id 已存在则自动替换旧版本)
    bool add_document(uint32_t doc_id,
                      std::string_view text,
                      std::span<const float> raw_vector);

    // O(1) 墓碑软删除指定文档
    bool remove_document(uint32_t doc_id);
    bool contains_document(uint32_t doc_id) const noexcept;

    // 线程安全只读混合检索 (可由多个工作线程并发调用)
    // 若 query_vector 全为 0 或为空，自动退化为纯 BM25 稀疏检索
    std::vector<HitResult> search(std::string_view query_text,
                                  std::span<const float> query_vector,
                                  size_t top_k = 5,
                                  float k = 60.0f) const;

    // 保存/加载预量化编译索引快照 (*.vidx，免分词与量化极速挂载)
    bool save_index(const std::string& filepath) const;
    bool load_index(const std::string& filepath);

    // 从原始语料包 (*.bin: UTF-8 文本 + FP32 向量) 批量导入并构建索引
    bool load_from_binary(const std::string& filepath);

    size_t document_count() const noexcept;
    size_t dimension() const noexcept;
    size_t memory_used() const noexcept;
};

} // namespace vrag
```

---

## CLI & Tooling

仓库内置了从原始文本切块、向量化打包到交互式查询与基准压测的完整工具链：

### 1. 构建二进制语料包 (`scripts/build_corpus.py`)

支持对单个文件或目录递归切块，并通过 OpenAI 兼容的 Embedding 接口（默认配置为 SiliconFlow `BAAI/bge-large-zh-v1.5`）并发生成 `.bin` 语料包：

```bash
export SILICONFLOW_API_KEY="sk-..."
# 用法: python scripts/build_corpus.py <input_path> <output_bin> [max_chunks=0] [chunk_size=500]
python scripts/build_corpus.py ./docs corpus.bin
```

### 2. 生成离线压测数据集 (`scripts/generate_benchmark_corpus.py`)

无需调用外部 API，基于本地文本与正交归一化向量池快速生成指定规模的基准测试包（生成 10 万条 1024 维记录耗时 `< 0.5s`）：

```bash
python scripts/generate_benchmark_corpus.py ./docs bench_100k.bin 100000
```

### 3. 交互式命令行检索 (`cli_search`)

```bash
./build/cli_search corpus.bin
```
- 当环境变量 `SILICONFLOW_API_KEY` 已设置时，自动为输入 Query 请求实时 Embedding 并执行 **BM25 + Dense 双路混合检索**。
- 未设置环境变量时，自动以 **纯 BM25 模式** 运行。

### 4. 吞吐量与分位延迟压测 (`bench_search`)

```bash
# 用法: ./build/bench_search <bin_path> [num_queries=500]
./build/bench_search bench_100k.bin 500
```

---

## Benchmarks

**测试环境**
- **CPU**: Intel Core i7-12700H (14C/20T, AVX2 + FMA + AVX-VNNI)
- **Compiler**: GCC 16.1.0 (`-O3 -march=native -fopenmp`)
- **OS**: Windows 11 x86_64 / Dual-Channel DDR5

### 核心算子微基准 (Microbenchmarks)

| Target | Workload | Latency (Avg / P50) | Throughput |
| :--- | :--- | :--- | :--- |
| `tokenizer_bench` | 1 MiB UTF-8 Mixed Text (CJK + ASCII Alnum) | `0.303 ms` / `0.294 ms` | **3,303.0 MiB/s** (`1.15B tokens/s`) |
| `simd_bench` | 10,000 × 768d FP32 Dot Product (Single Thread) | `1.22 ms` (`122.1 ns/vec`) | **8.19M vec/s** (`12.58 GFLOPS`) |

### 端到端混合检索延迟与吞吐 (End-to-End Hybrid Search, Top-5)

| Dataset Scale | Dimension | Vector RAM (SQ8) | Index Build / Load | Avg Latency | P50 | P95 | P99 | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000 docs** | 768d | 7.3 MB | `23.6 ms` (`423k docs/s`) | **0.186 ms** | `0.177 ms` | `0.244 ms` | `0.307 ms` | **5,364.7 QPS** |
| **17,557 docs** | 1024d | 17.1 MB | `0.43 s` (from `.bin`) | **0.271 ms** | `0.259 ms` | `0.349 ms` | `0.482 ms` | **3,689.6 QPS** |
| **100,000 docs** | 1024d | 97.6 MB | `0.84 s` (from `.bin`) | **2.47 ms** | `2.41 ms` | `2.99 ms` | `4.10 ms` | **404.8 QPS** |

---

## Binary Format

`load_from_binary()` 读取的小端序（Little-Endian）原始语料流结构如下，可由任意语言离线生成：

```text
+-------------------+-------------------+
| doc_count (u32)   | dimension (u32)   |   <-- 8-byte File Header
+-------------------+-------------------+
| doc_id (u32)      | text_len (u32)    |   ─┐
+-------------------+-------------------+    │
| text_utf8_bytes (u8 × text_len)       |    ├─ Repeated `doc_count` times
+---------------------------------------+    │
| vector_fp32 (f32 × dimension)         |   ─┘
+---------------------------------------+
```

| Field | Type | Size (Bytes) | Description |
| :--- | :--- | :--- | :--- |
| `doc_count` | `uint32_t` | `4` | 文档切片总数 $N$ |
| `dimension` | `uint32_t` | `4` | 向量维度 $D$ |
| `doc_id` | `uint32_t` | `4` | 文档外部唯一标识符 |
| `text_len` | `uint32_t` | `4` | UTF-8 文本字节长度 $L$ |
| `text_utf8_bytes` | `char[L]` | $L$ | 原始 UTF-8 字节流（不含末尾 `\0`） |
| `vector_fp32` | `float[D]` | $4 \times D$ | 原始 `float32` 向量（载入时由引擎执行归一化与 INT8 量化） |

---

## License

Released under the [MIT License](LICENSE).
