# `rag_core` 1.0 演进路线图 (Engineering Roadmap)

> **核心定位**：面向个人本地知识库与 AI Agent 的轻量级、嵌入式（In-Process）C++20 混合检索核心库（对标 SQLite / DuckDB / USearch 的进程内无服务架构，专注 $10^3 \sim 10^5$ 文档规模下的亚毫秒级精确检索）。  
> **核心工程原则**：**先保证数学与边界正确性（P0） $\to$ 再完善统一 API 与持久化可靠性（P1/P2） $\to$ 健全并发与跨规模基准（P3/P4） $\to$ 落地 Python 生态绑定与中文评测（P5/P6） $\to$ 按需扩展高级索引（P7）。**

---

## 对原路线建议的 3 处校准与修正 (Technical Calibrations)

在结合当前仓库已有代码（Commit `da515e6`）审视路线建议后，我们对部分设计细节做了如下工程校准：

1. **已实现能力盘点与 P2/P3 目标聚焦**
   - **已完成部分**：仓库目前已具备基础的预量化快照落盘/加载（[`VragEngine::save_index`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp) / [`load_index`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp)）、增量幂等更新（`add_document` Upsert）、墓碑删除（`remove_document`）以及基于 32 槽位无锁工作区池（`ScopedWorkspace`）的多线程只读并发 `search() const`。
   - **仍需补齐的真正痛点**：
     - **P2（持久化可靠性）**：目前缺少**文件尾部校验和（CRC32/FNV-1a Checksum）**（无法检测断电截断或位翻转损坏）、**原子写入（Write-to-temp + Rename）**以及墓碑堆积后的**空间回收（`compact()`）**。
     - **P3（读写并发安全）**：目前支持“多线程同时只读查询”，但尚未在引擎层提供读写锁（`std::shared_mutex`）或快照切换机制来保护“一边后台增量 `add_document`、一边前台多线程 `search`”的并发读写场景。
2. **`SearchResult` 字符串所有权修正（避免悬垂引用）**
   - 原建议示意结构体中使用了 `std::string_view content;`。在支持动态扩容（`std::vector<std::string> doc_texts_` 重新分配）、文档删除（`remove_document` 释放文本）以及后续 Python 绑定（跨语言生命周期）的库中，对外返回 `std::string_view` 极易引发 Use-After-Free 悬垂指针。
   - **修正方案**：检索仅对最终胜出的 Top-K（通常 $K \le 20$）构造返回结果，深拷贝几 KB 文本耗时不足 `1 μs`，因此主接口统一返回拥有所有权的 `std::string text`（同时暴露 `bm25_score`、`dense_score`、`bm25_rank`、`dense_rank` 等明细字段）。
3. **将 `Weighted RRF` 提前并入 P1 核心 API**
   - 此前在 `BEIR/SciFact` 真实向量消融实验中已观察到：当单路 Dense（`NDCG@10 = 0.7246`）显著强于单路 BM25（`0.6622`）时，等权（1:1）RRF（`0.7160`）会被弱路轻微稀释。支持 `bm25_weight` 与 `dense_weight` 的加权融合（Weighted RRF）仅需改动 10 行融合打分逻辑，却能直接提升混合检索上限，因此统一收归 **P1（核心检索 API）**，不再拆散到后期。

---

## 阶段划分与验收标准

### P0：基础正确性与测试体系（最高优先级 | ✅ 已完成）

**目标**：确保每一个算子、每一条检索公式、每一种异常输入下的行为 100% 确定且可信，并量化 `SQ8` 相比 `FP32` 的真实精度损失。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P0.1 BM25 数学与边界测试** | 验证 TF 饱和单调性、IDF 单调性、长短文档长度归一化、`tf=1` 快速路径与 `tf>1` 一致性、删除文档后活跃 `df` 精确性、高频词剪枝边界 | ✅ 已通过 ([`tests/p0_correctness_test.cpp`](file:///c:/Users/qq150/source/repos/rag_core/tests/p0_correctness_test.cpp))：解析解对比误差 $< 10^{-5}$，修复删除文档后活跃 `df` 统计，确保删除后得分与全新建库 bit-for-bit 一致 |
| **P0.2 Dense & SQ8 算子测试** | 验证向量 L2 归一化、零向量保护、非 64 整数倍维度（`dim = 1, 7, 31..768` 共 13 种奇偶维度）尾部处理、SIMD 与标量参考实现一致性 | ✅ 已通过：13 种维度下 VNNI / AVX2 整数点积与标量参考 100% 一致，余弦近似误差 $< 0.015$ |
| **P0.3 Hybrid & RRF 逻辑测试** | 验证 `1 / (k + rank)` 数学公式、双路共现文档分数累加与去重、单路缺失时退化为单路排序、候选截断边界 | ✅ 已通过：双路共现去重及单路退化行为 100% 符合解析公式 |
| **P0.4 Tokenizer 边界与乱码防护** | 中文 Bi-gram、单字边界、英文大小写折叠、数字与代码符号、全角/半角标点隔离、**截断/非法 UTF-8 字节序列安全跳过** | ✅ 已通过：加固 [`include/tokenizer.hpp`](file:///c:/Users/qq150/source/repos/rag_core/include/tokenizer.hpp) UTF-8 延续字节校验，损坏字节绝不吞噬相邻合法 ASCII/汉字 |
| **P0.5 异常与防御性输入测试** | 空语料查询、空 Query 字符串、全零 Query 向量、含 `NaN`/`Inf` 向量、维度不匹配、`top_k = 0`、非法 `k1`/`b`/`rrf_k` 参数、截断快照文件 | ✅ 已通过：26 项正确性与异常断言全部通过 (`6/6 CTest passed`) |
| **P0.6 FP32 vs SQ8 真实向量精度对比实验** | 在 `BEIR/NFCorpus` 与 `BEIR/SciFact` 同一批 `BAAI/bge-small-en-v1.5` (384d) 向量上，同场对比 **FP32 精确点积** 与 **SQ8 量化点积** 的 `NDCG@10`、`MRR@10`、`Recall@10/100` 及 `Top-10 重合率` | ✅ 已完成 ([`src/beir_bench.cpp`](file:///c:/Users/qq150/source/repos/rag_core/src/beir_bench.cpp))：`NFCorpus` Top-10 重合率 **`97.34%`**（$\Delta\text{NDCG@10} = -0.0010$），`SciFact` Top-10 重合率 **`97.20%`**（$\Delta\text{NDCG@10} = +0.0044$，`Recall@100` 零损失），内存降低 **4.0 倍**，扫描提速 **1.67x ~ 1.90x** |

---

### P1：完善核心检索能力与统一 API（高优先级 | ✅ 已完成）

**目标**：将 `VragEngine` 从实验原型接口升级为结构清晰、可配置、可观测的现代 C++ 检索接口。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P1.1 统一结果结构 `SearchResult`** | 包含 `doc_id`、`score`（最终融合分或单路分）、`bm25_score`、`dense_score`、`bm25_rank`、`dense_rank`、`text`、`metadata` | ✅ 已完成 ([`include/vrag.hpp`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp))：支持一次检索同时观测双路原始得分与排名，并提供 `search_bm25` / `search_dense` 单路便捷方法 |
| **P1.2 统一查询配置 `SearchOptions`** | 支持 `top_k`、`mode` (`Hybrid`/`BM25`/`Dense`)、`rrf_k`、`bm25_weight`、`dense_weight`、`candidate_limit`、`min_score`、`bm25_k1`、`bm25_b` | ✅ 已完成：100% 向后兼容旧版简写签名，同时支持全参数细粒度配置 |
| **P1.3 Weighted RRF（加权融合）** | 实现 $\text{Score}(d) = \frac{w_{\text{bm25}}}{k + r_{\text{bm25}}(d)} + \frac{w_{\text{dense}}}{k + r_{\text{dense}}(d)}$，支持单路权重置 0 自动退化为纯 BM25 或纯 Dense | ✅ 已完成并验证：在 `BEIR/SciFact` 上使用 `0.3:1 W-RRF` 后，`NDCG@10` 提升至 **`0.7264`**（超越单路 Dense `0.7246` 与 1:1 RRF `0.7160`），`MRR@10` 达 **`0.6916`**，`Recall@10` 达 **`0.8546`** |
| **P1.4 Metadata Filter（元数据过滤）** | 支持为文档绑定轻量级属性字符串与 `FilterPredicate` 回调，在主线程预评估活跃槽位掩码并在 BM25 与 Dense 扫描中前置过滤（Pre-filtering） | ✅ 已完成：主线程预计算掩码避免在 20 个 OpenMP 线程中并发调用非线程安全回调，且 `VRAGIDX2` 快照完整持久化 `metadata` |
| **P1.5 Reranker 扩展接口** | 提供 `RerankCallback`（接收 Query 与初步召回的 Top-N 候选列表，返回重排分数并截断至 `top_k`） | ✅ 已完成并通过 [`tests/p1_api_test.cpp`](file:///c:/Users/qq150/source/repos/rag_core/tests/p1_api_test.cpp) 全部 12 项断言验证 (`7/7 CTest passed`) |

---

### P2：索引持久化加固与空间回收（高优先级 | ✅ 已完成）

**目标**：让本地知识库在反复增删改、意外断电或文件损坏场景下依然安全可靠。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P2.1 索引格式版本与完整性校验** | 升级 `VRAGIDX3` 格式，通过流式 [`HashingStreamBuf`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp#L39-L72) 单遍计算 64 位 FNV-1a 校验和并写入 `VRAGEND3` 尾部；`load_index()` 采用事务性加载（先校验并加载至临时状态，全部成功后再提交，失败时原实例状态 100% 完好） | ✅ 已完成 ([`tests/p2_persistence_test.cpp`](file:///c:/Users/qq150/source/repos/rag_core/tests/p2_persistence_test.cpp))：单比特翻转、文件截断 100% 识别拦截，且失败时内存中原数据零污染 |
| **P2.2 原子保存（Atomic Save）** | `save_index(path)` 先写入 `path.tmp` 并刷盘验证，成功后调用 Win32 `MoveFileExW(MOVEFILE_REPLACE_EXISTING \| MOVEFILE_WRITE_THROUGH)` / POSIX `rename` 原子替换目标文件 | ✅ 已完成：保存中途失败绝不破坏磁盘旧索引，成功后无残留 `.tmp` 文件 |
| **P2.3 墓碑空间压缩 (`compact()`)** | 提供 [`VragEngine::compact()`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp#L897-L953) 与 [`InvertedIndex::compact()`](file:///c:/Users/qq150/source/repos/rag_core/include/inverted_index.hpp#L450-L492)，免重新量化直接拷贝活跃 INT8 槽位并重映射倒排表，物理清除死槽位与空词项 | ✅ 已完成：测试删除/覆盖 50/70 槽位后 `compact()` 精确回收 `71.4%` 向量池内存并清理死词项，压缩前后检索结果与子分数 bit-for-bit 一致 (`8/8 CTest passed`) |

---

### P3：读写并发与线程安全体系（中优先级 | ✅ 已完成）

**目标**：在个人 Agent 多线程后台写入新笔记、前台并发检索的场景下保证零数据竞争（Data Race Free）。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P3.1 读写锁同步 (`std::shared_mutex`)** | 查询操作获取共享读锁（`std::shared_lock`），写操作（`add_document` / `add_documents_batch` / `remove_document` / `compact` / `load_index`）获取独占写锁（`std::unique_lock`） | ✅ 已完成 ([`tests/p3_concurrency_test.cpp`](file:///c:/Users/qq150/source/repos/rag_core/tests/p3_concurrency_test.cpp))：6 读线程 + 2 写线程混合高频执行 `search` + `add` + `remove` + `compact` 零竞争通过；16 线程并发读取吞吐在 `NFCorpus` 上达 `107,562 QPS`（`10.47x` 加速比），在 `SciFact` 上达 `66,596 QPS`（`9.98x` 加速比） |
| **P3.2 批量并行建库 (`add_documents_batch`)** | 支持一次性传入一批文档与向量，利用 OpenMP 并行完成向量归一化与 SQ8 量化 | ✅ 已完成：OpenMP 并行归一化与量化结果与单条顺序建库 bit-for-bit 完全一致 (`9/9 CTest passed`) |

---

### P4：系统化性能评测与多规模 Benchmark（中优先级 | ✅ 已完成）

**目标**：补齐不同规模、不同线程数、不同指令集下的可复现性能曲线与严谨统计口径。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P4.1 多轮中位数计时与量化误差分布** | 每条 Query 重复测量 5 轮取中位数消除调度抖动；输出 `Overlap@10`（均值与 P10）及余弦绝对误差（`Mean / P95 / Max`） | ✅ 已完成 ([`src/beir_bench.cpp`](file:///c:/Users/qq150/source/repos/rag_core/src/beir_bench.cpp))：余弦绝对误差均值仅 `0.0009`（Max `0.0041`），`Overlap@10` P10 达 `90.00%` |
| **P4.2 单线程 vs OpenMP 线程策略对照与自适应阈值** | 支持 `SearchOptions::num_threads` 显式控制，同场实测 `1-Thread` vs `OpenMP`；验证 L3 Cache 驻留边界并调整自适应阈值为 `16,384` 槽位 | ✅ 已完成：小语料（`SciFact` 5.1K）单线程比 OpenMP 快 `46%`（`0.079 ms` vs `0.131 ms`）；中大语料（`FiQA` 57.6K）OpenMP 比单线程快 `2.51x`（`1.085 ms` vs `2.718 ms`） |
| **P4.3 全量索引内存明细 (`IndexMemoryStats`)** | 明确区分向量池（`Arena` 4.0x 压缩）、向量元数据（`VecMeta`）、倒排索引（`InvIdx`）与原文字符串（`DocTexts`）的内存占比 | ✅ 已完成：新增 [`VragEngine::memory_stats()`](file:///c:/Users/qq150/source/repos/rag_core/include/vrag.hpp#L1060-L1077) 并在 `README.md` 中完整披露各部分内存明细 |

---

### P5：Python 绑定与生态集成（中优先级 | ✅ 已完成）

**目标**：让 Python 编写的本地 AI Agent / LangChain / LlamaIndex 脚本能够直接 `import rag_core` 调用 C++ 引擎。

| 子任务 | 具体内容 | 验收状态与实测结果 (Acceptance Criteria & Results) |
| :--- | :--- | :--- |
| **P5.1 稳定 C ABI 与零拷贝 Python SDK** | 提供跨编译器 ABI 兼容的 C 接口（[`include/rag_core_c.h`](file:///c:/Users/qq150/source/repos/rag_core/include/rag_core_c.h) / [`src/rag_core_c.cpp`](file:///c:/Users/qq150/source/repos/rag_core/src/rag_core_c.cpp)）与 Python SDK（[`python/rag_core/__init__.py`](file:///c:/Users/qq150/source/repos/rag_core/python/rag_core/__init__.py)），封装 `Index` (`VragEngine`)、`SearchMode`、`SearchOptions`、`SearchResult`、`MemoryStats` | ✅ 已完成：支持 1D/2D `np.ndarray(dtype=np.float32)` 原始内存指针零拷贝直传 C++ 引擎，C ABI 调用期间自动释放 Python GIL，支持 `ThreadPoolExecutor` 多线程并发查询 |
| **P5.2 `pyproject.toml`、测试套件与 Agent 示例** | 提供 [`pyproject.toml`](file:///c:/Users/qq150/source/repos/rag_core/pyproject.toml)、Python 完整测试套件（[`tests/p5_python_test.py`](file:///c:/Users/qq150/source/repos/rag_core/tests/p5_python_test.py)）以及本地 AI Agent 混合检索示例（[`examples/python_agent_rag_demo.py`](file:///c:/Users/qq150/source/repos/rag_core/examples/python_agent_rag_demo.py)） | ✅ 已完成：覆盖单条/2D NumPy 批量入库、三路检索、Weighted RRF、Metadata Filter 回调、Python Reranker 钩子、`VRAGIDX3` 持久化、墓碑压缩与 8 线程并发检索，全部断言 100% 通过 |

---

### P6：中文及中英混合检索专项评测（中后置优先级 | 待执行）

**目标**：验证并优化中文与中英混排（代码/技术文档）场景下的真实检索质量。

| 子任务 | 具体内容 | 验收标准 (Acceptance Criteria) |
| :--- | :--- | :--- |
| **P6.1 中文真实向量评测** | 接入开源中文/多语言 Embedding 模型（如 `BAAI/bge-small-zh-v1.5`），在中文标准检索评测集（C-MTEB 子集，如 `T2Retrieval` / `DuRetrieval` 抽样或领域知识库）上运行三路消融实验 | 产出中文场景下 `BM25 (Bi-gram)` vs `Dense SQ8` vs `Hybrid RRF` 的实测报告 |

---

### P7：远期高级特性（明确暂缓，避免过度设计）

> **准则**：在 P0 ~ P6 全部扎实落地前，**严禁提前引入**以下复杂度：
- **HNSW / IVF 近似最近邻索引 (ANN)**：在 10 万文档以内，`AVX-VNNI` 精确扫描仅需 `0.1 ~ 2 ms` 且召回率 100%，无需承受 HNSW 高昂的内存开销、建图耗时与删除退化问题；待突破 20 万+ 文档需求时再作为可选索引后端引入。
- **GPU 算子 / 分布式服务端**：偏离个人本地轻量级进程内检索库定位，暂不考虑。
