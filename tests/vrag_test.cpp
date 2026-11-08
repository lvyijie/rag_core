#include <cstdio>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include "../include/vrag.hpp"

using namespace vrag;

int main() {
    constexpr std::size_t Dim = 8;
    // 使用较小的初始 Arena (4 KB)，同时验证自动扩容
    VragEngine engine(Dim, 4096);

    // Doc 0: 讨论天鹅，语义偏向鸟类
    std::vector<float> vec0 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(0, "每年冬天，大天鹅都会飞抵三门峡黄河湿地越冬。", vec0);

    // Doc 1: 讨论操作系统，语义偏向计算机
    std::vector<float> vec1 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(1, "现代操作系统依赖虚拟内存与缺页中断机制。", vec1);

    // Doc 2: 讨论仓鼠，语义偏向动物
    std::vector<float> vec2 = {0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(2, "小仓鼠是一种体型很小的啮齿动物，爱吃坚果。", vec2);

    // 1. 基础混合检索验证
    std::vector<float> q_vec = {0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto hits = engine.search("大天鹅", q_vec, 3);

    if (hits.empty() || hits[0].doc_id != 0) {
        std::cerr << "[FAIL] Doc 0 should be top result with highest RRF score!\n";
        return 1;
    }
    std::cout << "[PASS] 1. Basic hybrid search passed.\n";

    // 2. 超大稀疏 DocID 插入 + Upsert 更新 + Remove 删除验证
    constexpr std::uint32_t HugeDocId = 900000000u;
    std::vector<float> vec_huge = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(HugeDocId, "舊版量子計算機文檔", vec_huge);
    if (engine.document_count() != 4 || !engine.contains_document(HugeDocId)) {
        std::cerr << "[FAIL] Huge sparse DocID insertion failed!\n";
        return 1;
    }

    // 幂等覆盖更新同一 HugeDocId
    engine.add_document(HugeDocId, "新版超导量子比特纠错协议", vec_huge);
    if (engine.document_count() != 4) {
        std::cerr << "[FAIL] Document upsert should keep document_count() == 4!\n";
        return 1;
    }
    auto upsert_hits = engine.search("超导量子比特", vec_huge, 2);
    if (upsert_hits.empty() || upsert_hits[0].doc_id != HugeDocId ||
        upsert_hits[0].text.find("新版") == std::string::npos) {
        std::cerr << "[FAIL] Document upsert did not update content properly!\n";
        return 1;
    }

    // 删除该文档并验证其不再被检索召回
    if (!engine.remove_document(HugeDocId) || engine.contains_document(HugeDocId) ||
        engine.document_count() != 3) {
        std::cerr << "[FAIL] remove_document failed!\n";
        return 1;
    }
    auto after_del = engine.search("超导量子比特", vec_huge, 3);
    for (const auto& h : after_del) {
        if (h.doc_id == HugeDocId) {
            std::cerr << "[FAIL] Deleted document still appeared in search results!\n";
            return 1;
        }
    }
    std::cout << "[PASS] 2. Sparse DocID, Upsert, and Tombstone Removal passed.\n";

    // 3. 预量化索引序列化落盘与极速加载验证 (save_index / load_index)
    const std::string idx_path = "test_snapshot.vidx";
    if (!engine.save_index(idx_path)) {
        std::cerr << "[FAIL] save_index failed!\n";
        return 1;
    }

    VragEngine loaded_engine(Dim, 4096);
    if (!loaded_engine.load_index(idx_path)) {
        std::cerr << "[FAIL] load_index failed!\n";
        return 1;
    }
    std::remove(idx_path.c_str());

    auto loaded_hits = loaded_engine.search("大天鹅", q_vec, 3);
    if (loaded_hits.size() != hits.size() || loaded_hits[0].doc_id != hits[0].doc_id) {
        std::cerr << "[FAIL] Loaded index produced different results!\n";
        return 1;
    }
    std::cout << "[PASS] 3. Pre-quantized index save_index / load_index verified.\n";

    // 4. 多线程并发只读查询安全性压测 (const VragEngine&)
    const VragEngine& const_engine = loaded_engine;
    std::atomic<bool> thread_ok{true};
    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&const_engine, &q_vec, &thread_ok]() {
            for (int iter = 0; iter < 200; ++iter) {
                auto res = const_engine.search("大天鹅黄河湿地", q_vec, 2);
                if (res.empty() || res[0].doc_id != 0) {
                    thread_ok.store(false, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
    if (!thread_ok.load()) {
        std::cerr << "[FAIL] Concurrent multi-threaded search failed!\n";
        return 1;
    }
    std::cout << "[PASS] 4. Concurrent 8-thread const search stress test passed.\n";

    return 0;
}
