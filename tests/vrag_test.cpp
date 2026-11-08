#include <iomanip>
#include <iostream>
#include <vector>
#include "../include/vrag.hpp"

using namespace vrag;

int main() {
    constexpr std::size_t Dim = 8;
    VragEngine engine(Dim);

    // Doc 0: 讨论天鹅，语义偏向鸟类
    std::vector<float> vec0 = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(0, "每年冬天，大天鹅都会飞抵三门峡黄河湿地越冬。", vec0);

    // Doc 1: 讨论操作系统，语义偏向计算机
    std::vector<float> vec1 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(1, "现代操作系统依赖虚拟内存与缺页中断机制。", vec1);

    // Doc 2: 讨论仓鼠，语义偏向动物
    std::vector<float> vec2 = {0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine.add_document(2, "小仓鼠是一种体型很小的啮齿动物，爱吃坚果。", vec2);

    std::cout << "Engine loaded with " << engine.total_docs() << " documents.\n";

    // Query: 字面命中“大天鹅”，向量侧也偏向鸟类
    std::vector<float> q_vec = {0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto hits = engine.search("大天鹅", q_vec, 3);

    std::cout << "\nRRF Hybrid Search Results for '大天鹅':\n";
    std::cout << std::fixed << std::setprecision(5);
    for (const auto& hit : hits) {
        std::cout << "  DocID: " << hit.doc_id 
                  << " | RRF Score: " << hit.score 
                  << " | Text: " << hit.text << '\n';
    }

    if (hits.empty() || hits[0].doc_id != 0) {
        std::cerr << "[FAIL] Doc 0 should be top result with highest RRF score!\n";
        return 1;
    }

    std::cout << "\n[PASS] VragEngine RRF hybrid search test passed successfully!\n";
    return 0;
}
