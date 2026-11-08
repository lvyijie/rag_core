#include <iostream>
#include <cassert>
#include "../include/inverted_index.hpp"

using namespace vrag;

int main() {
    InvertedIndex index;

    // 灌入几篇测试语料（包含中文、英文、数字及标点）
    index.add_document(0, "三门峡位于河南省西部，是一座美丽的沿黄城市。");
    index.add_document(1, "仓鼠是一种非常可爱的小动物，平时喜欢吃瓜子。");
    index.add_document(2, "三门峡的大天鹅每年冬天都会从西伯利亚迁徙而来，吸引大量游客。");
    index.add_document(3, "计算机专业学生通常需要深入学习操作系统与体系结构。");
    index.add_document(4, "SimdVector算子使用AVX2与_mm256_fmadd_ps指令，支持32字节对齐加载。");

    std::cout << "Indexed documents    : " << index.total_docs() << '\n';
    std::cout << "Unique terms in index: " << index.vocab_size() << '\n';

    // 测试 1：搜索 "三门峡大天鹅"
    auto hits1 = index.search_bm25("三门峡大天鹅", 5);
    std::cout << "\nQuery: '三门峡大天鹅'\n";
    for (const auto& hit : hits1) {
        std::cout << "  DocID: " << hit.doc_id << ", BM25 Score: " << hit.score << '\n';
    }

    if (hits1.empty() || hits1[0].doc_id != 2) {
        std::cerr << "[FAIL] Doc 2 should rank highest for '三门峡大天鹅'!\n";
        return 1;
    }

    // 测试 2：搜索 "可爱小仓鼠"
    auto hits2 = index.search_bm25("可爱小仓鼠", 5);
    std::cout << "\nQuery: '可爱小仓鼠'\n";
    for (const auto& hit : hits2) {
        std::cout << "  DocID: " << hit.doc_id << ", BM25 Score: " << hit.score << '\n';
    }

    if (hits2.empty() || hits2[0].doc_id != 1) {
        std::cerr << "[FAIL] Doc 1 should rank highest for '可爱小仓鼠'!\n";
        return 1;
    }

    // 测试 3：搜索英文大小写不敏感、数字与代码标识符
    auto hits3 = index.search_bm25("avx2 _mm256_fmadd_ps 32", 5);
    std::cout << "\nQuery: 'avx2 _mm256_fmadd_ps 32'\n";
    for (const auto& hit : hits3) {
        std::cout << "  DocID: " << hit.doc_id << ", BM25 Score: " << hit.score << '\n';
    }

    if (hits3.empty() || hits3[0].doc_id != 4) {
        std::cerr << "[FAIL] Doc 4 should rank highest for 'avx2 _mm256_fmadd_ps 32'!\n";
        return 1;
    }

    std::cout << "\n[PASS] InvertedIndex BM25 retrieval tests passed successfully!\n";
    return 0;
}
