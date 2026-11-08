#include <iomanip>
#include <iostream>
#include <vector>
#include "../include/vrag.hpp"

using namespace vrag;

int main() {
    constexpr std::size_t Dim = 8;
    VragEngine engine(Dim);

    std::cout << "====================================================\n";
    std::cout << " VragEngine Real Codebase Knowledge Corpus Test     \n";
    std::cout << "====================================================\n";

    // 灌入 5 篇来自当前项目的真实知识切片
    // 向量定义: [0:Memory, 1:SIMD, 2:NLP, 3:IR, 4:Engine, 5:Reserved, 6:Reserved, 7:Reserved]

    // Doc 0: Arena 内存池
    std::vector<float> vec0 = {0.95f, 0.10f, 0.00f, 0.00f, 0.20f, 0.0f, 0.0f, 0.0f};
    engine.add_document(0,
        "Arena内存池设计：通过std::aligned_alloc或posix_memalign强制分配32字节对齐的连续大块物理内存。"
        "采用线性指针碰撞推进分配，完全杜绝内存碎片与堆锁竞争，为AVX2向量加载提供物理级硬件保障。",
        vec0);

    // Doc 1: SimdVector 算子
    std::vector<float> vec1 = {0.20f, 0.98f, 0.00f, 0.00f, 0.30f, 0.0f, 0.0f, 0.0f};
    engine.add_document(1,
        "SimdVector算子实现：使用x86 AVX2和FMA指令集，核心函数dot_product_avx2采用4路循环展开。"
        "利用_mm256_fmadd_ps并发计算，避免流水线数据冒险与延迟停顿，单核完成10k向量扫描仅需0.92ms。",
        vec1);

    // Doc 2: Tokenizer 分词器
    std::vector<float> vec2 = {0.00f, 0.00f, 0.95f, 0.30f, 0.20f, 0.0f, 0.0f, 0.0f};
    engine.add_document(2,
        "Tokenizer零拷贝分词器：针对中文字符流采用双字滑动窗口Bi-gram切分，直接解析UTF-8前缀字节判定汉字。"
        "使用std::string_view实现全局零动态内存分配，配合FNV-1a哈希算法，单核吞吐量超过2.3GiB/s。",
        vec2);

    // Doc 3: InvertedIndex 倒排表
    std::vector<float> vec3 = {0.10f, 0.00f, 0.40f, 0.95f, 0.20f, 0.0f, 0.0f, 0.0f};
    engine.add_document(3,
        "InvertedIndex倒排索引：基于紧凑的PostingEntry结构体，单条记录仅占8字节紧凑排布在连续数组中。"
        "实现Lucene平滑BM25打分，维护全局文档平均长度与文档频率IDF，对高频关键词与罕见专有名词准确加权。",
        vec3);

    // Doc 4: VragEngine 融合
    std::vector<float> vec4 = {0.15f, 0.30f, 0.20f, 0.40f, 0.90f, 0.0f, 0.0f, 0.0f};
    engine.add_document(4,
        "VragEngine顶层门面：将AVX2稠密向量路径与BM25稀疏字面路径合并，采用工业界标准的RRF倒数排名融合算法。"
        "以常数k=60抹平不同打分模型的量纲差异，使用std::priority_queue小顶堆高效率筛选Top-K结果。",
        vec4);

    std::cout << "Indexed " << engine.total_docs() << " architectural documents.\n\n";

    // ---------------------------------------------------------
    // 真实场景测试 1：精准专有名词搜索（稀疏字面通道测试）
    // 用户询问底层指令："_mm256_fmadd_ps循环展开"
    // ---------------------------------------------------------
    {
        std::string query = "循环展开指令";
        std::vector<float> q_vec = {0.1f, 0.9f, 0.0f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f}; // 偏向计算指令

        std::cout << ">>> Query 1: '" << query << "' (期望精准命中 Doc 1 SimdVector)\n";
        auto hits = engine.search(query, q_vec, 3);
        for (std::size_t i = 0; i < hits.size(); ++i) {
            std::cout << "  #" << i + 1 << " [Doc " << hits[i].doc_id << "] RRF: " 
                      << std::fixed << std::setprecision(5) << hits[i].score << '\n'
                      << "     " << hits[i].text.substr(0, 70) << "...\n";
        }
        std::cout << '\n';
    }

    // ---------------------------------------------------------
    // 真实场景测试 2：跨表述概念搜索（语义向量挽救字面未命中）
    // 用户询问："内存碎片的杜绝与对齐"
    // ---------------------------------------------------------
    {
        std::string query = "内存对齐杜绝碎片";
        std::vector<float> q_vec = {0.9f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f}; // 偏向内存架构

        std::cout << ">>> Query 2: '" << query << "' (期望精准命中 Doc 0 Arena)\n";
        auto hits = engine.search(query, q_vec, 3);
        for (std::size_t i = 0; i < hits.size(); ++i) {
            std::cout << "  #" << i + 1 << " [Doc " << hits[i].doc_id << "] RRF: " 
                      << std::fixed << std::setprecision(5) << hits[i].score << '\n'
                      << "     " << hits[i].text.substr(0, 70) << "...\n";
        }
        std::cout << '\n';
    }

    // ---------------------------------------------------------
    // 真实场景测试 3：高层概念提问（双路共识协同测试）
    // 用户询问："如何融合稀疏与稠密两路结果"
    // ---------------------------------------------------------
    {
        std::string query = "融合双路算法";
        std::vector<float> q_vec = {0.1f, 0.2f, 0.1f, 0.3f, 0.85f, 0.0f, 0.0f, 0.0f}; // 偏向上层引擎

        std::cout << ">>> Query 3: '" << query << "' (期望精准命中 Doc 4 VragEngine)\n";
        auto hits = engine.search(query, q_vec, 3);
        for (std::size_t i = 0; i < hits.size(); ++i) {
            std::cout << "  #" << i + 1 << " [Doc " << hits[i].doc_id << "] RRF: " 
                      << std::fixed << std::setprecision(5) << hits[i].score << '\n'
                      << "     " << hits[i].text.substr(0, 70) << "...\n";
        }
        std::cout << '\n';
    }

    return 0;
}
