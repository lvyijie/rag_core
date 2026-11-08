#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#if defined(_WIN32)
#include <windows.h>
#endif
#include "../include/vrag.hpp"

using namespace vrag;

struct TestCase {
    std::string type;
    std::string query_text;
    float vec_seed;
};

// 安全截取 UTF-8 字符串前 max_bytes 字节（不截断半个多字节汉字，并替换换行符）
std::string utf8_preview(std::string_view sv, std::size_t max_bytes = 60) {
    std::string clean;
    clean.reserve(std::min(sv.size(), max_bytes + 16));
    std::size_t i = 0;
    while (i < sv.size() && clean.size() < max_bytes) {
        unsigned char c = static_cast<unsigned char>(sv[i]);
        if (c == '\r' || c == '\n' || c == '\t') {
            if (!clean.empty() && clean.back() != ' ') clean.push_back(' ');
            ++i;
            continue;
        }
        std::size_t char_len = 1;
        if ((c & 0xE0u) == 0xC0u) char_len = 2;
        else if ((c & 0xF0u) == 0xE0u) char_len = 3;
        else if ((c & 0xF8u) == 0xF0u) char_len = 4;

        if (i + char_len > sv.size()) break;
        clean.append(sv.data() + i, char_len);
        i += char_len;
    }
    if (i < sv.size()) clean += "...";
    return clean;
}

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <corpus.bin>\n";
        return 1;
    }

    constexpr std::size_t Dim = 1024;
    std::string bin_path = argv[1];

    std::cout << "=== VRAG 精确检索验证 ===\n";
    std::cout << "[*] 正在加载测试语料库: " << bin_path << " ...\n";

    VragEngine engine(Dim);
    if (!engine.load_from_binary(bin_path)) {
        std::cerr << "[-] 语料载入失败！\n";
        return 1;
    }
    std::cout << "[+] 语料加载完毕，当前索引文档数: " << engine.total_docs() << "\n\n";

    std::vector<TestCase> test_cases = {
        {"专有名词", "贾宝玉神游太虚境", 0.03f},
        {"专有名词", "林黛玉焚稿断痴情", 0.05f},
        {"专有名词", "王熙凤毒设相思局", 0.08f},
        {"专有名词", "冷子兴演说荣国府", 0.12f},
        {"专有名词", "薛宝钗冷香丸配方", 0.15f},
        {"名著典故", "诸葛亮孔明草船借箭", 0.20f},
        {"名著典故", "关羽温酒斩华雄千里走单骑", 0.25f},
        {"名著典故", "孙悟空大闹天宫齐天大圣", 0.30f},
        {"名著典故", "三打白骨精唐僧取经", 0.35f},
        {"名著典故", "鲁智深倒拔垂杨柳林冲风雪山神庙", 0.40f}
    };

    constexpr std::size_t TopK = 5;

    std::cout << std::left 
              << std::setw(12) << "类型" 
              << std::setw(34) << "查询内容 (Query)" 
              << std::setw(10) << "Top1 ID" 
              << std::setw(12) << "RRF 得分" 
              << "Top 1 文本预览\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& tc : test_cases) {
        std::vector<float> q_vec(Dim, 0.03125f);
        for (std::size_t i = 0; i < Dim; ++i) {
            q_vec[i] += (i % 7 == 0) ? tc.vec_seed : -tc.vec_seed * 0.5f;
        }

        auto hits = engine.search(tc.query_text, q_vec, TopK, 60.0f);

        if (!hits.empty()) {
            std::cout << std::left 
                      << std::setw(12) << tc.type 
                      << std::setw(34) << tc.query_text 
                      << std::setw(10) << hits[0].doc_id 
                      << std::fixed << std::setprecision(4) << std::setw(12) << hits[0].score 
                      << utf8_preview(hits[0].text, 54) << "\n";
        }
    }

    std::cout << std::string(100, '-') << "\n";
    return 0;
}
