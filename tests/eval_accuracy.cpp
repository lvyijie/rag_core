#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "../include/vrag.hpp"

using namespace vrag;

struct TestCase {
    std::string type;
    std::string query_text;
    float vec_seed;
};

int main(int argc, char* argv[]) {
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
        {"专有名词", "賈寶玉神遊太虛境", 0.03f},
        {"专有名词", "林黛玉焚稿斷癡情", 0.05f},
        {"专有名词", "王熙鳳毒設相思局", 0.08f},
        {"专有名词", "冷子興演說榮國府衰敗", 0.12f},
        {"专有名词", "薛寶釵服冷香丸配方", 0.15f},
        {"抽象语义", "豪門望族盛極而衰的命運悲劇", 0.20f},
        {"抽象语义", "女主角生病時服用的靈丹妙藥與仙氣", 0.25f},
        {"抽象语义", "道士與僧人勘破紅塵的神仙預言", 0.30f},
        {"抽象语义", "富貴人家奢華宴席背後的虛偽與勾心鬥角", 0.35f},
        {"抽象语义", "對天地未開前頑石幻化人世的哲理思考", 0.40f}
    };

    constexpr std::size_t TopK = 5;

    std::cout << std::left 
              << std::setw(12) << "类型" 
              << std::setw(36) << "查询内容 (Query)" 
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
            std::string preview = hits[0].text;
            if (preview.size() > 30) {
                preview = preview.substr(0, 30) + "...";
            }
            std::cout << std::left 
                      << std::setw(12) << tc.type 
                      << std::setw(36) << tc.query_text 
                      << std::setw(10) << hits[0].doc_id 
                      << std::fixed << std::setprecision(4) << std::setw(12) << hits[0].score 
                      << preview << "\n";
        }
    }

    std::cout << std::string(100, '-') << "\n";
    return 0;
}
