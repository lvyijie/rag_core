#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include "../include/vrag.hpp"

using namespace vrag;

// 调用系统的 curl 获取 query 的 1024 维实时向量
bool fetch_query_embedding(const std::string& query, std::vector<float>& out_vec) {
    const char* api_key = std::getenv("EMBED_API_KEY");
    if (!api_key || std::string(api_key).empty()) {
        return false;
    }

    // 简单转义双引号
    std::string safe_query;
    for (char c : query) {
        if (c == '"') safe_query += "\\\"";
        else safe_query += c;
    }

    std::string cmd = "curl -s -X POST https://api.siliconflow.cn/v1/embeddings "
                      "-H \"Authorization: Bearer " + std::string(api_key) + "\" "
                      "-H \"Content-Type: application/json\" "
                      "-d '{\"model\": \"BAAI/bge-m3\", \"input\": [\"" + safe_query + "\"]}'";

    std::array<char, 4096> buffer;
    std::string response;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return false;

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        response += buffer.data();
    }

    // 极速解析 JSON 中的 embedding 浮点数数组
    auto pos = response.find("\"embedding\":[");
    if (pos == std::string::npos) return false;
    pos += 13;

    out_vec.clear();
    while (pos < response.size() && response[pos] != ']') {
        while (pos < response.size() && (response[pos] == ' ' || response[pos] == ',')) pos++;
        if (response[pos] == ']') break;
        char* end_ptr = nullptr;
        float val = std::strtof(&response[pos], &end_ptr);
        if (&response[pos] == end_ptr) break;
        out_vec.push_back(val);
        pos = end_ptr - &response[0];
    }

    return out_vec.size() == 1024;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "使用方法: " << argv[0] << " <knowledge_base.bin>\n";
        return 1;
    }

    constexpr std::size_t Dim = 1024;
    VragEngine engine(Dim);

    std::cout << "正在从 " << argv[1] << " 加载知识库...\n";
    if (!engine.load_from_binary(argv[1])) {
        return 1;
    }
    std::cout << "加载完成！已成功索引 " << engine.total_docs() << " 篇文档切片。\n";

    const bool has_key = (std::getenv("EMBED_API_KEY") != nullptr);
    if (has_key) {
        std::cout << "[+] 检测到 EMBED_API_KEY，已激活【BM25 + BGE-M3 语义向量】双路混合检索模式！\n";
    } else {
        std::cout << "[-] 未检测到 EMBED_API_KEY，当前仅运行【BM25 稀疏字面检索】模式。\n";
    }

    std::string query;
    while (true) {
        std::cout << "\n请输入检索内容 (输入 exit 退出): ";
        if (!std::getline(std::cin, query) || query == "exit") {
            break;
        }
        if (query.empty()) continue;

        std::vector<float> query_vec(Dim, 0.0f);
        if (has_key) {
            std::cout << "[*] 正在请求 BGE-M3 语义向量...";
            std::cout.flush();
            if (fetch_query_embedding(query, query_vec)) {
                std::cout << " 成功！开始双路融合检索。\n";
            } else {
                std::cout << " 失败，降级为字面检索。\n";
                std::fill(query_vec.begin(), query_vec.end(), 0.0f);
            }
        }

        auto hits = engine.search(query, query_vec, 5);

        std::cout << "\n================ 检索结果 (Top 5) ================\n";
        for (std::size_t i = 0; i < hits.size(); ++i) {
            std::cout << "#" << i + 1 << " [DocID: " << hits[i].doc_id 
                      << "] RRF得分: " << std::fixed << std::setprecision(5) << hits[i].score << '\n';
            std::cout << "内容: " << hits[i].text << "\n";
            std::cout << "--------------------------------------------------\n";
        }
    }

    return 0;
}
