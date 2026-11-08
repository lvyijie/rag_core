#pragma once
// 头文件保护
#include <cstdint>
#include <string_view>
#include <vector>

namespace vrag {
// hash func
// 独立函数
constexpr uint32_t fnv1a_hash(std::string_view sv) noexcept {
    uint32_t hash = 2166136261u;
    for (unsigned char c : sv) {
        hash ^= c;
        hash *= 16777619u;
    }
    return hash;
}

inline bool is_cjk(uint32_t code_point) noexcept {
    return code_point >= 0x4E00 && code_point <= 0x9FA5;
}

class Tokenizer {
public:
    std::vector<std::string_view> cut(std::string_view text) {
        std::vector<std::string_view> result;
        std::string_view prev_cjk = "";
        for (int i = 0; i < text.size(); ++i) {
            unsigned char c = text[i];
            if ((c & 0xF0) == 0xE0) {
                if (i + 2 < text.size()) {
                    uint32_t b0 = c;
                    uint32_t b1 = text[i + 1];
                    uint32_t b2 = text[i + 2];
                    uint32_t code_point =
                        ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                    if (is_cjk(code_point)) {
                        std::string_view curr_cjk(&text[i], 3);
                        if (!prev_cjk.empty()) {
                            std::string_view bigram(prev_cjk.data(), 6);
                            result.push_back(bigram);
                        }
                        prev_cjk = curr_cjk;
                        i += 2;
                    } else {
                        prev_cjk = "";
                    }
                }
            }
        }
        return result;
    }
};

} // namespace vrag
