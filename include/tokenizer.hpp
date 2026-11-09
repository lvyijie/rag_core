#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

namespace vrag {

namespace detail {

// 编译期 ASCII 字符分类表：0 = 标点/空白/其他, 1 = 字母或数字 [a-zA-Z0-9]
inline constexpr std::array<std::uint8_t, 256> kAsciiAlnumTable = []() constexpr {
    std::array<std::uint8_t, 256> table{};
    for (int c = '0'; c <= '9'; ++c) table[c] = 1;
    for (int c = 'A'; c <= 'Z'; ++c) table[c] = 1;
    for (int c = 'a'; c <= 'z'; ++c) table[c] = 1;
    return table;
}();

// 编译期 ASCII 小写转换表（用于大小写不敏感哈希，UTF-8 高位字节保持原样）
inline constexpr std::array<unsigned char, 256> kAsciiLowerTable = []() constexpr {
    std::array<unsigned char, 256> table{};
    for (int i = 0; i < 256; ++i) {
        table[i] = (i >= 'A' && i <= 'Z') ? static_cast<unsigned char>(i + 32)
                                          : static_cast<unsigned char>(i);
    }
    return table;
}();

} // namespace detail

constexpr uint32_t fnv1a_hash(std::string_view sv) noexcept {
    uint32_t hash = 2166136261u;
    for (unsigned char c : sv) {
        hash ^= detail::kAsciiLowerTable[c];
        hash *= 16777619u;
    }
    return hash;
}

inline constexpr bool is_cjk(uint32_t code_point) noexcept {
    return code_point >= 0x4E00 && code_point <= 0x9FA5;
}

inline constexpr bool is_cjk_utf8(const char* p) noexcept {
    const auto b0 = static_cast<unsigned char>(p[0]);
    if (b0 < 0xE4 || b0 > 0xE9) {
        return false;
    }
    const uint32_t b1 = static_cast<unsigned char>(p[1]);
    const uint32_t b2 = static_cast<unsigned char>(p[2]);
    if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u) {
        return false;
    }
    const uint32_t code_point = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
    return is_cjk(code_point);
}

class Tokenizer {
public:
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::string_view;
        using difference_type = std::ptrdiff_t;
        using pointer = const std::string_view*;
        using reference = const std::string_view&;

        // 构造 end 迭代器（零开销）
        constexpr Iterator() noexcept = default;

        Iterator(std::string_view text, size_t pos) noexcept
            : cur_(text.data() + pos),
              end_(text.data() + text.size()),
              current_token_(nullptr, 0),
              has_prev_cjk_(false) {
            if (cur_ < end_) {
                advance();
            }
        }

        reference operator*() const noexcept { return current_token_; }
        pointer operator->() const noexcept { return &current_token_; }

        Iterator& operator++() noexcept {
            advance();
            return *this;
        }

        bool operator==(const Iterator& other) const noexcept {
            return current_token_.data() == other.current_token_.data();
        }

        bool operator!=(const Iterator& other) const noexcept {
            return current_token_.data() != other.current_token_.data();
        }

    private:
        void advance() noexcept {
            // 极速热路径：连续中文字符流直接配对构成 6 字节 Bi-gram
            if (has_prev_cjk_ && cur_ + 3 <= end_ && is_cjk_utf8(cur_)) {
                const char* token_start = cur_ - 3;
                cur_ += 3;
                current_token_ = std::string_view(token_start, 6);
                return;
            }

            while (cur_ < end_) {
                const auto c = static_cast<unsigned char>(*cur_);

                // 1. ASCII 字符分支 (0x00..0x7F)：处理英文、数字与半角标点
                if (c < 0x80u) {
                    has_prev_cjk_ = false;
                    if (detail::kAsciiAlnumTable[c]) {
                        const char* start = cur_++;
                        while (cur_ < end_ && detail::kAsciiAlnumTable[static_cast<unsigned char>(*cur_)]) {
                            ++cur_;
                        }
                        current_token_ = std::string_view(start, static_cast<size_t>(cur_ - start));
                        return;
                    }
                    // 连续跳过 ASCII 标点与空白符
                    ++cur_;
                    while (cur_ < end_) {
                        const auto nc = static_cast<unsigned char>(*cur_);
                        if (nc >= 0x80u || detail::kAsciiAlnumTable[nc]) {
                            break;
                        }
                        ++cur_;
                    }
                    continue;
                }

                // 2. 3 字节 UTF-8 字符分支 (0xE0..0xEF)：覆盖汉字与全角标点（如 ，。！？“”【】）
                if ((c & 0xF0u) == 0xE0u) {
                    if (cur_ + 3 <= end_) {
                        if (is_cjk_utf8(cur_)) {
                            cur_ += 3;
                            if (has_prev_cjk_) {
                                current_token_ = std::string_view(cur_ - 6, 6);
                                return;
                            }
                            has_prev_cjk_ = true;
                            // 立即尝试匹配紧邻的下一个汉字，减少外层循环跳转
                            if (cur_ + 3 <= end_ && is_cjk_utf8(cur_)) {
                                cur_ += 3;
                                current_token_ = std::string_view(cur_ - 6, 6);
                                return;
                            }
                            continue;
                        }
                        // 3 字节非汉字（全角标点符号等）：若后续字节为合法 UTF-8 延续字节则跳过 3 字节，否则仅跳过当前损坏字节
                        has_prev_cjk_ = false;
                        const auto b1 = static_cast<unsigned char>(cur_[1]);
                        const auto b2 = static_cast<unsigned char>(cur_[2]);
                        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
                            cur_ += 3;
                        } else {
                            ++cur_;
                        }
                        continue;
                    }
                    has_prev_cjk_ = false;
                    ++cur_;
                    continue;
                }

                // 3. 2 字节或 4 字节 UTF-8 字符（其他语种或 Emoji 符号）：打断滑窗并按合法延续字节长度跳过
                has_prev_cjk_ = false;
                if ((c & 0xE0u) == 0xC0u) {
                    if (cur_ + 2 <= end_ && (static_cast<unsigned char>(cur_[1]) & 0xC0u) == 0x80u) {
                        cur_ += 2;
                    } else {
                        ++cur_;
                    }
                } else if ((c & 0xF8u) == 0xF0u) {
                    if (cur_ + 4 <= end_ &&
                        (static_cast<unsigned char>(cur_[1]) & 0xC0u) == 0x80u &&
                        (static_cast<unsigned char>(cur_[2]) & 0xC0u) == 0x80u &&
                        (static_cast<unsigned char>(cur_[3]) & 0xC0u) == 0x80u) {
                        cur_ += 4;
                    } else {
                        ++cur_;
                    }
                } else {
                    ++cur_;
                }
            }

            current_token_ = std::string_view(nullptr, 0);
        }

        const char* cur_{nullptr};
        const char* end_{nullptr};
        std::string_view current_token_{nullptr, 0};
        bool has_prev_cjk_{false};
    };

    struct Range {
        std::string_view text;
        Iterator begin() const noexcept { return Iterator(text, 0); }
        Iterator end() const noexcept { return Iterator(); }
    };

    // 流式范围接口：支持 for (auto token : tokenizer.cut(text))
    [[nodiscard]] static Range cut(std::string_view text) noexcept {
        return Range{text};
    }
};

} // namespace vrag
