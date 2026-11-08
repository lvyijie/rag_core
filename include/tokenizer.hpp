#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

namespace vrag {

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
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::string_view;
        using difference_type = std::ptrdiff_t;
        using pointer = const std::string_view*;
        using reference = const std::string_view&;

        Iterator(std::string_view text, size_t pos)
            : text_(text), pos_(pos), prev_cjk_(""), current_token_("") {
            if (pos_ < text_.size()) {
                advance();
            }
        }

        reference operator*() const noexcept { return current_token_; }
        pointer operator->() const noexcept { return &current_token_; }

        Iterator& operator++() {
            advance();
            return *this;
        }

        bool operator==(const Iterator& other) const noexcept {
            return pos_ == other.pos_ && current_token_ == other.current_token_;
        }

        bool operator!=(const Iterator& other) const noexcept {
            return !(*this == other);
        }

    private:
        void advance() {
            current_token_ = "";
            while (pos_ < text_.size()) {
                unsigned char c = text_[pos_];

                // 尝试匹配 3 字节 UTF-8 字符
                if ((c & 0xF0) == 0xE0) {
                    if (pos_ + 2 < text_.size()) {
                        uint32_t b0 = c;
                        uint32_t b1 = text_[pos_ + 1];
                        uint32_t b2 = text_[pos_ + 2];
                        uint32_t code_point =
                            ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);

                        if (is_cjk(code_point)) {
                            std::string_view curr_cjk(&text_[pos_], 3);
                            pos_ += 3; // 消费掉当前 3 字节汉字

                            if (!prev_cjk_.empty()) {
                                // 成功构成 Bi-gram，装填后立即返回
                                current_token_ = std::string_view(
                                    prev_cjk_.data(),
                                    prev_cjk_.size() + curr_cjk.size()
                                );
                                prev_cjk_ = curr_cjk;
                                return;
                            }

                            // 第一个汉字，暂存后继续向后扫下一个字
                            prev_cjk_ = curr_cjk;
                            continue;
                        }
                    }
                }

                // 遇到非汉字（英文、标点、空格等），强制打断滑窗并前进一步
                prev_cjk_ = "";
                ++pos_;
            }
        }

        std::string_view text_;
        size_t pos_;
        std::string_view prev_cjk_;
        std::string_view current_token_;
    };

    struct Range {
        std::string_view text;
        Iterator begin() const { return Iterator(text, 0); }
        Iterator end() const { return Iterator(text, text.size()); }
    };

    // 流式范围接口：支持 for (auto token : tokenizer.cut(text))
    [[nodiscard]] static Range cut(std::string_view text) noexcept {
        return Range{text};
    }
};

} // namespace vrag
