#include <iostream>
#include <string_view>
#include <vector>

#include "../include/tokenizer.hpp"

using namespace vrag;

int passed = 0;
int failed = 0;

template <typename RangeType>
void expect_equal(
    RangeType&& actual_range,
    const std::vector<std::string_view>& expected,
    std::string_view test_name
) {
    // 将惰性 Range 转为 vector 统一对比断言
    std::vector<std::string_view> actual(actual_range.begin(), actual_range.end());

    if (actual == expected) {
        ++passed;
        std::cout << "[PASS] " << test_name << '\n';
    } else {
        ++failed;
        std::cout << "[FAIL] " << test_name << '\n';

        std::cout << "  Expected: ";
        for (auto s : expected) std::cout << '[' << s << ']';

        std::cout << "\n  Actual:   ";
        for (auto s : actual) std::cout << '[' << s << ']';

        std::cout << '\n';
    }
}

int main() {
    Tokenizer tokenizer;

    // 测试 1：正常中文
    {
        auto result = tokenizer.cut("我喜欢仓鼠");

        expect_equal(
            result,
            {"我喜", "喜欢", "欢仓", "仓鼠"},
            "Chinese bigram"
        );
    }

    // 测试 2：空文本
    {
        auto result = tokenizer.cut("");

        expect_equal(
            result,
            {},
            "Empty input"
        );
    }

    // 测试 3：单个汉字
    {
        auto result = tokenizer.cut("鼠");

        expect_equal(
            result,
            {},
            "Single Chinese character"
        );
    }

    // 测试 4：两个汉字
    {
        auto result = tokenizer.cut("仓鼠");

        expect_equal(
            result,
            {"仓鼠"},
            "Two Chinese characters"
        );
    }

    // 测试 5：非汉字字符（英文/标点隔离打断）
    {
        auto result = tokenizer.cut("三sal门下");

        expect_equal(
            result,
            {"门下"},
            "Non-CJK boundary separation"
        );
    }

    std::cout << "\n";
    std::cout << "Passed: " << passed << '\n';
    std::cout << "Failed: " << failed << '\n';

    return failed == 0 ? 0 : 1;
}
