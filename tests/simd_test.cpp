#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../include/arena.hpp"
#include "../include/simd_vector.hpp"

using namespace vrag;

float scalar_dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    constexpr std::size_t Dimension = 768;
    Arena arena(1024 * 1024);

    auto vec_a = arena.allocate_vector(Dimension);
    auto vec_b = arena.allocate_vector(Dimension);

    for (std::size_t i = 0; i < Dimension; ++i) {
        vec_a[i] = static_cast<float>(i % 10) * 0.1f;
        vec_b[i] = static_cast<float>((i + 5) % 10) * 0.1f;
    }

    float ground_truth = scalar_dot(vec_a, vec_b);
    float avx_result = SimdVector::dot_product_avx2(vec_a, vec_b);

    // 打印高精度浮点细节，查看微小差值
    std::cout << std::setprecision(8);
    std::cout << "Scalar dot product: " << ground_truth << '\n';
    std::cout << "AVX2   dot product: " << avx_result << '\n';
    std::cout << "Diff              : " << std::abs(ground_truth - avx_result) << '\n';

    // 768 维累加的合理误差容限 (1e-3f 或相对误差 < 0.01%)
    float diff = std::abs(ground_truth - avx_result);
    if (diff > 1e-3f && (diff / ground_truth) > 1e-4f) {
        std::cerr << "[FAIL] AVX2 result does not match scalar ground truth!\n";
        return 1;
    }

    std::cout << "[PASS] AVX2 dot product precision verified.\n";

    // 测试归一化
    SimdVector::normalize(vec_a);
    float norm_a = SimdVector::dot_product_avx2(vec_a, vec_a);
    std::cout << "Normalized vector length (A dot A): " << norm_a << '\n';

    if (std::abs(norm_a - 1.0f) > 1e-3f) {
        std::cerr << "[FAIL] Vector normalization failed!\n";
        return 1;
    }

    std::cout << "[PASS] Vector normalization verified.\n";
    return 0;
}
