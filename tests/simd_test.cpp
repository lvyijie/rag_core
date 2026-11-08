#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include "../include/arena.hpp"
#include "../include/simd_vector.hpp"
#include "../include/simd_int8.hpp"

using namespace vrag;

float scalar_dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float scalar_fp32_int8_dot(const float* a, const int8_t* b, std::size_t dim, float scale) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * static_cast<float>(b[i]);
    }
    return sum * scale;
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

    // 测试 SQ8 标量量化与混合精度/全INT8点积（含非 64 整数倍维度尾部处理）
    constexpr std::size_t OddDim = 773;
    std::vector<float> q_raw(OddDim);
    std::vector<float> d_raw(OddDim);
    for (std::size_t i = 0; i < OddDim; ++i) {
        q_raw[i] = std::sin(static_cast<float>(i) * 0.05f);
        d_raw[i] = std::cos(static_cast<float>(i) * 0.03f);
    }

    std::vector<float> q_vec = q_raw;
    std::vector<float> d_vec = d_raw;
    SimdVector::normalize(q_vec);
    SimdVector::normalize(d_vec);

    std::vector<int8_t> d_int8(OddDim);
    float scale = SimdInt8::quantize(d_vec, d_int8);
    float expected_i8_dot = scalar_fp32_int8_dot(q_vec.data(), d_int8.data(), OddDim, scale);
    float actual_i8_dot = SimdInt8::dot_product_fp32_int8_avx2(q_vec.data(), d_int8.data(), OddDim, scale);
    float fp32_dot = SimdVector::dot_product_avx2(q_vec, d_vec);

    if (std::abs(expected_i8_dot - actual_i8_dot) > 1e-4f) {
        std::cerr << "[FAIL] SimdInt8 AVX2 dot product mismatch against scalar reference!\n";
        return 1;
    }
    if (std::abs(fp32_dot - actual_i8_dot) > 1e-2f) {
        std::cerr << "[FAIL] SQ8 quantization error too large: fp32=" << fp32_dot
                  << " sq8=" << actual_i8_dot << '\n';
        return 1;
    }

    // 测试融合 normalize_and_quantize 与 INT8 x INT8 点积
    std::vector<int8_t> q_int8(OddDim);
    std::vector<uint8_t> q_abs(OddDim);
    std::vector<int8_t> d_fused_int8(OddDim);
    float q_scale = SimdInt8::normalize_and_quantize(q_raw, q_int8, q_abs.data());
    float d_scale = SimdInt8::normalize_and_quantize(d_raw, d_fused_int8);

    float i8_x_i8_dot = SimdInt8::dot_product_int8_avx2(
        q_int8.data(), q_abs.data(), d_fused_int8.data(), OddDim, q_scale * d_scale);

    if (std::abs(fp32_dot - i8_x_i8_dot) > 1e-2f) {
        std::cerr << "[FAIL] INT8xINT8 dot product error too large: fp32=" << fp32_dot
                  << " i8xi8=" << i8_x_i8_dot << '\n';
        return 1;
    }

    std::cout << "[PASS] SimdInt8 quantization, FP32xINT8, and INT8xINT8 AVX2 dot product verified.\n";
    return 0;
}
