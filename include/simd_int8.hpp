#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <span>
#include <immintrin.h>
#include <algorithm>

namespace vrag {

class SimdInt8 {
public:
    // 计算 float 查询向量与 int8 量化文档向量的带权点积
    // a: float 查询向量 (dim 维)
    // b_int8: int8 文档向量 (dim 维)
    // scale: 该文档向量的量化 scale 因子
    static float dot_product_fp32_int8_avx2(
        const float* a, 
        const int8_t* b_int8, 
        std::size_t dim, 
        float scale
    ) noexcept {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        std::size_t i = 0;
        // 每次处理 16 个分量
        for (; i + 15 < dim; i += 16) {
            // 1. 读取 16 个 int8 并符号扩展为 16 个 16-bit int
            __m128i raw_i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_int8 + i));
            __m256i i16_256 = _mm256_cvtepi8_epi16(raw_i8);

            // 2. 拆分为两组 8 个 32-bit int 并转换为 float
            __m128i lo_i16 = _mm256_castsi256_si128(i16_256);
            __m128i hi_i16 = _mm256_extracti128_si256(i16_256, 1);

            __m256 b_float_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_i16));
            __m256 b_float_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_i16));

            // 3. 读取 float 查询向量并执行 FMA 乘加
            __m256 a_float_0 = _mm256_loadu_ps(a + i);
            __m256 a_float_1 = _mm256_loadu_ps(a + i + 8);

            acc0 = _mm256_fmadd_ps(a_float_0, b_float_0, acc0);
            acc1 = _mm256_fmadd_ps(a_float_1, b_float_1, acc1);
        }

        acc0 = _mm256_add_ps(acc0, acc1);

        // 水平规约求和
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float sum = _mm_cvtss_f32(sum128);

        // 处理尾部余数
        for (; i < dim; ++i) {
            sum += a[i] * static_cast<float>(b_int8[i]);
        }

        // 最终还原真实尺度
        return sum * scale;
    }

    // 将浮点向量量化为 int8 格式，并输出 scale
    static float quantize(std::span<const float> src, std::span<int8_t> dst) noexcept {
        float max_abs = 0.0f;
        for (float val : src) {
            float abs_val = std::fabs(val);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        if (max_abs == 0.0f) {
            std::fill(dst.begin(), dst.end(), 0);
            return 1.0f;
        }

        float scale = max_abs / 127.0f;
        float inv_scale = 127.0f / max_abs;

        for (std::size_t i = 0; i < src.size(); ++i) {
            int val = std::round(src[i] * inv_scale);
            dst[i] = static_cast<int8_t>(std::clamp(val, -128, 127));
        }

        return scale;
    }
};

} // namespace vrag
