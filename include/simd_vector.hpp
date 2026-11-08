#pragma once

#include <cmath>
#include <cstddef>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vrag {

class SimdVector {
public:
    // 计算两个 float 向量的点积 (AVX2 + FMA 4路循环展开，非 AVX2 平台自动回退至 4 路标量流水线)
    static float dot_product_avx2(std::span<const float> a, std::span<const float> b) noexcept {
        const std::size_t size = a.size();
        const float* __restrict ptr_a = a.data();
        const float* __restrict ptr_b = b.data();

#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        std::size_t i = 0;
        for (; i + 31 < size; i += 32) {
            __m256 va0 = _mm256_loadu_ps(ptr_a + i);
            __m256 vb0 = _mm256_loadu_ps(ptr_b + i);
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

            __m256 va1 = _mm256_loadu_ps(ptr_a + i + 8);
            __m256 vb1 = _mm256_loadu_ps(ptr_b + i + 8);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

            __m256 va2 = _mm256_loadu_ps(ptr_a + i + 16);
            __m256 vb2 = _mm256_loadu_ps(ptr_b + i + 16);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

            __m256 va3 = _mm256_loadu_ps(ptr_a + i + 24);
            __m256 vb3 = _mm256_loadu_ps(ptr_b + i + 24);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }

        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(ptr_a + i);
            __m256 vb = _mm256_loadu_ps(ptr_b + i);
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);

        float result = _mm_cvtss_f32(sums);

        for (; i < size; ++i) {
            result += ptr_a[i] * ptr_b[i];
        }

        return result;
#else
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        std::size_t i = 0;
        for (; i + 3 < size; i += 4) {
            acc0 += ptr_a[i] * ptr_b[i];
            acc1 += ptr_a[i + 1] * ptr_b[i + 1];
            acc2 += ptr_a[i + 2] * ptr_b[i + 2];
            acc3 += ptr_a[i + 3] * ptr_b[i + 3];
        }
        float result = (acc0 + acc1) + (acc2 + acc3);
        for (; i < size; ++i) {
            result += ptr_a[i] * ptr_b[i];
        }
        return result;
#endif
    }

    // 向量 L2 范数原位归一化
    static void normalize(std::span<float> vec) noexcept {
        float dot = dot_product_avx2(vec, vec);
        if (dot > 0.0f) {
            const float inv_norm = 1.0f / std::sqrt(dot);
            const std::size_t size = vec.size();
            float* ptr = vec.data();
            std::size_t i = 0;

#if defined(__AVX2__)
            __m256 v_inv = _mm256_set1_ps(inv_norm);
            for (; i + 31 < size; i += 32) {
                _mm256_storeu_ps(ptr + i,      _mm256_mul_ps(_mm256_loadu_ps(ptr + i),      v_inv));
                _mm256_storeu_ps(ptr + i + 8,  _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 8),  v_inv));
                _mm256_storeu_ps(ptr + i + 16, _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 16), v_inv));
                _mm256_storeu_ps(ptr + i + 24, _mm256_mul_ps(_mm256_loadu_ps(ptr + i + 24), v_inv));
            }
            for (; i + 7 < size; i += 8) {
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(_mm256_loadu_ps(ptr + i), v_inv));
            }
#endif
            for (; i < size; ++i) {
                ptr[i] *= inv_norm;
            }
        }
    }
};

} // namespace vrag
