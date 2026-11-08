#pragma once

#include <cstddef>
#include <cmath>
#include <span>
#include <immintrin.h>

namespace vrag {

class SimdVector {
public:
    // 计算两个 float 向量的点积 (AVX2 + FMA)
    // 允许任意内存对齐 (对齐时与 _mm256_load_ps 吞吐完全相同，未对齐时不触发 #GP 段错误)
    static float dot_product_avx2(std::span<const float> a, std::span<const float> b) noexcept {
        const std::size_t size = a.size();
        const float* ptr_a = a.data();
        const float* ptr_b = b.data();

        // 4 个独立的 YMM 累加寄存器，利用 ILP（指令级并行）打满流水线
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        std::size_t i = 0;
        // 每次步进 32 个 float (4 * 8)
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

        // 处理剩余的 8 的倍数部分
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(ptr_a + i);
            __m256 vb = _mm256_loadu_ps(ptr_b + i);
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }

        // 合并 4 个累加器
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // 横向求和 (Horizontal Sum)
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float result = _mm_cvtss_f32(sum128);

        // 标量处理尾部不能被 8 整除的部分
        for (; i < size; ++i) {
            result += ptr_a[i] * ptr_b[i];
        }

        return result;
    }

    // 向量 L2 范数归一化 (inplace)
    static void normalize(std::span<float> vec) noexcept {
        float dot = dot_product_avx2(vec, vec);
        if (dot > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(dot);
            __m256 v_inv = _mm256_set1_ps(inv_norm);
            
            std::size_t i = 0;
            const std::size_t size = vec.size();
            float* ptr = vec.data();

            for (; i + 7 < size; i += 8) {
                __m256 v = _mm256_loadu_ps(ptr + i);
                _mm256_storeu_ps(ptr + i, _mm256_mul_ps(v, v_inv));
            }
            for (; i < size; ++i) {
                ptr[i] *= inv_norm;
            }
        }
    }
};

} // namespace vrag
