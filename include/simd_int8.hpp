#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vrag {

class SimdInt8 {
public:
    // 计算 float 查询向量与 int8 量化文档向量的带权点积
    static float dot_product_fp32_int8_avx2(
        const float* __restrict a,
        const int8_t* __restrict b_int8,
        std::size_t dim,
        float scale
    ) noexcept {
        std::size_t i = 0;
#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (; i + 31 < dim; i += 32) {
            __m256 b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_int8 + i))));
            __m256 b1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_int8 + i + 8))));
            __m256 b2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_int8 + i + 16))));
            __m256 b3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_int8 + i + 24))));

            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      b0, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  b1, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), b2, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), b3, acc3);
        }

        for (; i + 7 < dim; i += 8) {
            __m256 b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_int8 + i))));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), b0, acc0);
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

        float sum = _mm_cvtss_f32(sums);
#else
        float sum = 0.0f;
#endif
        for (; i < dim; ++i) {
            sum += a[i] * static_cast<float>(b_int8[i]);
        }

        return sum * scale;
    }

    // INT8 x INT8 点积算子 (支持 AVX-VNNI / AVX2 / 标量跨平台回退)
    static float dot_product_int8_avx2(
        const int8_t* __restrict q_int8,
        const uint8_t* __restrict q_abs,
        const int8_t* __restrict b_int8,
        std::size_t dim,
        float combined_scale
    ) noexcept {
        std::size_t i = 0;
#if defined(__AVX2__)
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
#if !defined(__AVXVNNI__)
        const __m256i ones = _mm256_set1_epi16(1);
#endif

        for (; i + 127 < dim; i += 128) {
            __m256i vq_sig0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_int8 + i));
            __m256i vq_abs0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_abs + i));
            __m256i vb0     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i));

            __m256i vq_sig1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_int8 + i + 32));
            __m256i vq_abs1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_abs + i + 32));
            __m256i vb1     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 32));

            __m256i vq_sig2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_int8 + i + 64));
            __m256i vq_abs2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_abs + i + 64));
            __m256i vb2     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 64));

            __m256i vq_sig3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_int8 + i + 96));
            __m256i vq_abs3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_abs + i + 96));
            __m256i vb3     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 96));

#if defined(__AVXVNNI__)
            acc0 = _mm256_dpbusd_epi32(acc0, vq_abs0, _mm256_sign_epi8(vb0, vq_sig0));
            acc1 = _mm256_dpbusd_epi32(acc1, vq_abs1, _mm256_sign_epi8(vb1, vq_sig1));
            acc2 = _mm256_dpbusd_epi32(acc2, vq_abs2, _mm256_sign_epi8(vb2, vq_sig2));
            acc3 = _mm256_dpbusd_epi32(acc3, vq_abs3, _mm256_sign_epi8(vb3, vq_sig3));
#else
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(
                _mm256_maddubs_epi16(vq_abs0, _mm256_sign_epi8(vb0, vq_sig0)), ones));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(
                _mm256_maddubs_epi16(vq_abs1, _mm256_sign_epi8(vb1, vq_sig1)), ones));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(
                _mm256_maddubs_epi16(vq_abs2, _mm256_sign_epi8(vb2, vq_sig2)), ones));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(
                _mm256_maddubs_epi16(vq_abs3, _mm256_sign_epi8(vb3, vq_sig3)), ones));
#endif
        }

        for (; i + 31 < dim; i += 32) {
            __m256i vq_sig = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_int8 + i));
            __m256i vq_abs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_abs + i));
            __m256i vb     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i));
#if defined(__AVXVNNI__)
            acc0 = _mm256_dpbusd_epi32(acc0, vq_abs, _mm256_sign_epi8(vb, vq_sig));
#else
            __m256i prod16 = _mm256_maddubs_epi16(vq_abs, _mm256_sign_epi8(vb, vq_sig));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod16, ones));
#endif
        }

        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc0 = _mm256_add_epi32(acc0, acc2);

        __m128i lo = _mm256_castsi256_si128(acc0);
        __m128i hi = _mm256_extracti128_si256(acc0, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
        int32_t total_i32 = _mm_cvtsi128_si32(sum128);
#else
        (void)q_abs;
        int32_t total_i32 = 0;
#endif
        for (; i < dim; ++i) {
            total_i32 += static_cast<int32_t>(q_int8[i]) * static_cast<int32_t>(b_int8[i]);
        }

        return static_cast<float>(total_i32) * combined_scale;
    }

    // 零符号开销 VNNI 无符号偏移点积：sum((q_i8 + 128) * b_i8) - 128 * sum(b_i8)
    static float dot_product_u8_s8_fast(
        const uint8_t* __restrict q_u8,
        const int8_t* __restrict q_int8,
        const uint8_t* __restrict q_abs,
        const int8_t* __restrict b_int8,
        std::size_t padded_dim,
        int32_t b_sum_128,
        float combined_scale
    ) noexcept {
#if defined(__AVXVNNI__)
        (void)q_int8;
        (void)q_abs;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        std::size_t i = 0;
        for (; i + 127 < padded_dim; i += 128) {
            __m256i vq0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_u8 + i));
            __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i));
            acc0 = _mm256_dpbusd_epi32(acc0, vq0, vb0);

            __m256i vq1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_u8 + i + 32));
            __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 32));
            acc1 = _mm256_dpbusd_epi32(acc1, vq1, vb1);

            __m256i vq2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_u8 + i + 64));
            __m256i vb2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 64));
            acc2 = _mm256_dpbusd_epi32(acc2, vq2, vb2);

            __m256i vq3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_u8 + i + 96));
            __m256i vb3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i + 96));
            acc3 = _mm256_dpbusd_epi32(acc3, vq3, vb3);
        }

        for (; i + 31 < padded_dim; i += 32) {
            __m256i vq0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_u8 + i));
            __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_int8 + i));
            acc0 = _mm256_dpbusd_epi32(acc0, vq0, vb0);
        }

        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc0 = _mm256_add_epi32(acc0, acc2);

        __m128i lo = _mm256_castsi256_si128(acc0);
        __m128i hi = _mm256_extracti128_si256(acc0, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
        int32_t total_i32 = _mm_cvtsi128_si32(sum128) - b_sum_128;

        return static_cast<float>(total_i32) * combined_scale;
#else
        (void)q_u8;
        (void)b_sum_128;
        return dot_product_int8_avx2(q_int8, q_abs, b_int8, padded_dim, combined_scale);
#endif
    }

    // 将浮点向量对称量化为 int8 格式，并输出 scale
    static float quantize(std::span<const float> src, std::span<int8_t> dst) noexcept {
        const std::size_t size = src.size();
        const float* __restrict src_ptr = src.data();
        int8_t* __restrict dst_ptr = dst.data();

        std::size_t i = 0;
        float max_abs = 0.0f;

#if defined(__AVX2__)
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 vmax0 = _mm256_setzero_ps();
        __m256 vmax1 = _mm256_setzero_ps();

        for (; i + 15 < size; i += 16) {
            __m256 v0 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(src_ptr + i));
            __m256 v1 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(src_ptr + i + 8));
            vmax0 = _mm256_max_ps(vmax0, v0);
            vmax1 = _mm256_max_ps(vmax1, v1);
        }
        vmax0 = _mm256_max_ps(vmax0, vmax1);
        __m128 max128 = _mm_max_ps(_mm256_castps256_ps128(vmax0), _mm256_extractf128_ps(vmax0, 1));
        max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
        max_abs = _mm_cvtss_f32(max128);
#endif

        for (; i < size; ++i) {
            float abs_val = std::fabs(src_ptr[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        if (max_abs == 0.0f || !std::isfinite(max_abs)) {
            std::fill(dst.begin(), dst.end(), 0);
            return 0.0f;
        }

        const float scale = max_abs / 127.0f;
        const float inv_scale = 127.0f / max_abs;
        i = 0;

#if defined(__AVX2__)
        const __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        for (; i + 31 < size; i += 32) {
            __m256i i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i),      v_inv_scale));
            __m256i i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 8),  v_inv_scale));
            __m256i i32_2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 16), v_inv_scale));
            __m256i i32_3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 24), v_inv_scale));

            __m256i i16_0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32_0, i32_1), 0xD8);
            __m256i i16_1 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32_2, i32_3), 0xD8);
            __m256i i8_all = _mm256_permute4x64_epi64(_mm256_packs_epi16(i16_0, i16_1), 0xD8);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + i), i8_all);
        }
#endif

        for (; i < size; ++i) {
            int val = static_cast<int>(std::nearbyint(src_ptr[i] * inv_scale));
            dst_ptr[i] = static_cast<int8_t>(std::clamp(val, -128, 127));
        }

        return scale;
    }

    // 融合归一化 + 量化（单遍扫描同时计算 L2 范数、最大绝对值、无符号偏移向量及元素和）
    static float normalize_and_quantize(
        std::span<const float> src,
        std::span<int8_t> dst,
        uint8_t* __restrict dst_abs = nullptr,
        uint8_t* __restrict dst_u8 = nullptr,
        int32_t* __restrict out_sum_128 = nullptr
    ) noexcept {
        const std::size_t size = src.size();
        const float* __restrict src_ptr = src.data();
        int8_t* __restrict dst_ptr = dst.data();

        std::size_t i = 0;
        float dot = 0.0f;
        float max_abs = 0.0f;

#if defined(__AVX2__)
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 vmax0 = _mm256_setzero_ps();
        __m256 vmax1 = _mm256_setzero_ps();
        __m256 vdot0 = _mm256_setzero_ps();
        __m256 vdot1 = _mm256_setzero_ps();

        for (; i + 15 < size; i += 16) {
            __m256 v0 = _mm256_loadu_ps(src_ptr + i);
            __m256 v1 = _mm256_loadu_ps(src_ptr + i + 8);
            vdot0 = _mm256_fmadd_ps(v0, v0, vdot0);
            vdot1 = _mm256_fmadd_ps(v1, v1, vdot1);
            vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(sign_mask, v0));
            vmax1 = _mm256_max_ps(vmax1, _mm256_andnot_ps(sign_mask, v1));
        }

        vdot0 = _mm256_add_ps(vdot0, vdot1);
        __m128 dlo = _mm256_castps256_ps128(vdot0);
        __m128 dhi = _mm256_extractf128_ps(vdot0, 1);
        __m128 dsum = _mm_add_ps(dlo, dhi);
        __m128 dshuf = _mm_movehdup_ps(dsum);
        __m128 dsums = _mm_add_ps(dsum, dshuf);
        dshuf = _mm_movehl_ps(dshuf, dsums);
        dsums = _mm_add_ss(dsums, dshuf);
        dot = _mm_cvtss_f32(dsums);

        vmax0 = _mm256_max_ps(vmax0, vmax1);
        __m128 max128 = _mm_max_ps(_mm256_castps256_ps128(vmax0), _mm256_extractf128_ps(vmax0, 1));
        max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
        max_abs = _mm_cvtss_f32(max128);
#endif

        for (; i < size; ++i) {
            float v = src_ptr[i];
            dot += v * v;
            float abs_val = std::fabs(v);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        if (max_abs == 0.0f || dot <= 0.0f || !std::isfinite(dot) || !std::isfinite(max_abs)) {
            std::fill(dst.begin(), dst.end(), 0);
            if (dst_abs) std::fill_n(dst_abs, size, uint8_t{0});
            if (dst_u8) std::fill_n(dst_u8, size, uint8_t{128});
            if (out_sum_128) *out_sum_128 = 0;
            return 0.0f;
        }

        const float inv_norm = 1.0f / std::sqrt(dot);
        const float scale = (max_abs * inv_norm) / 127.0f;
        const float inv_scale = 127.0f / max_abs;
        i = 0;

#if defined(__AVX2__)
        const __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        const __m256i offset_128 = _mm256_set1_epi8(static_cast<char>(0x80));

        for (; i + 31 < size; i += 32) {
            __m256i i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i),      v_inv_scale));
            __m256i i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 8),  v_inv_scale));
            __m256i i32_2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 16), v_inv_scale));
            __m256i i32_3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src_ptr + i + 24), v_inv_scale));

            __m256i i16_0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32_0, i32_1), 0xD8);
            __m256i i16_1 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32_2, i32_3), 0xD8);
            __m256i i8_all = _mm256_permute4x64_epi64(_mm256_packs_epi16(i16_0, i16_1), 0xD8);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_ptr + i), i8_all);
            if (dst_abs) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_abs + i), _mm256_abs_epi8(i8_all));
            }
            if (dst_u8) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_u8 + i), _mm256_xor_si256(i8_all, offset_128));
            }
        }
#endif

        for (; i < size; ++i) {
            int val = static_cast<int>(std::nearbyint(src_ptr[i] * inv_scale));
            int8_t qv = static_cast<int8_t>(std::clamp(val, -127, 127));
            dst_ptr[i] = qv;
            if (dst_abs) {
                dst_abs[i] = static_cast<uint8_t>(qv < 0 ? -qv : qv);
            }
            if (dst_u8) {
                dst_u8[i] = static_cast<uint8_t>(static_cast<int>(qv) + 128);
            }
        }

        if (out_sum_128) {
            int32_t s = 0;
            for (std::size_t k = 0; k < size; ++k) {
                s += static_cast<int32_t>(dst_ptr[k]);
            }
            *out_sum_128 = s * 128;
        }

        return scale;
    }
};

} // namespace vrag
