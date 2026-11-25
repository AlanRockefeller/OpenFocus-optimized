#include "simd_dispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define OPENFOCUS_X86_64 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define OPENFOCUS_X86_64 0
#endif

namespace openfocus {
namespace simd {
namespace {

std::atomic<bool> g_force_scalar{false};

struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
};

CpuFeatures detect_cpu_features() {
#if OPENFOCUS_X86_64
    int info1[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(info1, 0);
    const int max_leaf = info1[0];
    if (max_leaf < 7) {
        return {};
    }
    __cpuidex(info1, 1, 0);
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) || eax < 7) {
        return {};
    }
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    info1[0] = static_cast<int>(eax);
    info1[1] = static_cast<int>(ebx);
    info1[2] = static_cast<int>(ecx);
    info1[3] = static_cast<int>(edx);
#endif
    const uint32_t ecx1 = static_cast<uint32_t>(info1[2]);
    // Bit 27 = OSXSAVE, Bit 28 = AVX
    if ((ecx1 & (1u << 27)) == 0 || (ecx1 & (1u << 28)) == 0) {
        return {};
    }
    uint64_t xcr0 = 0;
#if defined(_MSC_VER)
    xcr0 = _xgetbv(0);
#else
    uint32_t xlo = 0, xhi = 0;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    xcr0 = (static_cast<uint64_t>(xhi) << 32) | xlo;
#endif
    // XMM (bit 1) + YMM (bit 2) state enabled by OS
    if ((xcr0 & 0x6ull) != 0x6ull) {
        return {};
    }
    uint32_t ebx7 = 0;
#if defined(_MSC_VER)
    int info7[4] = {0, 0, 0, 0};
    __cpuidex(info7, 7, 0);
    ebx7 = static_cast<uint32_t>(info7[1]);
#else
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    ebx7 = ebx;
#endif
    // Leaf 7 EBX bit 5 = AVX2, Leaf 1 ECX bit 12 = FMA3
    const bool has_avx2 = (ebx7 & (1u << 5)) != 0;
    const bool has_fma = has_avx2 && ((ecx1 & (1u << 12)) != 0);
    return {has_avx2, has_fma};
#else
    return {};
#endif
}

const CpuFeatures& cpu_features() {
    static const CpuFeatures kFeatures = detect_cpu_features();
    return kFeatures;
}

#if OPENFOCUS_X86_64
inline uint16_t encode_weight_scalar(float strength, bool is_border, float w_scale) {
    if (is_border) {
        return 0;
    }
    const float scaled = strength * w_scale;
    if (scaled >= 65535.0f) {
        return 65535;
    }
    if (scaled <= 0.0f) {
        return 0;
    }
    return static_cast<uint16_t>(static_cast<int32_t>(scaled));
}
#endif

}  // namespace

bool use_avx2() {
    return !g_force_scalar.load(std::memory_order_relaxed) && cpu_features().avx2;
}

bool use_avx2_fma() {
    return !g_force_scalar.load(std::memory_order_relaxed) && cpu_features().fma;
}

void set_force_scalar(bool force) {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

std::string simd_info() {
    if (use_avx2_fma()) {
        return "AVX2+FMA";
    }
    return use_avx2() ? "AVX2" : "Scalar";
}

#if OPENFOCUS_X86_64

namespace {

// (a[3k] + a[3k+1]) + a[3k+2] for eight packed 3-channel pixels held in three
// vectors. Each channel takes disjoint lanes from the three inputs, so two
// blends and one permute separate it.
inline __m256 packed3_pixel_sums(__m256 v0, __m256 v1, __m256 v2) {
    const __m256 c0 = _mm256_permutevar8x32_ps(
        _mm256_blend_ps(_mm256_blend_ps(v0, v1, 0x92), v2, 0x24),
        _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5));
    const __m256 c1 = _mm256_permutevar8x32_ps(
        _mm256_blend_ps(_mm256_blend_ps(v0, v1, 0x24), v2, 0x49),
        _mm256_setr_epi32(1, 4, 7, 2, 5, 0, 3, 6));
    const __m256 c2 = _mm256_permutevar8x32_ps(
        _mm256_blend_ps(_mm256_blend_ps(v0, v1, 0x49), v2, 0x92),
        _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7));
    return _mm256_add_ps(_mm256_add_ps(c0, c1), c2);
}

// encode_weight_scalar for eight interior pixels.
inline void store_encoded_weights_x8(__m256 strength, __m256 v_scale, uint16_t* dst) {
    const __m256 scaled = _mm256_mul_ps(strength, v_scale);
    const __m256 high = _mm256_cmp_ps(scaled, _mm256_set1_ps(65535.0f), _CMP_GE_OQ);
    const __m256 low = _mm256_cmp_ps(scaled, _mm256_setzero_ps(), _CMP_LE_OQ);
    __m256i q = _mm256_cvttps_epi32(scaled);
    q = _mm256_blendv_epi8(q, _mm256_set1_epi32(65535), _mm256_castps_si256(high));
    q = _mm256_andnot_si256(_mm256_castps_si256(low), q);
    // A NaN converts to INT_MIN, which the signed saturating pack maps to 0,
    // the same value its truncation to uint16 gives.
    const __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(q, q), 0xD8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(p));
}

// Border pixels carry no weight: the whole first/last row, and both row ends.
inline void clear_border_weights(uint16_t* w_row, int64_t width, bool y_border) {
    if (y_border) {
        std::memset(w_row, 0, static_cast<size_t>(width) * sizeof(uint16_t));
    } else if (width > 2) {
        w_row[0] = 0;
        w_row[width - 1] = 0;
    }
}

}  // namespace

void pyr_down_vertical_row_f32_avx2(
    const float* p0, const float* p1, const float* p2, const float* p3,
    float* dst, int64_t elems, float d0, float d1
) {
    const __m256 vd0 = _mm256_set1_ps(d0);
    const __m256 vd1 = _mm256_set1_ps(d1);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(p0 + i), vd0);
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p1 + i), vd1));
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p2 + i), vd1));
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p3 + i), vd0));
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < elems; ++i) {
        float v = p0[i] * d0;
        v += p1[i] * d1;
        v += p2[i] * d1;
        dst[i] = v + p3[i] * d0;
    }
}

void pyr_reconstruct_row_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* tr, int64_t elems,
    float w0, float w1, float w2, float w3, float w4, float point_one
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 lo = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(hr0 + i), vw0),
                _mm256_mul_ps(_mm256_loadu_ps(hr1 + i), vw1)),
            _mm256_mul_ps(_mm256_loadu_ps(hr2 + i), vw2));
        lo = _mm256_add_ps(
            _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + i), vw3)),
            _mm256_mul_ps(_mm256_loadu_ps(hr4 + i), vw4));
        const __m256 t = _mm256_loadu_ps(tr + i);
        __m256 val = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(lo, vp1), t), lo);
        const __m256 lt0 = _mm256_cmp_ps(val, vzero, _CMP_LT_OQ);
        val = _mm256_blendv_ps(val, vzero, lt0);
        const __m256 gt1 = _mm256_cmp_ps(val, vone, _CMP_GT_OQ);
        val = _mm256_blendv_ps(val, vone, gt1);
        _mm256_storeu_ps(tr + i, val);
    }
    for (; i < elems; ++i) {
        float lo = (hr0[i] * w0 + hr1[i] * w1) + hr2[i] * w2;
        lo = (lo + hr3[i] * w3) + hr4[i] * w4;
        const float val = (lo + point_one) * tr[i] + lo;
        tr[i] = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
    }
}

void pyr_detail_weight_3c_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* target_row, uint16_t* w_row, int64_t width,
    float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    const __m256 v_scale = _mm256_set1_ps(w_scale);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const int64_t off = x * 3;
        __m256 av[3];
        for (int blk = 0; blk < 3; ++blk) {
            const int64_t b = off + blk * 8;
            __m256 lo = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(hr0 + b), vw0),
                    _mm256_mul_ps(_mm256_loadu_ps(hr1 + b), vw1)),
                _mm256_mul_ps(_mm256_loadu_ps(hr2 + b), vw2));
            lo = _mm256_add_ps(
                _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + b), vw3)),
                _mm256_mul_ps(_mm256_loadu_ps(hr4 + b), vw4));
            const __m256 s = _mm256_loadu_ps(target_row + b);
            const __m256 v = _mm256_div_ps(_mm256_sub_ps(s, lo), _mm256_add_ps(lo, vp1));
            _mm256_storeu_ps(target_row + b, v);
            av[blk] = _mm256_and_ps(v, vabs);
        }
        store_encoded_weights_x8(packed3_pixel_sums(av[0], av[1], av[2]), v_scale, w_row + x);
    }
    for (; x < width; ++x) {
        const int64_t off3 = x * 3;
        float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
        float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
        float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
        lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
        lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
        lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
        const float v0 = (target_row[off3] - lo0) / (lo0 + point_one);
        const float v1 = (target_row[off3 + 1] - lo1) / (lo1 + point_one);
        const float v2 = (target_row[off3 + 2] - lo2) / (lo2 + point_one);
        target_row[off3] = v0;
        target_row[off3 + 1] = v1;
        target_row[off3 + 2] = v2;
        const bool is_border = y_border || (width > 2 && (x == 0 || x == width - 1));
        const float strength = (std::abs(v0) + std::abs(v1)) + std::abs(v2);
        w_row[x] = encode_weight_scalar(strength, is_border, w_scale);
    }
    clear_border_weights(w_row, width, y_border);
}

void pyr_level0_weights_3c_f32_from_raw_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, uint16_t* w_row, int64_t width,
    float factor, float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vfac = _mm256_set1_ps(factor);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    const uint8_t* r8 = static_cast<const uint8_t*>(raw_row);
    const uint16_t* r16 = static_cast<const uint16_t*>(raw_row);

    if (y_border) {
        // Nothing but the weights is produced here, and a border row has none.
        clear_border_weights(w_row, width, true);
        return;
    }
    const __m256 v_scale = _mm256_set1_ps(w_scale);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const int64_t off = x * 3;
        __m256 av[3];
        for (int blk = 0; blk < 3; ++blk) {
            const int64_t b = off + blk * 8;
            __m256 lo = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(hr0 + b), vw0),
                    _mm256_mul_ps(_mm256_loadu_ps(hr1 + b), vw1)),
                _mm256_mul_ps(_mm256_loadu_ps(hr2 + b), vw2));
            lo = _mm256_add_ps(
                _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + b), vw3)),
                _mm256_mul_ps(_mm256_loadu_ps(hr4 + b), vw4));
            __m256i raw_i;
            if (raw_is_u8) {
                raw_i = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(r8 + b)));
            } else {
                raw_i = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r16 + b)));
            }
            const __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_i), vfac);
            const __m256 v = _mm256_div_ps(_mm256_sub_ps(s, lo), _mm256_add_ps(lo, vp1));
            av[blk] = _mm256_and_ps(v, vabs);
        }
        store_encoded_weights_x8(packed3_pixel_sums(av[0], av[1], av[2]), v_scale, w_row + x);
    }
    for (; x < width; ++x) {
        const int64_t off3 = x * 3;
        float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
        float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
        float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
        lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
        lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
        lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
        float s0, s1, s2;
        if (raw_is_u8) {
            s0 = static_cast<float>(r8[off3]) * factor;
            s1 = static_cast<float>(r8[off3 + 1]) * factor;
            s2 = static_cast<float>(r8[off3 + 2]) * factor;
        } else {
            s0 = static_cast<float>(r16[off3]) * factor;
            s1 = static_cast<float>(r16[off3 + 1]) * factor;
            s2 = static_cast<float>(r16[off3 + 2]) * factor;
        }
        const float v0 = (s0 - lo0) / (lo0 + point_one);
        const float v1 = (s1 - lo1) / (lo1 + point_one);
        const float v2 = (s2 - lo2) / (lo2 + point_one);
        const bool is_border = width > 2 && (x == 0 || x == width - 1);
        const float strength = (std::abs(v0) + std::abs(v1)) + std::abs(v2);
        w_row[x] = encode_weight_scalar(strength, is_border, w_scale);
    }
    clear_border_weights(w_row, width, false);
}

void depth_axis_iir_y_unguided_avx2(
    const uint16_t* src, int64_t src_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t q, uint32_t t, uint16_t* first, uint32_t* acc
) {
    const uint32_t t16 = t * 16u;
    const uint32_t t_plus_q = t + q;
    std::fill(acc, acc + cols, 0u);
    for (int64_t y = 0; y < cnt; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        for (int64_t c = 0; c < cols; ++c) {
            acc[c] += static_cast<uint32_t>(sr[c]);
        }
    }
    for (int64_t c = 0; c < cols; ++c) {
        const uint64_t init = ((static_cast<uint64_t>(acc[c]) << 4u) / static_cast<uint64_t>(cnt)) * t_plus_q;
        acc[c] = static_cast<uint32_t>(init & 0xffffffffu);
    }

    const __m256i v_t16 = _mm256_set1_epi32(static_cast<int32_t>(t16));
    const __m256i v_q = _mm256_set1_epi32(static_cast<int32_t>(q));
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);
    const int64_t vec_cols = cols & ~int64_t{7};

    for (int64_t y = 0; y < h; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        uint16_t* fr = first + y * cols;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i sv = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i inn = _mm256_mullo_epi32(sv, v_t16);
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i st = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, v_2048), inn), 12);
            const __m256i fv = _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(fv, fv), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(fr + c), _mm256_castsi256_si128(p16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),
                                _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t inn = static_cast<uint32_t>(sr[c]) * t16;
            const uint32_t st = (acc[c] + 2048u + inn) >> 12u;
            fr[c] = static_cast<uint16_t>((st + 4u) >> 4u);
            acc[c] = st * q + inn;
        }
    }

    const uint16_t* last_fr = first + (h - 1) * cols;
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = (static_cast<uint32_t>(last_fr[c]) << 4u) * t_plus_q;
    }

    for (int64_t y = h - 1; y >= 0; --y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* fr = first + y * cols;
        uint16_t* orow = out + y * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i sv = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i f32 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fr + c)));
            const __m256i inn = _mm256_mullo_epi32(sv, v_t16);
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i st = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, v_2048), inn), 12);
            const __m256i term = _mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4);
            const __m256i ov = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(term, v_1), f32), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),
                                _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t inn = static_cast<uint32_t>(sr[c]) * t16;
            const uint32_t st = (acc[c] + 2048u + inn) >> 12u;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + static_cast<uint32_t>(fr[c])) >> 1u);
            acc[c] = st * q + inn;
        }
    }
}

void depth_axis_iir_y_guided_avx2(
    const uint16_t* src, int64_t src_sy,
    const uint16_t* guide, int64_t guide_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t tm, uint32_t tl, uint32_t* first, uint32_t* acc
) {
    std::fill(acc, acc + cols, 0u);
    for (int64_t y = 0; y < cnt; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        for (int64_t c = 0; c < cols; ++c) {
            acc[c] += static_cast<uint32_t>(sr[c]);
        }
    }
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = static_cast<uint32_t>((static_cast<uint64_t>(acc[c]) << 4u) / static_cast<uint64_t>(cnt));
    }

    const __m256i v_65536 = _mm256_set1_epi32(65536);
    const __m256i v_4096 = _mm256_set1_epi32(4096);
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_tl = _mm256_set1_epi32(static_cast<int32_t>(tl));
    const __m256i v_tm = _mm256_set1_epi32(static_cast<int32_t>(tm));
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);
    const int64_t vec_cols = cols & ~int64_t{7};

    // Forward y = 0
    {
        const uint16_t* sr0 = src + left;
        const uint16_t* gr0 = guide + left;
        uint32_t* fr0 = first;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr0 + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr0 + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, v), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c), st);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(fr0 + c),
                                _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr0[c];
            const uint32_t s = gr0[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + v) * (amt << 4u)) >> 12u;
            acc[c] = st;
            fr0[c] = (st + 4u) >> 4u;
        }
    }

    for (int64_t y = 1; y < h; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* pr = src + (y - 1) * src_sy + left;
        const uint16_t* gr = guide + y * guide_sy + left;
        uint32_t* fr = first + y * cols;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pr + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c), st);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(fr + c),
                                _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = pr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            fr[c] = (st + 4u) >> 4u;
        }
    }

    const uint32_t* last_fr = first + (h - 1) * cols;
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = last_fr[c] << 4u;
    }

    // Backward y = h - 1
    {
        const int64_t yt = h - 1;
        const uint16_t* sr = src + yt * src_sy + left;
        const uint16_t* gr = guide + yt * guide_sy + left;
        const uint32_t* fr = first + yt * cols;
        uint16_t* orow = out + yt * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fr + c));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c), st);
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), p), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = fr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + p) >> 1u);
        }
    }

    for (int64_t y = h - 2; y >= 0; --y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* pr = src + (y + 1) * src_sy + left;
        const uint16_t* gr = guide + y * guide_sy + left;
        const uint32_t* fr = first + y * cols;
        uint16_t* orow = out + y * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pr + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fr + c));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c), st);
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), f), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = pr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + fr[c]) >> 1u);
        }
    }
}

namespace {

// 8x8 transpose of uint16 samples. On entry r[k] holds eight consecutive
// samples of row k; on exit r[k] holds column k (one sample from each row).
// The mapping is its own inverse.
inline void transpose8x8_u16(__m128i r[8]) {
    const __m128i t0 = _mm_unpacklo_epi16(r[0], r[1]);
    const __m128i t1 = _mm_unpackhi_epi16(r[0], r[1]);
    const __m128i t2 = _mm_unpacklo_epi16(r[2], r[3]);
    const __m128i t3 = _mm_unpackhi_epi16(r[2], r[3]);
    const __m128i t4 = _mm_unpacklo_epi16(r[4], r[5]);
    const __m128i t5 = _mm_unpackhi_epi16(r[4], r[5]);
    const __m128i t6 = _mm_unpacklo_epi16(r[6], r[7]);
    const __m128i t7 = _mm_unpackhi_epi16(r[6], r[7]);
    const __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    const __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    const __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    const __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    const __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    const __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    const __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    const __m128i u7 = _mm_unpackhi_epi32(t5, t7);
    r[0] = _mm_unpacklo_epi64(u0, u4);
    r[1] = _mm_unpackhi_epi64(u0, u4);
    r[2] = _mm_unpacklo_epi64(u1, u5);
    r[3] = _mm_unpackhi_epi64(u1, u5);
    r[4] = _mm_unpacklo_epi64(u2, u6);
    r[5] = _mm_unpackhi_epi64(u2, u6);
    r[6] = _mm_unpacklo_epi64(u3, u7);
    r[7] = _mm_unpackhi_epi64(u3, u7);
}

// Column-major copy of eight rows: cols[x * 8 + k] = rows[k * stride + x].
inline void pack_rows8_u16(const uint16_t* rows, int64_t stride, int64_t w, uint16_t* cols) {
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        __m128i r[8];
        for (int k = 0; k < 8; ++k) {
            r[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rows + k * stride + x));
        }
        transpose8x8_u16(r);
        for (int k = 0; k < 8; ++k) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(cols + (x + k) * 8), r[k]);
        }
    }
    for (; x < w; ++x) {
        for (int k = 0; k < 8; ++k) {
            cols[x * 8 + k] = rows[k * stride + x];
        }
    }
}

// Inverse of pack_rows8_u16: rows[k * stride + x] = cols[x * 8 + k].
inline void unpack_rows8_u16(const uint16_t* cols, int64_t w, uint16_t* rows, int64_t stride) {
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        __m128i r[8];
        for (int k = 0; k < 8; ++k) {
            r[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cols + (x + k) * 8));
        }
        transpose8x8_u16(r);
        for (int k = 0; k < 8; ++k) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(rows + k * stride + x), r[k]);
        }
    }
    for (; x < w; ++x) {
        for (int k = 0; k < 8; ++k) {
            rows[k * stride + x] = cols[x * 8 + k];
        }
    }
}

// Truncating uint32 -> uint16 narrowing, matching static_cast<uint16_t>.
inline __m128i narrow_u32_to_u16(__m256i v) {
    const __m256i shuf = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_shuffle_epi8(v, shuf), 0x08));
}

inline __m256i load_u16x8(const uint16_t* p) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// The recurrence runs along x, so each vector lane is one row. G groups of
// eight rows advance together; their dependency chains are independent, which
// hides the latency of the 32-bit multiply in the feedback path.
template <int G>
void depth_iir_x_unguided_groups(
    const uint16_t* src, int64_t src_sy, uint16_t* out, int64_t out_sy,
    int64_t w, int64_t cnt, uint32_t q, uint32_t t, uint16_t* work
) {
    const uint32_t t16 = t * 16u;
    const uint32_t t_plus_q = t + q;
    const __m256i v_t16 = _mm256_set1_epi32(static_cast<int32_t>(t16));
    const __m256i v_q = _mm256_set1_epi32(static_cast<int32_t>(q));
    const __m256i v_tq = _mm256_set1_epi32(static_cast<int32_t>(t_plus_q));
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);

    uint16_t* s_cols[G];
    uint16_t* f_cols[G];
    __m256i acc[G];
    for (int g = 0; g < G; ++g) {
        s_cols[g] = work + (2 * g) * w * 8;
        f_cols[g] = work + (2 * g + 1) * w * 8;
        const uint16_t* rows = src + (8 * g) * src_sy;
        pack_rows8_u16(rows, src_sy, w, s_cols[g]);
        alignas(32) uint32_t init[8];
        for (int k = 0; k < 8; ++k) {
            const uint16_t* sr = rows + k * src_sy;
            uint64_t sum = 0;
            for (int64_t x = 0; x < cnt; ++x) {
                sum += sr[x];
            }
            init[k] = static_cast<uint32_t>((((sum << 4u) / static_cast<uint64_t>(cnt)) * t_plus_q) & 0xffffffffu);
        }
        acc[g] = _mm256_load_si256(reinterpret_cast<const __m256i*>(init));
    }

    for (int64_t x = 0; x < w; ++x) {
        for (int g = 0; g < G; ++g) {
            const __m256i inn = _mm256_mullo_epi32(load_u16x8(s_cols[g] + x * 8), v_t16);
            const __m256i st = _mm256_srli_epi32(
                _mm256_add_epi32(acc[g], _mm256_add_epi32(inn, v_2048)), 12);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(f_cols[g] + x * 8),
                             narrow_u32_to_u16(_mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4)));
            acc[g] = _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn);
        }
    }

    for (int g = 0; g < G; ++g) {
        acc[g] = _mm256_mullo_epi32(_mm256_slli_epi32(load_u16x8(f_cols[g] + (w - 1) * 8), 4), v_tq);
    }

    for (int64_t x = w - 1; x >= 0; --x) {
        for (int g = 0; g < G; ++g) {
            const __m256i inn = _mm256_mullo_epi32(load_u16x8(s_cols[g] + x * 8), v_t16);
            const __m256i st = _mm256_srli_epi32(
                _mm256_add_epi32(acc[g], _mm256_add_epi32(inn, v_2048)), 12);
            const __m256i fv = load_u16x8(f_cols[g] + x * 8);
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), fv), 1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(f_cols[g] + x * 8), narrow_u32_to_u16(ov));
            acc[g] = _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn);
        }
    }

    for (int g = 0; g < G; ++g) {
        unpack_rows8_u16(f_cols[g], w, out + (8 * g) * out_sy, out_sy);
    }
}

template <int G>
void depth_iir_x_guided_groups(
    const uint16_t* src, int64_t src_sy, const uint16_t* guide, int64_t guide_sy,
    uint16_t* out, int64_t out_sy,
    int64_t w, int64_t cnt, uint32_t tm, uint32_t tl, uint16_t* work
) {
    const __m256i v_65536 = _mm256_set1_epi32(65536);
    const __m256i v_4096 = _mm256_set1_epi32(4096);
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_tl = _mm256_set1_epi32(static_cast<int32_t>(tl));
    const __m256i v_tm = _mm256_set1_epi32(static_cast<int32_t>(tm));
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);

    uint16_t* s_cols[G];
    uint16_t* g_cols[G];
    uint16_t* f_cols[G];
    __m256i acc[G];
    __m256i neighbour[G];
    for (int g = 0; g < G; ++g) {
        s_cols[g] = work + (3 * g) * w * 8;
        g_cols[g] = work + (3 * g + 1) * w * 8;
        f_cols[g] = work + (3 * g + 2) * w * 8;
        const uint16_t* rows = src + (8 * g) * src_sy;
        pack_rows8_u16(rows, src_sy, w, s_cols[g]);
        pack_rows8_u16(guide + (8 * g) * guide_sy, guide_sy, w, g_cols[g]);
        alignas(32) uint32_t init[8];
        for (int k = 0; k < 8; ++k) {
            const uint16_t* sr = rows + k * src_sy;
            uint64_t sum = 0;
            for (int64_t x = 0; x < cnt; ++x) {
                sum += sr[x];
            }
            init[k] = static_cast<uint32_t>((sum << 4u) / static_cast<uint64_t>(cnt));
        }
        acc[g] = _mm256_load_si256(reinterpret_cast<const __m256i*>(init));
        // The first sample is its own left neighbour.
        neighbour[g] = load_u16x8(s_cols[g]);
    }

    for (int64_t x = 0; x < w; ++x) {
        for (int g = 0; g < G; ++g) {
            const __m256i v = load_u16x8(s_cols[g] + x * 8);
            const __m256i s = load_u16x8(g_cols[g] + x * 8);
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i input = _mm256_add_epi32(
                _mm256_mullo_epi32(_mm256_add_epi32(v, neighbour[g]), _mm256_slli_epi32(amt, 4)), v_2048);
            const __m256i st = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, acc[g]), input), 12);
            acc[g] = st;
            neighbour[g] = v;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(f_cols[g] + x * 8),
                             narrow_u32_to_u16(_mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4)));
        }
    }

    for (int g = 0; g < G; ++g) {
        // The last sample pairs with its own forward result.
        neighbour[g] = load_u16x8(f_cols[g] + (w - 1) * 8);
        acc[g] = _mm256_slli_epi32(neighbour[g], 4);
    }

    for (int64_t x = w - 1; x >= 0; --x) {
        for (int g = 0; g < G; ++g) {
            const __m256i v = load_u16x8(s_cols[g] + x * 8);
            const __m256i s = load_u16x8(g_cols[g] + x * 8);
            const __m256i fv = load_u16x8(f_cols[g] + x * 8);
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i input = _mm256_add_epi32(
                _mm256_mullo_epi32(_mm256_add_epi32(v, neighbour[g]), _mm256_slli_epi32(amt, 4)), v_2048);
            const __m256i st = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, acc[g]), input), 12);
            acc[g] = st;
            neighbour[g] = v;
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), fv), 1);
            // The guide column is no longer needed; reuse it for the output.
            _mm_storeu_si128(reinterpret_cast<__m128i*>(g_cols[g] + x * 8), narrow_u32_to_u16(ov));
        }
    }

    for (int g = 0; g < G; ++g) {
        unpack_rows8_u16(g_cols[g], w, out + (8 * g) * out_sy, out_sy);
    }
}

}  // namespace

void depth_axis_iir_x_unguided_avx2(
    const uint16_t* src, int64_t src_sy, uint16_t* out, int64_t out_sy,
    int64_t w, int64_t rows, int64_t cnt, uint32_t q, uint32_t t, uint16_t* work
) {
    int64_t y = 0;
    for (; y + 32 <= rows; y += 32) {
        depth_iir_x_unguided_groups<4>(src + y * src_sy, src_sy, out + y * out_sy, out_sy, w, cnt, q, t, work);
    }
    if (y + 16 <= rows) {
        depth_iir_x_unguided_groups<2>(src + y * src_sy, src_sy, out + y * out_sy, out_sy, w, cnt, q, t, work);
        y += 16;
    }
    if (y + 8 <= rows) {
        depth_iir_x_unguided_groups<1>(src + y * src_sy, src_sy, out + y * out_sy, out_sy, w, cnt, q, t, work);
    }
}

void depth_axis_iir_x_guided_avx2(
    const uint16_t* src, int64_t src_sy, const uint16_t* guide, int64_t guide_sy,
    uint16_t* out, int64_t out_sy,
    int64_t w, int64_t rows, int64_t cnt, uint32_t tm, uint32_t tl, uint16_t* work
) {
    int64_t y = 0;
    for (; y + 16 <= rows; y += 16) {
        depth_iir_x_guided_groups<2>(src + y * src_sy, src_sy, guide + y * guide_sy, guide_sy,
                                     out + y * out_sy, out_sy, w, cnt, tm, tl, work);
    }
    if (y + 8 <= rows) {
        depth_iir_x_guided_groups<1>(src + y * src_sy, src_sy, guide + y * guide_sy, guide_sy,
                                     out + y * out_sy, out_sy, w, cnt, tm, tl, work);
    }
}

void select_a_combine_m2_to_m5_avx2(
    const uint32_t* da2, const uint32_t* db2, uint32_t* m2, uint32_t wy2, uint32_t fy2,
    const uint32_t* da3, const uint32_t* db3, uint32_t* m3, uint32_t wy3, uint32_t fy3,
    const uint32_t* da4, const uint32_t* db4, uint32_t* m4, uint32_t wy4, uint32_t fy4,
    const uint32_t* da5, const uint32_t* db5, uint32_t* m5, uint32_t wy5, uint32_t fy5,
    int64_t n_out
) {
    const __m256i vwy2 = _mm256_set1_epi32(static_cast<int32_t>(wy2));
    const __m256i vfy2 = _mm256_set1_epi32(static_cast<int32_t>(fy2));
    const __m256i vwy3 = _mm256_set1_epi32(static_cast<int32_t>(wy3));
    const __m256i vfy3 = _mm256_set1_epi32(static_cast<int32_t>(fy3));
    const __m256i vwy4 = _mm256_set1_epi32(static_cast<int32_t>(wy4));
    const __m256i vfy4 = _mm256_set1_epi32(static_cast<int32_t>(fy4));
    const __m256i vwy5 = _mm256_set1_epi32(static_cast<int32_t>(wy5));
    const __m256i vfy5 = _mm256_set1_epi32(static_cast<int32_t>(fy5));
    int64_t j = 0;
    for (; j + 8 <= n_out; j += 8) {
        const auto step = [j](const uint32_t* da, const uint32_t* db, uint32_t* dst, __m256i wy, __m256i fy) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(da + j));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db + j));
            const __m256i r = _mm256_add_epi32(_mm256_mullo_epi32(a, wy), _mm256_mullo_epi32(b, fy));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + j), r);
        };
        step(da2, db2, m2, vwy2, vfy2);
        step(da3, db3, m3, vwy3, vfy3);
        step(da4, db4, m4, vwy4, vfy4);
        step(da5, db5, m5, vwy5, vfy5);
    }
    for (; j < n_out; ++j) {
        m2[j] = da2[j] * wy2 + db2[j] * fy2;
        m3[j] = da3[j] * wy3 + db3[j] * fy3;
        m4[j] = da4[j] * wy4 + db4[j] * fy4;
        m5[j] = da5[j] * wy5 + db5[j] * fy5;
    }
}

void select_a_response_pairs_avx2(
    const void* g_row, bool g_is_u16,
    const uint32_t* m0, const uint32_t* m1,
    const uint32_t* m2, const uint32_t* m3,
    const uint32_t* m4, const uint32_t* m5,
    int32_t* response, int64_t full_pairs
) {
    const uint16_t* g16 = static_cast<const uint16_t*>(g_row);
    const int32_t* g32 = static_cast<const int32_t*>(g_row);
    const __m256i v3 = _mm256_set1_epi32(3);
    const __m256i v8 = _mm256_set1_epi32(8);
    const __m256i v32 = _mm256_set1_epi32(32);
    const __m256i v128 = _mm256_set1_epi32(128);
    const __m256i v512 = _mm256_set1_epi32(512);
    const __m256i v2048 = _mm256_set1_epi32(2048);
    const __m256i v_mask16 = _mm256_set1_epi32(0x0000FFFF);

    int64_t j = 0;
    for (; j + 8 <= full_pairs; j += 8) {
        const int64_t ev = 2 * j;
        __m256i ge, go;
        if (g_is_u16) {
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g16 + ev));
            ge = _mm256_and_si256(raw, v_mask16);
            go = _mm256_srli_epi32(raw, 16);
        } else {
            const __m256 p0 = _mm256_loadu_ps(reinterpret_cast<const float*>(g32 + ev));
            const __m256 p1 = _mm256_loadu_ps(reinterpret_cast<const float*>(g32 + ev + 8));
            ge = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0x88)), 0xD8);
            go = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0xDD)), 0xD8);
        }
        const __m256i a0l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j));
        const __m256i a0c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j + 1));
        const __m256i a0r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j + 2));
        const __m256i a0c3 = _mm256_mullo_epi32(a0c, v3);
        const __m256i d0 = _mm256_sub_epi32(_mm256_slli_epi32(ge, 4), _mm256_add_epi32(a0l, a0c3));
        const __m256i d1 = _mm256_sub_epi32(_mm256_slli_epi32(go, 4), _mm256_add_epi32(a0c3, a0r));
        __m256i s0 = _mm256_abs_epi32(d0);
        __m256i s1 = _mm256_abs_epi32(d1);

        const __m256i a1l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j));
        const __m256i a1c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j + 1));
        const __m256i a1r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j + 2));
        const __m256i a1c3 = _mm256_mullo_epi32(a1c, v3);
        s0 = _mm256_add_epi32(s0, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a1l, a1c3), v8), 4));
        s1 = _mm256_add_epi32(s1, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a1c3, a1r), v8), 4));

        const auto accum = [&](const uint32_t* m, __m256i bias, int shift) {
            const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + j));
            const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + j + 1));
            const __m256i l3 = _mm256_mullo_epi32(l, v3);
            const __m256i r3 = _mm256_mullo_epi32(r, v3);
            s0 = _mm256_add_epi32(s0, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(l3, r), bias), shift));
            s1 = _mm256_add_epi32(s1, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(l, r3), bias), shift));
        };
        accum(m2, v32, 6);
        accum(m3, v128, 8);
        accum(m4, v512, 10);
        accum(m5, v2048, 12);

        const __m256i u0 = _mm256_unpacklo_epi32(s0, s1);
        const __m256i u1 = _mm256_unpackhi_epi32(s0, s1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(response + ev), _mm256_permute2x128_si256(u0, u1, 0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(response + ev + 8), _mm256_permute2x128_si256(u0, u1, 0x31));
    }
    for (; j < full_pairs; ++j) {
        const int64_t ev = 2 * j;
        const uint32_t gev = g_is_u16 ? static_cast<uint32_t>(g16[ev]) : static_cast<uint32_t>(g32[ev]);
        const uint32_t god = g_is_u16 ? static_cast<uint32_t>(g16[ev + 1]) : static_cast<uint32_t>(g32[ev + 1]);
        const uint32_t a0_l = m0[j], a0_c = m0[j + 1], a0_r = m0[j + 2];
        const uint32_t a1_l = m1[j], a1_c = m1[j + 1], a1_r = m1[j + 2];
        const int32_t d0 = static_cast<int32_t>(gev << 4u) - static_cast<int32_t>(a0_l + a0_c * 3u);
        const int32_t d1 = static_cast<int32_t>(god << 4u) - static_cast<int32_t>(a0_c * 3u + a0_r);
        int32_t s0 = (d0 < 0) ? -d0 : d0;
        int32_t s1 = (d1 < 0) ? -d1 : d1;
        s0 += static_cast<int32_t>((a1_l + a1_c * 3u + 8u) >> 4u);
        s1 += static_cast<int32_t>((a1_c * 3u + a1_r + 8u) >> 4u);
        s0 += static_cast<int32_t>((m2[j] * 3u + m2[j + 1] + 32u) >> 6u);
        s1 += static_cast<int32_t>((m2[j] + m2[j + 1] * 3u + 32u) >> 6u);
        s0 += static_cast<int32_t>((m3[j] * 3u + m3[j + 1] + 128u) >> 8u);
        s1 += static_cast<int32_t>((m3[j] + m3[j + 1] * 3u + 128u) >> 8u);
        s0 += static_cast<int32_t>((m4[j] * 3u + m4[j + 1] + 512u) >> 10u);
        s1 += static_cast<int32_t>((m4[j] + m4[j + 1] * 3u + 512u) >> 10u);
        s0 += static_cast<int32_t>((m5[j] * 3u + m5[j + 1] + 2048u) >> 12u);
        s1 += static_cast<int32_t>((m5[j] + m5[j + 1] * 3u + 2048u) >> 12u);
        response[ev] = s0;
        response[ev + 1] = s1;
    }
}

uint64_t sum_u8_row_avx2(const uint8_t* p, int64_t n) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc = zero;
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(v, zero));
    }
    alignas(32) uint64_t parts[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(parts), acc);
    uint64_t total = parts[0] + parts[1] + parts[2] + parts[3];
    for (; i < n; ++i) {
        total += p[i];
    }
    return total;
}

uint64_t sum_u16_row_avx2(const uint16_t* p, int64_t n) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi16(0x00FF);
    __m256i acc_lo = zero;
    __m256i acc_hi = zero;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_sad_epu8(_mm256_and_si256(v, low_mask), zero));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_sad_epu8(_mm256_srli_epi16(v, 8), zero));
    }
    alignas(32) uint64_t lo[4], hi[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lo), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(hi), acc_hi);
    uint64_t total = (lo[0] + lo[1] + lo[2] + lo[3]) + ((hi[0] + hi[1] + hi[2] + hi[3]) << 8u);
    for (; i < n; ++i) {
        total += p[i];
    }
    return total;
}

bool sum_u16_normalized_row_avx2(const uint16_t* p, int64_t n, double* out) {
    // Each term float(v) * (255 / 65535) is a float32 that is zero or at least
    // 2^-9, hence a multiple of 2^-32. A sum below 2^21 therefore has fewer
    // than 2^53 such units: every partial sum is exact in double, in any
    // order, and equals the element-by-element sum. A larger result may have
    // rounded, so the caller repeats that row in scalar order.
    constexpr float scale = 255.0f / 65535.0f;
    constexpr double exact_limit = 2097152.0;
    const __m256 v_scale = _mm256_set1_ps(scale);
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i vi = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i)));
        const __m256 vf = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), v_scale);
        acc0 = _mm256_add_pd(acc0, _mm256_cvtps_pd(_mm256_castps256_ps128(vf)));
        acc1 = _mm256_add_pd(acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(vf, 1)));
    }
    alignas(32) double parts[4];
    _mm256_store_pd(parts, _mm256_add_pd(acc0, acc1));
    double total = (parts[0] + parts[1]) + (parts[2] + parts[3]);
    for (; i < n; ++i) {
        total += static_cast<double>(static_cast<float>(p[i]) * scale);
    }
    if (!(total < exact_limit)) {
        return false;
    }
    *out = total;
    return true;
}

void quantize_f32_to_u16_row_avx2(
    const float* sr, uint16_t* dr, int64_t elems, float scale_f32
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_scale = _mm256_set1_ps(scale_f32);
    const __m256i v_32768 = _mm256_set1_epi32(32768);
    const __m256i v_xor = _mm256_set1_epi16(static_cast<short>(-32768));

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_255);
        v = _mm256_mul_ps(v, v_scale);
        __m256i qi = _mm256_cvtps_epi32(v);
        qi = _mm256_sub_epi32(qi, v_32768);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        p16 = _mm256_xor_si256(p16, v_xor);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dr + i), _mm256_castsi256_si128(p16));
    }
    for (; i < elems; ++i) {
        const float s = sr[i];
        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
        dr[i] = static_cast<uint16_t>(std::nearbyintf(c * scale_f32));
    }
}

void quantize_f32_to_u8_row_avx2(
    const float* sr, uint8_t* dr, int64_t elems
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_255);
        __m256i qi = _mm256_cvtps_epi32(v);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i lo = _mm256_castsi256_si128(p16);
        __m128i p8 = _mm_packus_epi16(lo, lo);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float s = sr[i];
        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
        dr[i] = static_cast<uint8_t>(std::nearbyintf(c));
    }
}

void exposure_gain_u16_row_avx2(
    const uint16_t* sr, uint16_t* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);
    const __m256i v_32768 = _mm256_set1_epi32(32768);
    const __m256i v_xor = _mm256_set1_epi16(static_cast<short>(-32768));

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        const __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + i)));
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(u), v_gain);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_limit);
        __m256i qi = _mm256_cvttps_epi32(v);
        qi = _mm256_sub_epi32(qi, v_32768);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        p16 = _mm256_xor_si256(p16, v_xor);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dr + i), _mm256_castsi256_si128(p16));
    }
    for (; i < elems; ++i) {
        const float v = std::clamp(static_cast<float>(sr[i]) * gain, 0.0f, limit);
        dr[i] = static_cast<uint16_t>(static_cast<int32_t>(v));
    }
}

void exposure_gain_u8_row_avx2(
    const uint8_t* sr, uint8_t* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        const __m256i u = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sr + i)));
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(u), v_gain);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_limit);
        __m256i qi = _mm256_cvttps_epi32(v);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float v = std::clamp(static_cast<float>(sr[i]) * gain, 0.0f, limit);
        dr[i] = static_cast<uint8_t>(static_cast<int32_t>(v));
    }
}

void exposure_gain_f32_row_avx2(
    const float* sr, float* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(sr + i), v_gain);
        // MINPS and MAXPS return their second operand for ties and NaNs.
        // Put the sample second to match std::clamp for NaNs and signed zero.
        v = _mm256_min_ps(v_limit, _mm256_max_ps(v_zero, v));
        _mm256_storeu_ps(dr + i, v);
    }
    for (; i < elems; ++i) {
        dr[i] = std::clamp(sr[i] * gain, 0.0f, limit);
    }
}

void down2_row_u16_to_i32_avx2(
    const uint16_t* r0, const uint16_t* r1, int32_t* dst, int64_t full_w, int32_t scale
) {
    const __m256i v_mask16 = _mm256_set1_epi32(0x0000FFFF);
    const __m256i v_scale = _mm256_set1_epi32(scale);
    int64_t x = 0;
    for (; x + 8 <= full_w; x += 8) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x));
        const __m256i pa = _mm256_add_epi32(_mm256_and_si256(a, v_mask16), _mm256_srli_epi32(a, 16));
        const __m256i pb = _mm256_add_epi32(_mm256_and_si256(b, v_mask16), _mm256_srli_epi32(b, 16));
        const __m256i tot = _mm256_add_epi32(pa, pb);
        const __m256i out = _mm256_srai_epi32(_mm256_mullo_epi32(tot, v_scale), 2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), out);
    }
    for (; x < full_w; ++x) {
        const int64_t left = 2 * x;
        const int32_t tot = static_cast<int32_t>(r0[left]) + static_cast<int32_t>(r0[left + 1])
                          + static_cast<int32_t>(r1[left]) + static_cast<int32_t>(r1[left + 1]);
        dst[x] = (tot * scale) >> 2;
    }
}

namespace {

// r + g + b of eight packed uint8 pixels (24 bytes) as eight uint32 lanes.
// Both 16-byte loads stay inside the 24 bytes.
inline __m256i rgb_sum_u8x8(const uint8_t* p) {
    const __m256i src = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8)), 1);
    const __m256i quads = _mm256_shuffle_epi8(src, _mm256_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
        4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1));
    const __m256i pairs = _mm256_maddubs_epi16(quads, _mm256_set1_epi8(1));
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

// r + g + b of eight packed uint16 pixels (24 samples) as eight uint32 lanes.
// Lane 0 of each load covers samples 0..7 / 8..15, lane 1 samples 8..15 / 16..23.
inline __m256i rgb_sum_u16x8(const uint16_t* p) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8));
    // One uint32 lane per pixel: red in the low half, green in the high half.
    const __m256i rg = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_setr_epi8(
            0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, -1, -1, -1, -1,
            8, 9, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
        _mm256_shuffle_epi8(hi, _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 4, 5,
            -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 6, 7, 10, 11, 12, 13)));
    const __m256i blue = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_setr_epi8(
            4, 5, -1, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
        _mm256_shuffle_epi8(hi, _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 6, 7, -1, -1,
            -1, -1, -1, -1, 2, 3, -1, -1, 8, 9, -1, -1, 14, 15, -1, -1)));
    const __m256i red_green = _mm256_add_epi32(
        _mm256_and_si256(rg, _mm256_set1_epi32(0xFFFF)), _mm256_srli_epi32(rg, 16));
    return _mm256_add_epi32(red_green, blue);
}

}  // namespace

void gray_rgb_u8_row_avx2(
    const uint8_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m256i v_magic = _mm256_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m256i v_scale = _mm256_set1_epi32(scale);
    const __m256i v_s32 = _mm256_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const __m256i t = rgb_sum_u8x8(sr + x * 3);
        __m256i res;
        if (depth_color) {
            res = _mm256_srli_epi32(_mm256_mullo_epi32(t, v_s32), 8);
        } else {
            const __m256i q_ev = _mm256_srli_epi64(_mm256_mul_epu32(t, v_magic), 33);
            const __m256i t_od = _mm256_srli_epi64(t, 32);
            const __m256i q_od = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_mul_epu32(t_od, v_magic), 33), 32);
            const __m256i q = _mm256_blend_epi32(q_ev, q_od, 0xAA);
            res = _mm256_mullo_epi32(q, v_scale);
        }
        if (dst_is_u16) {
            const __m256i low = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(low, low), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst16 + x), _mm256_castsi256_si128(p16));
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + x), res);
        }
    }
    for (; x < width; ++x) {
        const uint8_t* p = sr + x * 3;
        const uint32_t tot = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        const uint32_t v = depth_color ? ((tot * static_cast<uint32_t>(scale * 85)) >> 8u)
                                       : ((tot / 3u) * static_cast<uint32_t>(scale));
        if (dst_is_u16) {
            dst16[x] = static_cast<uint16_t>(v);
        } else {
            dst32[x] = static_cast<int32_t>(v);
        }
    }
}

void gray_rgb_u16_row_avx2(
    const uint16_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m256i v_magic = _mm256_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m256i v_scale = _mm256_set1_epi32(scale);
    const __m256i v_s32 = _mm256_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const __m256i t = rgb_sum_u16x8(sr + x * 3);
        __m256i res;
        if (depth_color) {
            res = _mm256_srli_epi32(_mm256_mullo_epi32(t, v_s32), 8);
        } else {
            const __m256i q_ev = _mm256_srli_epi64(_mm256_mul_epu32(t, v_magic), 33);
            const __m256i t_od = _mm256_srli_epi64(t, 32);
            const __m256i q_od = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_mul_epu32(t_od, v_magic), 33), 32);
            const __m256i q = _mm256_blend_epi32(q_ev, q_od, 0xAA);
            res = _mm256_mullo_epi32(q, v_scale);
        }
        if (dst_is_u16) {
            const __m256i low = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(low, low), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst16 + x), _mm256_castsi256_si128(p16));
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + x), res);
        }
    }
    for (; x < width; ++x) {
        const uint16_t* p = sr + x * 3;
        const uint32_t tot = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        const uint32_t v = depth_color ? ((tot * static_cast<uint32_t>(scale * 85)) >> 8u)
                                       : ((tot / 3u) * static_cast<uint32_t>(scale));
        if (dst_is_u16) {
            dst16[x] = static_cast<uint16_t>(v);
        } else {
            dst32[x] = static_cast<int32_t>(v);
        }
    }
}

namespace {
template <typename SrcT, typename DstT>
void merge_recip_row_impl(
    const uint16_t* drow, const SrcT* irow, DstT* arow, int64_t width,
    int64_t is1, int64_t is2, int64_t as1, int64_t as2,
    int32_t base32, int32_t step32, uint64_t recip48, int32_t s32
) {
    const int32_t neg_step = -step32;
    const __m256i v_base = _mm256_set1_epi32(base32);
    const __m256i v_step = _mm256_set1_epi32(step32);
    const __m256i v_nstep = _mm256_set1_epi32(neg_step);
    const bool packed3 = (is1 == 3 && is2 == 1 && as1 == 3 && as2 == 1);

    int64_t x = 0;
    while (x < width) {
        if (x + 16 <= width) {
            const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(drow + x));
            const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(drow + x + 8));
            const __m256i d0 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(r0), v_base);
            const __m256i d1 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(r1), v_base);
            const __m256i a0 = _mm256_andnot_si256(_mm256_cmpgt_epi32(d0, v_step), _mm256_cmpgt_epi32(d0, v_nstep));
            const __m256i a1 = _mm256_andnot_si256(_mm256_cmpgt_epi32(d1, v_step), _mm256_cmpgt_epi32(d1, v_nstep));
            const __m256i any = _mm256_or_si256(a0, a1);
            if (_mm256_testz_si256(any, any)) {
                x += 16;
                continue;
            }
        }
        const int64_t x_end = (x + 16 <= width) ? (x + 16) : width;
        for (; x < x_end; ++x) {
            const int32_t delta = static_cast<int32_t>(drow[x]) - base32;
            if (delta >= 0) {
                if (delta <= step32) {
                    const SrcT* sp = irow + x * is1;
                    DstT* dp = arow + x * as1;
                    if (packed3) {
                        dp[0] = static_cast<DstT>(static_cast<int32_t>(sp[0]) * s32);
                        dp[1] = static_cast<DstT>(static_cast<int32_t>(sp[1]) * s32);
                        dp[2] = static_cast<DstT>(static_cast<int32_t>(sp[2]) * s32);
                    } else {
                        for (int64_t ch = 0; ch < 3; ++ch) {
                            dp[ch * as2] = static_cast<DstT>(static_cast<int32_t>(sp[ch * is2]) * s32);
                        }
                    }
                }
            } else if (delta > neg_step) {
                const int32_t w = 1024 - static_cast<int32_t>(
                    (static_cast<uint64_t>(-delta) << 10u) * recip48 >> 48u);
                if (w > 0) {
                    const int32_t iw = 1024 - w;
                    const SrcT* sp = irow + x * is1;
                    DstT* dp = arow + x * as1;
                    if (packed3) {
                        dp[0] = static_cast<DstT>((static_cast<int32_t>(dp[0]) * iw + static_cast<int32_t>(sp[0]) * s32 * w) >> 10);
                        dp[1] = static_cast<DstT>((static_cast<int32_t>(dp[1]) * iw + static_cast<int32_t>(sp[1]) * s32 * w) >> 10);
                        dp[2] = static_cast<DstT>((static_cast<int32_t>(dp[2]) * iw + static_cast<int32_t>(sp[2]) * s32 * w) >> 10);
                    } else {
                        for (int64_t ch = 0; ch < 3; ++ch) {
                            const int32_t cur = static_cast<int32_t>(dp[ch * as2]);
                            const int32_t val = static_cast<int32_t>(sp[ch * is2]) * s32;
                            dp[ch * as2] = static_cast<DstT>((cur * iw + val * w) >> 10);
                        }
                    }
                }
            }
        }
    }
}

template <typename SrcT, typename DstT>
int select_a_no_bg_row_impl(
    int32_t* b_row, const int32_t* response, int64_t width,
    const SrcT* irow, DstT* orow, int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, int64_t max_response, int sharp_state, int saturated_state
) {
    int64_t x = 0;
    while (x < width) {
        if (state != saturated_state && x + 8 <= width) {
            const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(response + x));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + x));
            if (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(r, b))) == 0) {
                x += 8;
                continue;
            }
        }
        const int64_t x_end = (state != saturated_state && x + 8 <= width) ? (x + 8) : width;
        for (; x < x_end; ++x) {
            const int64_t prev = b_row[x];
            if (prev == max_response) {
                continue;
            }
            int64_t sc = response[x];
            if (sc >= max_response) {
                sc = max_response;
            } else if (state == saturated_state) {
                state = sharp_state;
            }
            if (sc > prev) {
                b_row[x] = static_cast<int32_t>(sc);
                const SrcT* sp = irow + x * is1;
                DstT* dp = orow + x * os1;
                dp[0] = static_cast<DstT>(static_cast<int64_t>(sp[0]) * scale);
                dp[os2] = static_cast<DstT>(static_cast<int64_t>(sp[is2]) * scale);
                dp[2 * os2] = static_cast<DstT>(static_cast<int64_t>(sp[2 * is2]) * scale);
            }
        }
    }
    return state;
}
}  // namespace

void merge_recip_packed_row_avx2(
    const uint16_t* drow, const void* irow, bool src_u8,
    void* arow, bool dst_u16, int64_t width,
    int64_t is1, int64_t is2, int64_t as1, int64_t as2,
    int32_t base32, int32_t step32, uint64_t recip48, int32_t s32
) {
    if (src_u8 && dst_u16) {
        merge_recip_row_impl<uint8_t, uint16_t>(
            drow, static_cast<const uint8_t*>(irow), static_cast<uint16_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else if (src_u8 && !dst_u16) {
        merge_recip_row_impl<uint8_t, int32_t>(
            drow, static_cast<const uint8_t*>(irow), static_cast<int32_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else if (!src_u8 && dst_u16) {
        merge_recip_row_impl<uint16_t, uint16_t>(
            drow, static_cast<const uint16_t*>(irow), static_cast<uint16_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else {
        merge_recip_row_impl<uint16_t, int32_t>(
            drow, static_cast<const uint16_t*>(irow), static_cast<int32_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    }
}

int select_a_no_bg_packed_row_avx2(
    int32_t* b_row, const int32_t* response, int64_t width,
    const void* irow, bool src_u8, void* orow, bool dst_u16,
    int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, int64_t max_response, int sharp_state, int saturated_state
) {
    if (src_u8 && dst_u16) {
        return select_a_no_bg_row_impl<uint8_t, uint16_t>(
            b_row, response, width, static_cast<const uint8_t*>(irow), static_cast<uint16_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    if (src_u8 && !dst_u16) {
        return select_a_no_bg_row_impl<uint8_t, int32_t>(
            b_row, response, width, static_cast<const uint8_t*>(irow), static_cast<int32_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    if (!src_u8 && dst_u16) {
        return select_a_no_bg_row_impl<uint16_t, uint16_t>(
            b_row, response, width, static_cast<const uint16_t*>(irow), static_cast<uint16_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    return select_a_no_bg_row_impl<uint16_t, int32_t>(
        b_row, response, width, static_cast<const uint16_t*>(irow), static_cast<int32_t*>(orow),
        is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
}

namespace {

// Memory channels 0, 1 and 2 of eight packed RGB pixels (24 samples at p) as
// eight int32 lanes each. Both 32-byte loads stay inside the 24 samples.
inline void deinterleave_rgb8(const uint16_t* p, __m256i& c0, __m256i& c1, __m256i& c2) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));      // 0..15
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 8));  // 8..23
    c0 = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_setr_epi8(
            0, 1, -1, -1, 6, 7, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1,
            8, 9, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
        _mm256_shuffle_epi8(hi, _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1, 10, 11, -1, -1)));
    c1 = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_setr_epi8(
            2, 3, -1, -1, 8, 9, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1,
            10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
        _mm256_shuffle_epi8(hi, _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1,
            -1, -1, -1, -1, 0, 1, -1, -1, 6, 7, -1, -1, 12, 13, -1, -1)));
    c2 = _mm256_or_si256(
        _mm256_shuffle_epi8(lo, _mm256_setr_epi8(
            4, 5, -1, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
        _mm256_shuffle_epi8(hi, _mm256_setr_epi8(
            -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 6, 7, -1, -1,
            -1, -1, -1, -1, 2, 3, -1, -1, 8, 9, -1, -1, 14, 15, -1, -1)));
}

inline void deinterleave_rgb8(const uint8_t* p, __m256i& c0, __m256i& c1, __m256i& c2) {
    // Lane 0 holds bytes 0..15 (pixels 0..3), lane 1 bytes 8..23 (pixels 4..7).
    const __m256i src = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8)), 1);
    c0 = _mm256_shuffle_epi8(src, _mm256_setr_epi8(
        0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1,
        4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1));
    c1 = _mm256_shuffle_epi8(src, _mm256_setr_epi8(
        1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1,
        5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1, 14, -1, -1, -1));
    c2 = _mm256_shuffle_epi8(src, _mm256_setr_epi8(
        2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1,
        6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1));
}

// The 24 output samples of eight pixels: the source samples scaled to 16 bits
// (by 1 for uint16, by 256 for uint8), in output channel order.
template <bool Rev>
inline void a_output_samples(const uint16_t* p, __m128i out[3]) {
    const __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i s8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8));
    const __m128i s16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
    if constexpr (!Rev) {
        out[0] = s0;
        out[1] = s8;
        out[2] = s16;
    } else {
        // Output sample 3k + c is source sample 3k + 2 - c.
        out[0] = _mm_or_si128(
            _mm_shuffle_epi8(s0, _mm_setr_epi8(4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7, -1, -1, 14, 15)),
            _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1)));
        out[1] = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(s0, _mm_setr_epi8(12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, 6, 7, 4, 5, 2, 3, 12, 13, 10, 11, 8, 9, -1, -1))),
            _mm_shuffle_epi8(s16, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3)));
        out[2] = _mm_or_si128(
            _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            _mm_shuffle_epi8(s16, _mm_setr_epi8(0, 1, -1, -1, 8, 9, 6, 7, 4, 5, 14, 15, 12, 13, 10, 11)));
    }
}

template <bool Rev>
inline void a_output_samples(const uint8_t* p, __m128i out[3]) {
    // Byte b becomes the 16-bit sample b << 8. Windows start at bytes 0, 4 and 8.
    const __m128i w0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i w4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4));
    const __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8));
    if constexpr (!Rev) {
        out[0] = _mm_shuffle_epi8(w0, _mm_setr_epi8(-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7));
        out[1] = _mm_shuffle_epi8(w4, _mm_setr_epi8(-1, 4, -1, 5, -1, 6, -1, 7, -1, 8, -1, 9, -1, 10, -1, 11));
        out[2] = _mm_shuffle_epi8(w8, _mm_setr_epi8(-1, 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15));
    } else {
        out[0] = _mm_shuffle_epi8(w0, _mm_setr_epi8(-1, 2, -1, 1, -1, 0, -1, 5, -1, 4, -1, 3, -1, 8, -1, 7));
        out[1] = _mm_shuffle_epi8(w4, _mm_setr_epi8(-1, 2, -1, 7, -1, 6, -1, 5, -1, 10, -1, 9, -1, 8, -1, 13));
        out[2] = _mm_shuffle_epi8(w8, _mm_setr_epi8(-1, 8, -1, 7, -1, 12, -1, 11, -1, 10, -1, 15, -1, 14, -1, 13));
    }
}

// Sixteen-bit masks for the 24 samples of the eight pixels selected in `pixels`.
inline void a_sample_masks(__m256i pixels, __m128i masks[3]) {
    const __m256i g0 = _mm256_permutevar8x32_epi32(pixels, _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2));
    const __m256i g1 = _mm256_permutevar8x32_epi32(pixels, _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5));
    const __m256i g2 = _mm256_permutevar8x32_epi32(pixels, _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7));
    masks[0] = _mm_packs_epi32(_mm256_castsi256_si128(g0), _mm256_extracti128_si256(g0, 1));
    masks[1] = _mm_packs_epi32(_mm256_castsi256_si128(g1), _mm256_extracti128_si256(g1, 1));
    masks[2] = _mm_packs_epi32(_mm256_castsi256_si128(g2), _mm256_extracti128_si256(g2, 1));
}

// Background statistics of eight pixels, five floats each (three weighted sums,
// the weight total and the packed response statistics), as one vector per field.
struct ABackground8 {
    __m256 sum[3];
    __m256 total;
    __m256i stat;
};

inline __m256 a_bg_pair(const float* bg, int k, int field) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(bg + 5 * k + field)),
                                _mm_loadu_ps(bg + 5 * (k + 4) + field), 1);
}

inline ABackground8 a_bg_load(const float* bg) {
    ABackground8 v;
    // Fields 0..3 of pixels k and k + 4, then a 4x4 transpose in each lane.
    const __m256 a0 = a_bg_pair(bg, 0, 0), a1 = a_bg_pair(bg, 1, 0);
    const __m256 a2 = a_bg_pair(bg, 2, 0), a3 = a_bg_pair(bg, 3, 0);
    const __m256 t0 = _mm256_unpacklo_ps(a0, a1), t1 = _mm256_unpackhi_ps(a0, a1);
    const __m256 t2 = _mm256_unpacklo_ps(a2, a3), t3 = _mm256_unpackhi_ps(a2, a3);
    v.sum[0] = _mm256_shuffle_ps(t0, t2, 0x44);
    v.sum[1] = _mm256_shuffle_ps(t0, t2, 0xEE);
    v.sum[2] = _mm256_shuffle_ps(t1, t3, 0x44);
    v.total = _mm256_shuffle_ps(t1, t3, 0xEE);
    // Field 4 is the last element of the fields 1..4 of each pixel.
    const __m256 b0 = a_bg_pair(bg, 0, 1), b1 = a_bg_pair(bg, 1, 1);
    const __m256 b2 = a_bg_pair(bg, 2, 1), b3 = a_bg_pair(bg, 3, 1);
    v.stat = _mm256_castps_si256(
        _mm256_shuffle_ps(_mm256_unpackhi_ps(b0, b1), _mm256_unpackhi_ps(b2, b3), 0xEE));
    return v;
}

inline void a_bg_store(float* bg, const ABackground8& v) {
    const __m256 t0 = _mm256_unpacklo_ps(v.sum[0], v.sum[1]), t1 = _mm256_unpackhi_ps(v.sum[0], v.sum[1]);
    const __m256 t2 = _mm256_unpacklo_ps(v.sum[2], v.total), t3 = _mm256_unpackhi_ps(v.sum[2], v.total);
    const __m256 rows[4] = {_mm256_shuffle_ps(t0, t2, 0x44), _mm256_shuffle_ps(t0, t2, 0xEE),
                            _mm256_shuffle_ps(t1, t3, 0x44), _mm256_shuffle_ps(t1, t3, 0xEE)};
    for (int k = 0; k < 4; ++k) {
        _mm_storeu_ps(bg + 5 * k, _mm256_castps256_ps128(rows[k]));
        _mm_storeu_ps(bg + 5 * (k + 4), _mm256_extractf128_ps(rows[k], 1));
    }
    alignas(32) uint32_t stat[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(stat), v.stat);
    for (int k = 0; k < 8; ++k) {
        std::memcpy(bg + 5 * k + 4, &stat[k], sizeof(uint32_t));
    }
}

inline __m256 select_ps(__m256 if_false, __m256 if_true, __m256i mask) {
    return _mm256_blendv_ps(if_false, if_true, _mm256_castsi256_ps(mask));
}

inline __m256i select_epi32(__m256i if_false, __m256i if_true, __m256i mask) {
    return _mm256_blendv_epi8(if_false, if_true, mask);
}

// Eight columns at a time of select_a_background_row in openfocus_ab.cpp. Each
// lane runs the scalar per-pixel sequence with the same float operations in
// the same order; the row state is the scalar sequence reduced over the chunk
// (any collected pixel makes it background, otherwise a non-collected pixel
// below the response maximum turns saturated into sharp).
template <typename SrcT, bool Rev, bool First>
int select_a_background_impl(
    int32_t* b_row, const int32_t* response, int64_t width,
    const SrcT* irow, uint16_t* orow, float* bg_row, int state, int64_t* done,
    int32_t max_response, int32_t blend_limit, int background_state, int sharp_state, int saturated_state
) {
    const __m256i v_max = _mm256_set1_epi32(max_response);
    const __m256i v_limit = _mm256_set1_epi32(blend_limit);
    const __m256i v_zero = _mm256_setzero_si256();
    const __m256i v_ones = _mm256_set1_epi32(-1);
    const __m256 f_one = _mm256_set1_ps(1.0f);
    const __m256 f_zero = _mm256_setzero_ps();
    const int shift = std::is_same<SrcT, uint8_t>::value ? 8 : 0;
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const __m256i prev = First ? v_zero : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + x));
        const __m256i sc = _mm256_min_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(response + x)), v_max);
        const __m256i live = First ? v_ones : _mm256_xor_si256(_mm256_cmpeq_epi32(prev, v_max), v_ones);
        const __m256i take = First ? v_ones : _mm256_cmpgt_epi32(sc, prev);
        const __m256i collect = _mm256_and_si256(
            _mm256_and_si256(_mm256_cmpgt_epi32(v_limit, prev), _mm256_cmpgt_epi32(v_limit, sc)), live);
        if (_mm256_movemask_epi8(collect) != 0) {
            state = background_state;
        } else if (state == saturated_state) {
            const __m256i below_max = _mm256_andnot_si256(_mm256_cmpeq_epi32(sc, v_max), live);
            if (_mm256_movemask_epi8(below_max) != 0) {
                state = sharp_state;
            }
        }
        if (_mm256_movemask_epi8(_mm256_or_si256(collect, take)) == 0) {
            continue;
        }
        const SrcT* sp = irow + 3 * x - (Rev ? 2 : 0);
        __m256i written = take;
        if (_mm256_movemask_epi8(collect) != 0) {
            float* bg = bg_row + 5 * x;
            ABackground8 v = a_bg_load(bg);
            __m256i c[3];
            deinterleave_rgb8(sp, c[0], c[1], c[2]);
            __m256 val[3];
            for (int ch = 0; ch < 3; ++ch) {
                val[ch] = _mm256_cvtepi32_ps(_mm256_slli_epi32(c[Rev ? 2 - ch : ch], shift));
            }
            const __m256i code = _mm256_srai_epi32(sc, 4);
            const __m256i is_zero = _mm256_cmpeq_epi32(sc, v_zero);
            if constexpr (First) {
                const __m256 weight = select_ps(f_one, f_zero, is_zero);
                v.stat = select_epi32(v.stat, code, collect);
                v.total = select_ps(v.total, weight, collect);
                for (int ch = 0; ch < 3; ++ch) {
                    v.sum[ch] = select_ps(v.sum[ch], _mm256_mul_ps(val[ch], weight), collect);
                }
            } else {
                const __m256 fprev = _mm256_cvtepi32_ps(prev);
                const __m256 fsc = _mm256_cvtepi32_ps(sc);
                const __m256 ratio = _mm256_div_ps(select_ps(fsc, fprev, take), select_ps(fprev, fsc, take));
                __m256 power = _mm256_mul_ps(ratio, ratio);
                power = _mm256_mul_ps(power, power);
                const __m256 factor = select_ps(f_one, power, take);
                const __m256 weight = select_ps(power, f_one, take);
                const __m256 total = select_ps(_mm256_add_ps(v.total, weight),
                                               _mm256_add_ps(_mm256_mul_ps(v.total, factor), f_one), take);
                const __m256i minimum = _mm256_and_si256(v.stat, _mm256_set1_epi32(4095));
                const __m256i second = _mm256_srli_epi32(v.stat, 12);
                const __m256i second_new = select_epi32(
                    _mm256_max_epu32(second, code), _mm256_srai_epi32(prev, 4), take);
                const __m256i stat = _mm256_or_si256(
                    _mm256_slli_epi32(second_new, 12), _mm256_min_epu32(minimum, code));
                // A zero response clears the minimum and skips the sums and the copy.
                const __m256i cleared = _mm256_and_si256(collect, is_zero);
                const __m256i update = _mm256_andnot_si256(is_zero, collect);
                v.stat = select_epi32(select_epi32(v.stat, stat, update), _mm256_slli_epi32(second, 12), cleared);
                v.total = select_ps(v.total, total, update);
                for (int ch = 0; ch < 3; ++ch) {
                    const __m256 sum = _mm256_add_ps(_mm256_mul_ps(v.sum[ch], factor), _mm256_mul_ps(val[ch], weight));
                    v.sum[ch] = select_ps(v.sum[ch], sum, update);
                }
                written = _mm256_andnot_si256(cleared, take);
            }
            a_bg_store(bg, v);
        }
        const int taken = _mm256_movemask_epi8(written);
        if (taken != 0) {
            __m128i samples[3];
            a_output_samples<Rev>(sp, samples);
            uint16_t* dp = orow + 3 * x;
            if (taken != -1) {
                __m128i masks[3];
                a_sample_masks(written, masks);
                for (int g = 0; g < 3; ++g) {
                    const __m128i old = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dp + 8 * g));
                    samples[g] = _mm_blendv_epi8(old, samples[g], masks[g]);
                }
            }
            for (int g = 0; g < 3; ++g) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 8 * g), samples[g]);
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(b_row + x), select_epi32(prev, sc, written));
        }
    }
    *done = x;
    return state;
}

}  // namespace

int select_a_background_row_avx2(
    int32_t* b_row, const int32_t* response, int64_t width,
    const void* irow, bool src_u8, bool src_reversed, uint16_t* orow,
    float* bg_row, bool first, int state, int64_t* done,
    int32_t max_response, int32_t blend_limit, int background_state, int sharp_state, int saturated_state
) {
    const auto run = [&](auto impl, const auto* src) {
        return impl(b_row, response, width, src, orow, bg_row, state, done,
                    max_response, blend_limit, background_state, sharp_state, saturated_state);
    };
    if (src_u8) {
        const auto* src = static_cast<const uint8_t*>(irow);
        if (src_reversed) {
            return first ? run(select_a_background_impl<uint8_t, true, true>, src)
                         : run(select_a_background_impl<uint8_t, true, false>, src);
        }
        return first ? run(select_a_background_impl<uint8_t, false, true>, src)
                     : run(select_a_background_impl<uint8_t, false, false>, src);
    }
    const auto* src = static_cast<const uint16_t*>(irow);
    if (src_reversed) {
        return first ? run(select_a_background_impl<uint16_t, true, true>, src)
                     : run(select_a_background_impl<uint16_t, true, false>, src);
    }
    return first ? run(select_a_background_impl<uint16_t, false, true>, src)
                 : run(select_a_background_impl<uint16_t, false, false>, src);
}

void tile_accumulate_f32_1c_row_avx2(
    const float* tr, const float* wr, float* ar, float* war, int64_t fw
) {
    int64_t dx = 0;
    for (; dx + 8 <= fw; dx += 8) {
        const __m256 w = _mm256_loadu_ps(wr + dx);
        const __m256 wa = _mm256_loadu_ps(war + dx);
        const __m256 t = _mm256_loadu_ps(tr + dx);
        const __m256 a = _mm256_loadu_ps(ar + dx);
        _mm256_storeu_ps(war + dx, _mm256_add_ps(wa, w));
        _mm256_storeu_ps(ar + dx, _mm256_add_ps(a, _mm256_mul_ps(t, w)));
    }
    for (; dx < fw; ++dx) {
        const float w = wr[dx];
        war[dx] += w;
        ar[dx] += tr[dx] * w;
    }
}

void tile_finalize_3c_row_avx2(
    const float* ar, const float* war, uint8_t* u8_dr, float* f32_dr,
    int64_t width, int64_t channels, bool out_u8
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    if (channels == 1) {
        int64_t x = 0;
        for (; x + 8 <= width; x += 8) {
            const __m256 w = _mm256_loadu_ps(war + x);
            const __m256 den = _mm256_blendv_ps(w, v_one, _mm256_cmp_ps(w, v_zero, _CMP_EQ_OQ));
            const __m256 raw = _mm256_div_ps(_mm256_loadu_ps(ar + x), den);
            const __m256 v = _mm256_min_ps(_mm256_max_ps(raw, v_zero), v_255);
            if (out_u8) {
                const __m256i qi = _mm256_cvtps_epi32(v);
                const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(qi, qi), 0xD8);
                const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(u8_dr + x), p8);
            } else {
                _mm256_storeu_ps(f32_dr + x, v);
            }
        }
        for (; x < width; ++x) {
            const float w = war[x];
            const float raw = ar[x] / ((w == 0.0f) ? 1.0f : w);
            const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
            if (out_u8) {
                u8_dr[x] = static_cast<uint8_t>(std::nearbyintf(val));
            } else {
                f32_dr[x] = val;
            }
        }
        return;
    }

    int64_t x = 0;
    alignas(32) float d_buf[24];
    if (channels == 3) {
        for (; x + 8 <= width; x += 8) {
            for (int k = 0; k < 8; ++k) {
                const float w = war[x + k];
                const float d = (w == 0.0f) ? 1.0f : w;
                d_buf[k * 3] = d;
                d_buf[k * 3 + 1] = d;
                d_buf[k * 3 + 2] = d;
            }
            const int64_t off = x * 3;
            for (int blk = 0; blk < 3; ++blk) {
                const int64_t b = off + blk * 8;
                const __m256 raw = _mm256_div_ps(_mm256_loadu_ps(ar + b), _mm256_load_ps(d_buf + blk * 8));
                const __m256 v = _mm256_min_ps(_mm256_max_ps(raw, v_zero), v_255);
                if (out_u8) {
                    const __m256i qi = _mm256_cvtps_epi32(v);
                    const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(qi, qi), 0xD8);
                    const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(u8_dr + b), p8);
                } else {
                    _mm256_storeu_ps(f32_dr + b, v);
                }
            }
        }
    }
    for (; x < width; ++x) {
        const float w = war[x];
        const float denom = (w == 0.0f) ? 1.0f : w;
        const int64_t off = x * channels;
        for (int64_t c = 0; c < channels; ++c) {
            const float raw = ar[off + c] / denom;
            const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
            if (out_u8) {
                u8_dr[off + c] = static_cast<uint8_t>(std::nearbyintf(val));
            } else {
                f32_dr[off + c] = val;
            }
        }
    }
}

bool pyr_any_gt_u16_avx2(const uint16_t* wr, const uint16_t* br, int64_t width) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    int64_t x = 0;
    for (; x + 16 <= width; x += 16) {
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i mx = _mm256_max_epu16(wv, bv);
        const __m256i le = _mm256_cmpeq_epi16(mx, bv);
        if (!_mm256_testc_si256(le, v_ones)) {
            return true;
        }
    }
    for (; x < width; ++x) {
        if (wr[x] > br[x]) {
            return true;
        }
    }
    return false;
}

void pyr_level0_merge_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, float* acc_row,
    int64_t width, float factor, uint16_t index, bool first,
    float w0, float w1, float w2, float w3, float w4, float point_one
) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vfac = _mm256_set1_ps(factor);
    const __m128i v_index = _mm_set1_epi16(static_cast<int16_t>(index));
    // Pixel mask lane k covers the three floats of pixel k.
    const __m256i spread0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i spread1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i spread2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    const uint8_t* s8 = static_cast<const uint8_t*>(raw_row);
    const uint16_t* s16 = static_cast<const uint16_t*>(raw_row);

    // Detail of eight pixels starting at x, written where `take` is set.
    const auto merge_block = [&](int64_t x, __m256i take, bool all) {
        const int64_t off = x * 3;
        const __m256i spread[3] = {spread0, spread1, spread2};
        for (int blk = 0; blk < 3; ++blk) {
            const int64_t b = off + blk * 8;
            __m256 lo = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(hr0 + b), vw0),
                    _mm256_mul_ps(_mm256_loadu_ps(hr1 + b), vw1)),
                _mm256_mul_ps(_mm256_loadu_ps(hr2 + b), vw2));
            lo = _mm256_add_ps(
                _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + b), vw3)),
                _mm256_mul_ps(_mm256_loadu_ps(hr4 + b), vw4));
            const __m256i raw_i = raw_is_u8
                ? _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s8 + b)))
                : _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s16 + b)));
            const __m256 sv = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_i), vfac);
            __m256 v = _mm256_div_ps(_mm256_sub_ps(sv, lo), _mm256_add_ps(lo, vp1));
            if (!all) {
                const __m256 mask = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(take, spread[blk]));
                v = _mm256_blendv_ps(_mm256_loadu_ps(acc_row + b), v, mask);
            }
            _mm256_storeu_ps(acc_row + b, v);
        }
    };

    int64_t x = 0;
    if (first) {
        for (; x + 8 <= width; x += 8) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(br + x),
                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + x)));
            if (win_row != nullptr) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(win_row + x), v_index);
            }
            merge_block(x, v_ones, true);
        }
    } else {
        for (; x + 8 <= width; x += 8) {
            if (x + 16 <= width && (x & 8) == 0) {
                // Most later frames improve nothing in a block: test 16 at once.
                const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
                const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
                if (_mm256_testc_si256(_mm256_cmpeq_epi16(_mm256_max_epu16(wv, bv), bv), v_ones)) {
                    x += 8;
                    continue;
                }
            }
            const __m128i wv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wr + x));
            const __m128i bv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(br + x));
            const __m128i mx = _mm_max_epu16(wv, bv);
            const __m128i keep = _mm_cmpeq_epi16(mx, bv);
            if (_mm_movemask_epi8(keep) == 0xFFFF) {
                continue;
            }
            const __m128i take16 = _mm_xor_si128(keep, _mm_set1_epi32(-1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(br + x), mx);
            if (win_row != nullptr) {
                const __m128i old = _mm_loadu_si128(reinterpret_cast<const __m128i*>(win_row + x));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(win_row + x),
                                 _mm_blendv_epi8(old, v_index, take16));
            }
            merge_block(x, _mm256_cvtepi16_epi32(take16), false);
        }
    }
    for (; x < width; ++x) {
        const uint16_t wval = wr[x];
        if (!first && wval <= br[x]) {
            continue;
        }
        br[x] = wval;
        if (win_row != nullptr) {
            win_row[x] = index;
        }
        const int64_t off3 = x * 3;
        float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
        float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
        float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
        lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
        lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
        lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
        float s0, s1, s2;
        if (raw_is_u8) {
            s0 = static_cast<float>(s8[off3]) * factor;
            s1 = static_cast<float>(s8[off3 + 1]) * factor;
            s2 = static_cast<float>(s8[off3 + 2]) * factor;
        } else {
            s0 = static_cast<float>(s16[off3]) * factor;
            s1 = static_cast<float>(s16[off3 + 1]) * factor;
            s2 = static_cast<float>(s16[off3 + 2]) * factor;
        }
        acc_row[off3] = (s0 - lo0) / (lo0 + point_one);
        acc_row[off3 + 1] = (s1 - lo1) / (lo1 + point_one);
        acc_row[off3 + 2] = (s2 - lo2) / (lo2 + point_one);
    }
}

void pyr_merge_f32_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* dr, float* ar, int64_t width, uint16_t index
) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    int64_t x = 0;
    while (x < width) {
        if (x + 16 <= width) {
            const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
            const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
            const __m256i mx = _mm256_max_epu16(wv, bv);
            if (_mm256_testc_si256(_mm256_cmpeq_epi16(mx, bv), v_ones)) {
                x += 16;
                continue;
            }
        }
        const int64_t x_end = (x + 16 <= width) ? (x + 16) : width;
        for (; x < x_end; ++x) {
            const uint16_t wval = wr[x];
            if (wval > br[x]) {
                br[x] = wval;
                if (win_row != nullptr) {
                    win_row[x] = index;
                }
                const int64_t off = x * 3;
                ar[off] = dr[off];
                ar[off + 1] = dr[off + 1];
                ar[off + 2] = dr[off + 2];
            }
        }
    }
}

void pyr_output_f32_to_u8_row_avx2(const float* sr, uint8_t* dr, int64_t elems) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_65535 = _mm256_set1_ps(65535.0f);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_blendv_ps(v, v_zero, _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        v = _mm256_blendv_ps(v, v_one, _mm256_cmp_ps(v, v_one, _CMP_GT_OQ));
        const __m256i q = _mm256_srli_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(v, v_65535)), 8);
        const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q, q), 0xD8);
        const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
        dr[i] = static_cast<uint8_t>(static_cast<int32_t>(v * 65535.0f) >> 8);
    }
}

void pyr_output_f32_to_f32_row_avx2(const float* sr, float* dr, int64_t elems) {
    constexpr float kInv257 = static_cast<float>(1.0 / 257.0);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_65535 = _mm256_set1_ps(65535.0f);
    const __m256 v_inv257 = _mm256_set1_ps(kInv257);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_blendv_ps(v, v_zero, _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        v = _mm256_blendv_ps(v, v_one, _mm256_cmp_ps(v, v_one, _CMP_GT_OQ));
        const __m256 rounded = _mm256_round_ps(_mm256_mul_ps(v, v_65535),
            _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
        _mm256_storeu_ps(dr + i, _mm256_mul_ps(rounded, v_inv257));
    }
    for (; i < elems; ++i) {
        const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
        dr[i] = std::nearbyintf(v * 65535.0f) * kInv257;
    }
}

void pyr_up_horiz_1c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3
) {
    const __m256 ve0 = _mm256_set1_ps(e0);
    const __m256 ve1 = _mm256_set1_ps(e1);
    const __m256 ve2 = _mm256_set1_ps(e2);
    const __m256 ve3 = _mm256_set1_ps(e3);
    int64_t x = ix_start;
    for (; x + 16 <= pair_stop; x += 16) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left;
        const __m256 s0 = _mm256_loadu_ps(p);
        const __m256 s1 = _mm256_loadu_ps(p + 1);
        const __m256 s2 = _mm256_loadu_ps(p + 2);
        const __m256 s3 = _mm256_loadu_ps(p + 3);
        const __m256 s4 = _mm256_loadu_ps(p + 4);
        const __m256 ve = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(s0, ve0), _mm256_mul_ps(s1, ve1)),
                          _mm256_mul_ps(s2, ve2)),
            _mm256_mul_ps(s3, ve3));
        const __m256 vo = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(s1, ve3), _mm256_mul_ps(s2, ve2)),
                          _mm256_mul_ps(s3, ve1)),
            _mm256_mul_ps(s4, ve0));
        const __m256 lo = _mm256_unpacklo_ps(ve, vo);
        const __m256 hi = _mm256_unpackhi_ps(ve, vo);
        _mm256_storeu_ps(dr + x, _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(dr + x + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
    for (; x < pair_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left;
        const float ve = (p[0] * e0 + p[1] * e1) + p[2] * e2;
        dr[x] = ve + p[3] * e3;
        const float vo = p[1] * e3 + p[2] * e2;
        dr[x + 1] = (vo + p[3] * e1) + p[4] * e0;
    }
}

void pyr_up_horiz_3c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3
) {
    // Keep last pair scalar to guarantee 4-float loads stay strictly inside row bounds.
    const int64_t safe_stop = (pair_stop > ix_start) ? (pair_stop - 2) : ix_start;
    const __m256 vw_a = _mm256_set_m128(_mm_set1_ps(e3), _mm_set1_ps(e0));
    const __m256 vw_b = _mm256_set_m128(_mm_set1_ps(e2), _mm_set1_ps(e1));
    const __m256 vw_c = _mm256_set_m128(_mm_set1_ps(e1), _mm_set1_ps(e2));
    const __m256 vw_d = _mm256_set_m128(_mm_set1_ps(e0), _mm_set1_ps(e3));
    const __m256 zero = _mm256_setzero_ps();
    int64_t x = ix_start;
    for (; x < safe_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left * 3;
        const __m128 a = _mm_loadu_ps(p);
        const __m128 b = _mm_loadu_ps(p + 3);
        const __m128 c = _mm_loadu_ps(p + 6);
        const __m128 d = _mm_loadu_ps(p + 9);
        const __m128 e = _mm_loadu_ps(p + 12);
        const __m256 p0 = _mm256_set_m128(b, a);
        const __m256 p1 = _mm256_set_m128(c, b);
        const __m256 p2 = _mm256_set_m128(d, c);
        const __m256 p3 = _mm256_set_m128(e, d);
        // The scalar odd sum starts with +0 and the even sum ends with +0.
        const __m256 first = _mm256_mul_ps(p0, vw_a);
        __m256 v = _mm256_add_ps(
            _mm256_add_ps(_mm256_blend_ps(first, _mm256_add_ps(zero, first), 0xF0),
                          _mm256_mul_ps(p1, vw_b)),
            _mm256_mul_ps(p2, vw_c));
        v = _mm256_add_ps(v, _mm256_mul_ps(p3, vw_d));
        const __m256 out = _mm256_blend_ps(v, _mm256_add_ps(v, zero), 0x0F);
        // Each store spills one lane onto the following pixel, which the next
        // store (and finally the scalar tail's pair) overwrites.
        float* d0 = dr + x * 3;
        _mm_storeu_ps(d0, _mm256_castps256_ps128(out));
        _mm_storeu_ps(d0 + 3, _mm256_extractf128_ps(out, 1));
    }
    for (; x < pair_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left * 3;
        float* d0 = dr + x * 3;
        float* d1 = dr + (x + 1) * 3;
        for (int64_t ch = 0; ch < 3; ++ch) {
            const float s0 = p[ch], s1 = p[3 + ch], s2 = p[6 + ch], s3 = p[9 + ch], s4 = p[12 + ch];
            float ve = (s0 * e0 + s1 * e1) + s2 * e2;
            ve = (ve + s3 * e3) + 0.0f;
            d0[ch] = ve;
            float vo = (0.0f + s1 * e3) + s2 * e2;
            vo = (vo + s3 * e1) + s4 * e0;
            d1[ch] = vo;
        }
    }
}

void raw_row_to_f32_scaled_avx2(
    const void* src, bool is_u8, float* dst, int64_t elems, float factor
) {
    const __m256 vf = _mm256_set1_ps(factor);
    int64_t i = 0;
    if (is_u8) {
        const uint8_t* sp = static_cast<const uint8_t*>(src);
        for (; i + 8 <= elems; i += 8) {
            const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sp + i)));
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_cvtepi32_ps(v), vf));
        }
        for (; i < elems; ++i) {
            dst[i] = static_cast<float>(sp[i]) * factor;
        }
    } else {
        const uint16_t* sp = static_cast<const uint16_t*>(src);
        for (; i + 8 <= elems; i += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sp + i)));
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_cvtepi32_ps(v), vf));
        }
        for (; i < elems; ++i) {
            dst[i] = static_cast<float>(sp[i]) * factor;
        }
    }
}

void pyr_down_horiz_f32_3c_row_avx2(
    const float* sr, float* h_row, int64_t ix_stop, float d0, float d1
) {
    const __m128 vd0 = _mm_set1_ps(d0);
    const __m128 vd1 = _mm_set1_ps(d1);
    const int64_t safe_stop = (ix_stop > 1) ? (ix_stop - 1) : 1;
    int64_t x = 1;
    for (; x < safe_stop; ++x) {
        const float* p = sr + (2 * x - 1) * 3;
        __m128 v = _mm_mul_ps(_mm_loadu_ps(p), vd0);
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 3), vd1));
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 6), vd1));
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 9), vd0));
        // The fourth lane lands on the next pixel, which a later iteration
        // (or the scalar tail below) overwrites.
        _mm_storeu_ps(h_row + x * 3, v);
    }
    for (; x < ix_stop; ++x) {
        const float* p = sr + (2 * x - 1) * 3;
        float* dst = h_row + x * 3;
        for (int64_t c = 0; c < 3; ++c) {
            float v = p[c] * d0;
            v += p[3 + c] * d1;
            v += p[6 + c] * d1;
            dst[c] = v + p[9 + c] * d0;
        }
    }
}

namespace {

constexpr uint32_t kAlignDownW0 = 12u;
constexpr uint32_t kAlignDownW1 = 64u;
constexpr uint32_t kAlignDownW2 = 37u;
constexpr uint32_t kAlignDownW3 = 2u;
// Exact unsigned division magic constants verified over full input ranges:
// sum / 3 == (sum * 0xAAAAAAABu) >> 33 for 0..196605
// (v + 57) / 115 == ((v + 57) * 0x0239E0D6u) >> 32 for 0..7536582
constexpr uint32_t kDiv3Magic = 0xAAAAAAABu;
constexpr uint32_t kDiv115Magic = 0x0239E0D6u;
constexpr float kAlignClampEdge = 1.0010000467300415f;

inline __m256i div3_epu32_avx2(__m256i sum) {
    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(kDiv3Magic));
    const __m256i q_even = _mm256_srli_epi64(_mm256_mul_epu32(sum, vm), 33);
    const __m256i q_odd = _mm256_srli_epi32(
        _mm256_mul_epu32(_mm256_srli_epi64(sum, 32), vm), 1);
    return _mm256_blend_epi32(q_even, q_odd, 0xAA);
}

inline __m256i div115_round_epu32_avx2(__m256i v) {
    const __m256i v57 = _mm256_add_epi32(v, _mm256_set1_epi32(57));
    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(kDiv115Magic));
    const __m256i q_even = _mm256_srli_epi64(_mm256_mul_epu32(v57, vm), 32);
    const __m256i q_odd = _mm256_mul_epu32(_mm256_srli_epi64(v57, 32), vm);
    return _mm256_blend_epi32(q_even, q_odd, 0xAA);
}

inline void store_u32x8_as_u16(__m256i v, uint16_t* dst) {
    const __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), 0xD8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(p));
}

// Bilinear blend of two horizontally adjacent packed RGB pixels on two rows.
// p0 and p1 address the left pixel; eight samples are read from each row, so
// the left pixel must be at most the third from the row end. Lane c of the
// result is output channel c. With gx + fx == 256 the horizontal step is one
// signed multiply-add of the (left, right) sample pair, and the vertical step
// top * gy + bot * fy equals (top << 8) + (bot - top) * fy modulo 2^32.
template <bool BgrRev>
inline __m128i bilinear_rgb_pair(const uint16_t* p0, const uint16_t* p1, uint32_t fx, uint32_t fy) {
    const __m128i shuf = BgrRev
        ? _mm_setr_epi8(4, 5, 10, 11, 2, 3, 8, 9, 0, 1, 6, 7, -1, -1, -1, -1)
        : _mm_setr_epi8(0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1);
    const __m128i bias = _mm_set1_epi16(static_cast<int16_t>(0x8000));
    const __m128i unbias = _mm_set1_epi32(1 << 23);
    const __m128i weights = _mm_set1_epi32(static_cast<int32_t>((256u - fx) | (fx << 16u)));
    const __m128i row0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0)), shuf);
    const __m128i row1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1)), shuf);
    const __m128i top = _mm_add_epi32(_mm_madd_epi16(_mm_xor_si128(row0, bias), weights), unbias);
    const __m128i bot = _mm_add_epi32(_mm_madd_epi16(_mm_xor_si128(row1, bias), weights), unbias);
    return _mm_srli_epi32(
        _mm_add_epi32(_mm_slli_epi32(top, 8),
                      _mm_mullo_epi32(_mm_sub_epi32(bot, top), _mm_set1_epi32(static_cast<int32_t>(fy)))), 16);
}

template <bool BgrRev>
inline __m128i bilinear_rgb_pair(const uint8_t* p0, const uint8_t* p1, uint32_t fx, uint32_t fy) {
    const __m128i shuf = BgrRev
        ? _mm_setr_epi8(2, -1, 5, -1, 1, -1, 4, -1, 0, -1, 3, -1, -1, -1, -1, -1)
        : _mm_setr_epi8(0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, -1, -1, -1, -1);
    const __m128i weights = _mm_set1_epi32(static_cast<int32_t>((256u - fx) | (fx << 16u)));
    const __m128i row0 = _mm_shuffle_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p0)), shuf);
    const __m128i row1 = _mm_shuffle_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p1)), shuf);
    const __m128i top = _mm_madd_epi16(row0, weights);
    const __m128i bot = _mm_madd_epi16(row1, weights);
    return _mm_srli_epi32(
        _mm_add_epi32(_mm_slli_epi32(top, 8),
                      _mm_mullo_epi32(_mm_sub_epi32(bot, top), _mm_set1_epi32(static_cast<int32_t>(fy)))), 16);
}

// Stores the three channel lanes of `value`. `spill` allows one extra sample
// to be written past the pixel when a later store overwrites it.
inline void store_rgb_lanes(__m128i value, uint16_t* dp, bool spill) {
    const __m128i packed = _mm_packus_epi32(value, value);
    if (spill) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dp), packed);
    } else {
        const uint32_t two = static_cast<uint32_t>(_mm_cvtsi128_si32(packed));
        dp[0] = static_cast<uint16_t>(two);
        dp[1] = static_cast<uint16_t>(two >> 16u);
        dp[2] = static_cast<uint16_t>(_mm_extract_epi16(packed, 2));
    }
}

inline void store_rgb_lanes(__m128i value, uint8_t* dp, bool spill) {
    const __m128i words = _mm_packus_epi32(value, value);
    const uint32_t bytes = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_packus_epi16(words, words)));
    if (spill) {
        std::memcpy(dp, &bytes, sizeof(bytes));
    } else {
        dp[0] = static_cast<uint8_t>(bytes);
        dp[1] = static_cast<uint8_t>(bytes >> 8u);
        dp[2] = static_cast<uint8_t>(bytes >> 16u);
    }
}

// One output pixel of a packed warp row; `len_x` is the source row length in
// pixels and `spill` as in store_rgb_lanes.
template <typename T, int Channels, bool BgrRev>
inline void warp_pixel(
    const T* r0, const T* r1, T* dp, int32_t ix, int32_t jx, uint32_t fxx, uint32_t fyy,
    int32_t len_x, bool spill
) {
    const int64_t ixx = static_cast<int64_t>(ix) * Channels;
    const int64_t jxx = static_cast<int64_t>(jx) * Channels;
    if constexpr (Channels == 1) {
        (void)len_x;
        (void)spill;
        const uint32_t gxx = 256u - fxx;
        const uint32_t gyy = 256u - fyy;
        const uint32_t top = static_cast<uint32_t>(r0[ixx]) * gxx + static_cast<uint32_t>(r0[jxx]) * fxx;
        const uint32_t bot = static_cast<uint32_t>(r1[ixx]) * gxx + static_cast<uint32_t>(r1[jxx]) * fxx;
        dp[0] = static_cast<T>((top * gyy + bot * fyy) >> 16u);
    } else {
        if (jx == ix + 1 && ix <= len_x - 3) {
            store_rgb_lanes(bilinear_rgb_pair<BgrRev>(r0 + ixx, r1 + ixx, fxx, fyy), dp, spill);
            return;
        }
        const uint32_t gxx = 256u - fxx;
        const uint32_t gyy = 256u - fyy;
        for (int c = 0; c < 3; ++c) {
            const int sc = BgrRev ? (2 - c) : c;
            const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx
                               + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
            const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx
                               + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
            dp[c] = static_cast<T>((top * gyy + bot * fyy) >> 16u);
        }
    }
}

// `more_after` tells whether the row continues past this chunk.
template <typename T, int Channels, bool BgrRev>
inline void warp_chunk_8(
    const T* img, int64_t img_s0, int64_t src_y0,
    T* out_row, int64_t ox,
    const int32_t* ix, const int32_t* jx, const int32_t* iy, const int32_t* jy,
    const uint32_t* fx, const uint32_t* fy, int inv_mask, int32_t len_x, bool more_after
) {
    for (int k = 0; k < 8; ++k) {
        T* dp = out_row + (ox + k) * Channels;
        if ((inv_mask >> k) & 1) {
            for (int c = 0; c < Channels; ++c) {
                dp[c] = 0;
            }
            continue;
        }
        const T* r0 = img + (static_cast<int64_t>(iy[k]) - src_y0) * img_s0;
        const T* r1 = img + (static_cast<int64_t>(jy[k]) - src_y0) * img_s0;
        warp_pixel<T, Channels, BgrRev>(r0, r1, dp, ix[k], jx[k], fx[k], fy[k], len_x, k < 7 || more_after);
    }
}

// ---- Dense warp chunks
// When the eight outputs of a chunk take their left taps from consecutive
// source columns ix0 .. ix0 + 7 and their right taps from ix0 + 1 .. ix0 + 8,
// all on the same two rows, the taps are read as whole spans instead of pixel
// by pixel. Every output sample keeps the scalar arithmetic: per row
// a * (256 - fx) + b * fx, then (top * (256 - fy) + bot * fy) >> 16, exact in
// 32 bits. `fx` and `fy` hold the eight outputs' fractions in 32-bit lanes.
// Loads stay inside columns ix0 .. ix0 + 8 and exactly eight pixels are stored.

// (256 - f) in the low and f in the high 16 bits of each lane, for madd.
inline __m256i dense_weights(__m256i frac) {
    return _mm256_or_si256(_mm256_sub_epi32(_mm256_set1_epi32(256), frac), _mm256_slli_epi32(frac, 16));
}

inline __m256i dense_vertical(__m256i top, __m256i bot, __m256i fy) {
    return _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_slli_epi32(top, 8), _mm256_mullo_epi32(_mm256_sub_epi32(bot, top), fy)), 16);
}

inline int32_t lane0_epi32(__m256i v) {
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(v));
}

// True when output k of the chunk reads columns ix[0] + k and ix[0] + k + 1 of
// one pair of rows for every k.
inline bool dense_chunk(__m256i ix, __m256i jx, __m256i iy, __m256i jy) {
    const __m256i first = _mm256_setzero_si256();
    const __m256i steps = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i columns = _mm256_and_si256(
        _mm256_cmpeq_epi32(ix, _mm256_add_epi32(_mm256_permutevar8x32_epi32(ix, first), steps)),
        _mm256_cmpeq_epi32(jx, _mm256_add_epi32(ix, _mm256_set1_epi32(1))));
    const __m256i rows = _mm256_and_si256(
        _mm256_cmpeq_epi32(iy, _mm256_permutevar8x32_epi32(iy, first)),
        _mm256_cmpeq_epi32(jy, _mm256_permutevar8x32_epi32(jy, first)));
    return _mm256_movemask_epi8(_mm256_and_si256(columns, rows)) == -1;
}

// The same test for a translation row, whose two rows are fixed.
inline bool dense_columns(const int32_t* ix, const int32_t* jx, const uint8_t* inv_x) {
    uint64_t invalid = 0;
    std::memcpy(&invalid, inv_x, sizeof(invalid));
    if (invalid != 0) {
        return false;
    }
    const __m256i vix = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ix));
    const __m256i vjx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(jx));
    const __m256i same = _mm256_and_si256(
        _mm256_cmpeq_epi32(vix, _mm256_add_epi32(_mm256_set1_epi32(ix[0]), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7))),
        _mm256_cmpeq_epi32(vjx, _mm256_add_epi32(vix, _mm256_set1_epi32(1))));
    return _mm256_movemask_epi8(same) == -1;
}

// Horizontal sums of one gray row for eight outputs.
inline __m256i dense_gray_row(const uint16_t* s, __m256i w) {
    const __m128i bias = _mm_set1_epi16(static_cast<int16_t>(0x8000));
    const __m128i a = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s)), bias);
    const __m128i b = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 1)), bias);
    const __m256i pairs = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_unpacklo_epi16(a, b)), _mm_unpackhi_epi16(a, b), 1);
    return _mm256_add_epi32(_mm256_madd_epi16(pairs, w), _mm256_set1_epi32(1 << 23));
}

inline __m256i dense_gray_row(const uint8_t* s, __m256i w) {
    const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s));
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + 1));
    return _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_unpacklo_epi8(a, b)), w);
}

inline void dense_store_gray(__m256i v, uint16_t* dp) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp),
                     _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

inline void dense_store_gray(__m256i v, uint8_t* dp) {
    const __m128i words = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dp), _mm_packus_epi16(words, words));
}

template <typename T>
inline void dense_gray8(const T* s0, const T* s1, T* dp, __m256i fx, __m256i fy) {
    const __m256i w = dense_weights(fx);
    dense_store_gray(dense_vertical(dense_gray_row(s0, w), dense_gray_row(s1, w), fy), dp);
}

// Eight packed RGB pixels hold 24 output samples, taken in three groups of
// eight. Group g covers samples 8g .. 8g + 7; this maps each to its pixel.
template <int G>
inline __m256i dense_rgb_pixels() {
    if constexpr (G == 0) {
        return _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    } else if constexpr (G == 1) {
        return _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    } else {
        return _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    }
}

// Horizontal sums of one row for group G, in source channel order: sample j
// pairs with sample j + 3 of the same channel one column right. Lane 0 reads
// s + 8G for samples 8G .. 8G + 3, lane 1 reads s + 8G + 3 for 8G + 4 ..
// 8G + 7, so group 2 ends at sample 26, the last channel of column ix0 + 8.
// The 0x8000 bias makes the sample pairs signed for madd; 1 << 23 removes it.
template <int G>
inline __m256i dense_rgb_row(const uint16_t* s, __m256i w) {
    const __m256i shuf = _mm256_setr_epi8(
        0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, 6, 7, 12, 13,
        2, 3, 8, 9, 4, 5, 10, 11, 6, 7, 12, 13, 8, 9, 14, 15);
    const __m256i span = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 8 * G))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 8 * G + 3)), 1);
    const __m256i pairs = _mm256_xor_si256(
        _mm256_shuffle_epi8(span, shuf), _mm256_set1_epi16(static_cast<int16_t>(0x8000)));
    return _mm256_add_epi32(_mm256_madd_epi16(pairs, w), _mm256_set1_epi32(1 << 23));
}

// The same for 8-bit samples, in output channel order: one 16-byte span holds
// every tap of the group, so the shuffle also reverses the channels. Spans
// start at 0, 8 (6 reversed) and 11 and end by sample 26.
template <int G, bool BgrRev>
inline __m256i dense_rgb_row(const uint8_t* s, __m256i w) {
    constexpr int start = (G == 0) ? 0 : ((G == 1) ? (BgrRev ? 6 : 8) : 11);
    __m256i shuf;
    if constexpr (!BgrRev && G < 2) {
        shuf = _mm256_setr_epi8(
            0, -1, 3, -1, 1, -1, 4, -1, 2, -1, 5, -1, 3, -1, 6, -1,
            4, -1, 7, -1, 5, -1, 8, -1, 6, -1, 9, -1, 7, -1, 10, -1);
    } else if constexpr (!BgrRev) {
        shuf = _mm256_setr_epi8(
            5, -1, 8, -1, 6, -1, 9, -1, 7, -1, 10, -1, 8, -1, 11, -1,
            9, -1, 12, -1, 10, -1, 13, -1, 11, -1, 14, -1, 12, -1, 15, -1);
    } else if constexpr (G == 0) {
        shuf = _mm256_setr_epi8(
            2, -1, 5, -1, 1, -1, 4, -1, 0, -1, 3, -1, 5, -1, 8, -1,
            4, -1, 7, -1, 3, -1, 6, -1, 8, -1, 11, -1, 7, -1, 10, -1);
    } else if constexpr (G == 1) {
        shuf = _mm256_setr_epi8(
            0, -1, 3, -1, 5, -1, 8, -1, 4, -1, 7, -1, 3, -1, 6, -1,
            8, -1, 11, -1, 7, -1, 10, -1, 6, -1, 9, -1, 11, -1, 14, -1);
    } else {
        shuf = _mm256_setr_epi8(
            5, -1, 8, -1, 4, -1, 7, -1, 9, -1, 12, -1, 8, -1, 11, -1,
            7, -1, 10, -1, 12, -1, 15, -1, 11, -1, 14, -1, 10, -1, 13, -1);
    }
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + start));
    const __m256i span = _mm256_inserti128_si256(_mm256_castsi128_si256(bytes), bytes, 1);
    return _mm256_madd_epi16(_mm256_shuffle_epi8(span, shuf), w);
}

// Stores 24 samples given in source channel order, reversing each pixel's
// channels when BgrRev.
template <bool BgrRev>
inline void dense_store_rgb(__m256i g0, __m256i g1, __m256i g2, uint16_t* dp) {
    const __m256i low = _mm256_permute4x64_epi64(_mm256_packus_epi32(g0, g1), 0xD8);
    const __m128i s16 = _mm_packus_epi32(_mm256_castsi256_si128(g2), _mm256_extracti128_si256(g2, 1));
    if constexpr (!BgrRev) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dp), low);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 16), s16);
    } else {
        const __m128i s0 = _mm256_castsi256_si128(low);
        const __m128i s8 = _mm256_extracti128_si256(low, 1);
        const __m128i o0 = _mm_or_si128(
            _mm_shuffle_epi8(s0, _mm_setr_epi8(4, 5, 2, 3, 0, 1, 10, 11, 8, 9, 6, 7, -1, -1, 14, 15)),
            _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1)));
        const __m128i o1 = _mm_or_si128(
            _mm_or_si128(
                _mm_shuffle_epi8(s0, _mm_setr_epi8(12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, 6, 7, 4, 5, 2, 3, 12, 13, 10, 11, 8, 9, -1, -1))),
            _mm_shuffle_epi8(s16, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3)));
        const __m128i o2 = _mm_or_si128(
            _mm_shuffle_epi8(s8, _mm_setr_epi8(-1, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            _mm_shuffle_epi8(s16, _mm_setr_epi8(0, 1, -1, -1, 8, 9, 6, 7, 4, 5, 14, 15, 12, 13, 10, 11)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dp), o0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 8), o1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 16), o2);
    }
}

// Stores 24 samples already in output channel order.
inline void dense_store_rgb(__m256i g0, __m256i g1, __m256i g2, uint8_t* dp) {
    // Words [0-3, 8-11 | 4-7, 12-15] and [16-19, 16-19 | 20-23, 20-23], then
    // bytes whose dwords are reordered to 0-3, 4-7, ..., 20-23.
    const __m256i bytes = _mm256_packus_epi16(_mm256_packus_epi32(g0, g1), _mm256_packus_epi32(g2, g2));
    const __m256i ordered = _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp), _mm256_castsi256_si128(ordered));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dp + 16), _mm256_extracti128_si256(ordered, 1));
}

// Eight packed RGB outputs. With RowFy, `fy` is one value for the whole row;
// otherwise each pixel's fraction is spread to its samples like `fx`.
template <int G, typename T, bool BgrRev, bool RowFy>
inline __m256i dense_rgb_group(const T* s0, const T* s1, __m256i w, __m256i fy) {
    const __m256i pixels = dense_rgb_pixels<G>();
    const __m256i wg = _mm256_permutevar8x32_epi32(w, pixels);
    const __m256i fyg = RowFy ? fy : _mm256_permutevar8x32_epi32(fy, pixels);
    if constexpr (std::is_same<T, uint8_t>::value) {
        return dense_vertical(dense_rgb_row<G, BgrRev>(s0, wg), dense_rgb_row<G, BgrRev>(s1, wg), fyg);
    } else {
        return dense_vertical(dense_rgb_row<G>(s0, wg), dense_rgb_row<G>(s1, wg), fyg);
    }
}

template <typename T, bool BgrRev, bool RowFy>
inline void dense_rgb8(const T* s0, const T* s1, T* dp, __m256i fx, __m256i fy) {
    const __m256i w = dense_weights(fx);
    const __m256i g0 = dense_rgb_group<0, T, BgrRev, RowFy>(s0, s1, w, fy);
    const __m256i g1 = dense_rgb_group<1, T, BgrRev, RowFy>(s0, s1, w, fy);
    const __m256i g2 = dense_rgb_group<2, T, BgrRev, RowFy>(s0, s1, w, fy);
    if constexpr (std::is_same<T, uint8_t>::value) {
        dense_store_rgb(g0, g1, g2, dp);
    } else {
        dense_store_rgb<BgrRev>(g0, g1, g2, dp);
    }
}

}  // namespace

namespace {

template <typename T, bool BgrRev>
void warp_translation_rgb_row(
    const T* r0, const T* r1, T* dst, int64_t rw,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x,
    uint32_t fy, int32_t len_x
) {
    const __m256i v_fy = _mm256_set1_epi32(static_cast<int32_t>(fy));
    const auto one = [&](int64_t ox) {
        T* dp = dst + ox * 3;
        if (inv_x[ox]) {
            dp[0] = 0;
            dp[1] = 0;
            dp[2] = 0;
            return;
        }
        warp_pixel<T, 3, BgrRev>(r0, r1, dp, ix[ox], jx[ox], fx[ox], fy, len_x, ox + 1 < rw);
    };
    int64_t ox = 0;
    for (; ox + 8 <= rw; ox += 8) {
        if (dense_columns(ix + ox, jx + ox, inv_x + ox)) {
            const int64_t first = static_cast<int64_t>(ix[ox]) * 3;
            dense_rgb8<T, BgrRev, true>(r0 + first, r1 + first, dst + ox * 3,
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fx + ox)), v_fy);
            continue;
        }
        for (int64_t k = ox; k < ox + 8; ++k) {
            one(k);
        }
    }
    for (; ox < rw; ++ox) {
        one(ox);
    }
}

}  // namespace

void align_warp_translation_rgb_row_avx2(
    const void* row0, const void* row1, bool is_u8, void* out_row, int64_t rw,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x,
    uint32_t fy, bool bgr_rev, int64_t src_w
) {
    const int32_t len_x = static_cast<int32_t>(src_w);
    if (is_u8) {
        const auto* r0 = static_cast<const uint8_t*>(row0);
        const auto* r1 = static_cast<const uint8_t*>(row1);
        auto* dst = static_cast<uint8_t*>(out_row);
        if (bgr_rev) {
            warp_translation_rgb_row<uint8_t, true>(r0, r1, dst, rw, ix, jx, fx, inv_x, fy, len_x);
        } else {
            warp_translation_rgb_row<uint8_t, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, fy, len_x);
        }
    } else {
        const auto* r0 = static_cast<const uint16_t*>(row0);
        const auto* r1 = static_cast<const uint16_t*>(row1);
        auto* dst = static_cast<uint16_t*>(out_row);
        if (bgr_rev) {
            warp_translation_rgb_row<uint16_t, true>(r0, r1, dst, rw, ix, jx, fx, inv_x, fy, len_x);
        } else {
            warp_translation_rgb_row<uint16_t, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, fy, len_x);
        }
    }
}

namespace {

template <typename T>
void warp_translation_gray_row(
    const T* r0, const T* r1, T* dst, int64_t rw,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x, uint32_t fy
) {
    const __m256i v_fy = _mm256_set1_epi32(static_cast<int32_t>(fy));
    const uint32_t gy = 256u - fy;
    const auto one = [&](int64_t ox) {
        if (inv_x[ox]) {
            dst[ox] = 0;
            return;
        }
        const uint32_t fxx = fx[ox];
        const uint32_t gxx = 256u - fxx;
        const uint32_t top = static_cast<uint32_t>(r0[ix[ox]]) * gxx + static_cast<uint32_t>(r0[jx[ox]]) * fxx;
        const uint32_t bot = static_cast<uint32_t>(r1[ix[ox]]) * gxx + static_cast<uint32_t>(r1[jx[ox]]) * fxx;
        dst[ox] = static_cast<T>((top * gy + bot * fy) >> 16u);
    };
    int64_t ox = 0;
    for (; ox + 8 <= rw; ox += 8) {
        if (dense_columns(ix + ox, jx + ox, inv_x + ox)) {
            dense_gray8<T>(r0 + ix[ox], r1 + ix[ox], dst + ox,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fx + ox)), v_fy);
            continue;
        }
        for (int64_t k = ox; k < ox + 8; ++k) {
            one(k);
        }
    }
    for (; ox < rw; ++ox) {
        one(ox);
    }
}

}  // namespace

void align_warp_translation_gray_row_avx2(
    const void* row0, const void* row1, bool is_u8, void* out_row, int64_t rw,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x,
    uint32_t fy
) {
    if (is_u8) {
        warp_translation_gray_row<uint8_t>(
            static_cast<const uint8_t*>(row0), static_cast<const uint8_t*>(row1),
            static_cast<uint8_t*>(out_row), rw, ix, jx, fx, inv_x, fy);
    } else {
        warp_translation_gray_row<uint16_t>(
            static_cast<const uint16_t*>(row0), static_cast<const uint16_t*>(row1),
            static_cast<uint16_t*>(out_row), rw, ix, jx, fx, inv_x, fy);
    }
}

void align_refine_dense_row_avx2(
    const uint16_t* a_row, const uint16_t* b_row0, const uint16_t* b_row1,
    const int32_t* weights, int64_t count, uint32_t fy, int64_t* sum, int64_t* sq
) {
    const __m256i bias = _mm256_set1_epi16(static_cast<int16_t>(0x8000));
    const __m256i unbias = _mm256_set1_epi32(1 << 23);
    const __m256i v_fy = _mm256_set1_epi32(static_cast<int32_t>(fy));
    __m256i v_sum = _mm256_setzero_si256();
    __m256i v_sq_even = _mm256_setzero_si256();
    __m256i v_sq_odd = _mm256_setzero_si256();
    for (int64_t k = 0; k + 8 <= count; k += 8) {
        const auto pairs = [k](const uint16_t* row) {
            const __m256i left = _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k)));
            const __m256i right = _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k + 1)));
            return _mm256_or_si256(left, _mm256_slli_epi32(right, 16));
        };
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + k));
        const __m256i top = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_xor_si256(pairs(b_row0), bias), w), unbias);
        const __m256i bot = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_xor_si256(pairs(b_row1), bias), w), unbias);
        const __m256i val = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(top, 8),
                             _mm256_mullo_epi32(_mm256_sub_epi32(bot, top), v_fy)), 16);
        const __m256i ref = _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_row + k)));
        const __m256i diff = _mm256_sub_epi32(ref, val);
        v_sum = _mm256_add_epi32(v_sum, diff);
        const __m256i diff_odd = _mm256_srli_epi64(diff, 32);
        v_sq_even = _mm256_add_epi64(v_sq_even, _mm256_mul_epi32(diff, diff));
        v_sq_odd = _mm256_add_epi64(v_sq_odd, _mm256_mul_epi32(diff_odd, diff_odd));
    }
    alignas(32) int32_t red_sum[8];
    alignas(32) int64_t red_sq[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sum), v_sum);
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sq), _mm256_add_epi64(v_sq_even, v_sq_odd));
    int64_t total = 0;
    for (int m = 0; m < 8; ++m) {
        total += red_sum[m];
    }
    *sum += total;
    *sq += red_sq[0] + red_sq[1] + red_sq[2] + red_sq[3];
}

void align_valid_mask_row_avx2(
    uint8_t* mr, int64_t w,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy,
    float wf, float hf
) {
    const __m256 v_cx = _mm256_set1_ps(cx);
    const __m256 v_p08 = _mm256_set1_ps(p08);
    const __m256 v_p18 = _mm256_set1_ps(p18);
    const __m256 v_tx = _mm256_set1_ps(tx);
    const __m256 v_ty = _mm256_set1_ps(ty);
    const __m256 v_vdy = _mm256_set1_ps(v_dy);
    const __m256 v_udy = _mm256_set1_ps(u_dy);
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_step = _mm256_set1_ps(8.0f);
    const __m256i v_one = _mm256_set1_epi32(1);
    __m256 vx = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256 dx = _mm256_sub_ps(vx, v_cx);
        const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx, v_p08)), v_ty);
        const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, v_p18), v_udy), v_tx);
        const __m256 ok_u = _mm256_and_ps(
            _mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_wf, _CMP_LT_OQ));
        const __m256 ok_v = _mm256_and_ps(
            _mm256_cmp_ps(v, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(v, v_hf, _CMP_LT_OQ));
        const __m256i bits = _mm256_and_si256(
            _mm256_castps_si256(_mm256_and_ps(ok_u, ok_v)), v_one);
        const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(bits, bits), 0xD8);
        const __m128i lo16 = _mm256_castsi256_si128(p16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(mr + x), _mm_packus_epi16(lo16, lo16));
        vx = _mm256_add_ps(vx, v_step);
    }
    for (; x < w; ++x) {
        const float dx_ = static_cast<float>(x) - cx;
        const float v = (v_dy - (dx_ * p08)) + ty;
        const float u = ((dx_ * p18) + u_dy) + tx;
        mr[x] = (u >= 0.0f && u < wf && v >= 0.0f && v < hf) ? 1 : 0;
    }
}

void align_gray16_rgb_row_avx2(
    const void* src, bool is_u8, uint16_t* dst, int64_t w, int64_t src_px_stride
) {
    int64_t x = 0;
    if (is_u8) {
        const uint8_t* sr = static_cast<const uint8_t*>(src);
        if (src_px_stride == 4) {
            const __m256i m8 = _mm256_set1_epi32(0xFF);
            for (; x + 8 <= w; x += 8) {
                const __m256i px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x * 4));
                const __m256i r = _mm256_and_si256(px, m8);
                const __m256i g = _mm256_and_si256(_mm256_srli_epi32(px, 8), m8);
                const __m256i b = _mm256_and_si256(_mm256_srli_epi32(px, 16), m8);
                const __m256i q = div3_epu32_avx2(_mm256_add_epi32(_mm256_add_epi32(r, g), b));
                store_u32x8_as_u16(_mm256_slli_epi32(q, 8), dst + x);
            }
        } else {
            const __m128i shuf_r = _mm_setr_epi8(0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1);
            const __m128i shuf_g = _mm_setr_epi8(1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1);
            const __m128i shuf_b = _mm_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1);
            for (; x + 10 <= w; x += 8) {
                const uint8_t* p = sr + x * 3;
                const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
                const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 12));
                const __m256i vr = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_r), _mm_shuffle_epi8(a, shuf_r));
                const __m256i vg = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_g), _mm_shuffle_epi8(a, shuf_g));
                const __m256i vb = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_b), _mm_shuffle_epi8(a, shuf_b));
                const __m256i q = div3_epu32_avx2(_mm256_add_epi32(_mm256_add_epi32(vr, vg), vb));
                store_u32x8_as_u16(_mm256_slli_epi32(q, 8), dst + x);
            }
        }
        for (; x < w; ++x) {
            const uint8_t* p = sr + x * src_px_stride;
            const uint32_t sum = static_cast<uint32_t>(p[0]) + p[1] + p[2];
            dst[x] = static_cast<uint16_t>((sum / 3u) << 8u);
        }
        return;
    }

    const uint16_t* sr = static_cast<const uint16_t*>(src);
    if (src_px_stride == 4) {
        const __m256i m16 = _mm256_set1_epi32(0xFFFF);
        const __m128i shuf_b = _mm_setr_epi8(
            4, 5, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        for (; x + 4 <= w; x += 4) {
            const uint16_t* p = sr + x * 4;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m128i lo = _mm256_castsi256_si128(raw);
            const __m128i hi = _mm256_extracti128_si256(raw, 1);
            // Deinterleave 4 RGBA u16 pixels in lower 128 bits via 64-bit chunks
            const __m128i rg = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(lo), _mm_castsi128_ps(hi), 0x88));
            const __m128i ba = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(lo), _mm_castsi128_ps(hi), 0xDD));
            const __m128i r = _mm_and_si128(rg, _mm256_castsi256_si128(m16));
            const __m128i g = _mm_srli_epi32(rg, 16);
            const __m128i b = _mm_and_si128(ba, _mm256_castsi256_si128(m16));
            const __m128i sum = _mm_add_epi32(_mm_add_epi32(r, g), b);
            const __m128i vm = _mm_set1_epi32(static_cast<int32_t>(kDiv3Magic));
            const __m128i qe = _mm_srli_epi64(_mm_mul_epu32(sum, vm), 33);
            const __m128i qo = _mm_srli_epi32(_mm_mul_epu32(_mm_srli_epi64(sum, 32), vm), 1);
            const __m128i q = _mm_blend_epi32(qe, qo, 0xA);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), _mm_packus_epi32(q, q));
        }
        (void)shuf_b;
    } else {
        const __m128i z = _mm_setzero_si128();
        const __m128i sr0 = _mm_setr_epi8(0, 1, -1, -1, 6, 7, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1);
        const __m128i sg0 = _mm_setr_epi8(2, 3, -1, -1, 8, 9, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1);
        const __m128i sb0 = _mm_setr_epi8(4, 5, -1, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i px3_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, -1, -1);
        const __m128i px3_g = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1);
        const __m128i px23_b = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 6, 7, -1, -1);
        const __m128i vm = _mm_set1_epi32(static_cast<int32_t>(kDiv3Magic));
        (void)z;
        for (; x + 4 <= w; x += 4) {
            const uint16_t* p = sr + x * 3;
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 8));
            const __m128i vr = _mm_or_si128(_mm_shuffle_epi8(a, sr0), _mm_shuffle_epi8(b, px3_r));
            const __m128i vg = _mm_or_si128(_mm_shuffle_epi8(a, sg0), _mm_shuffle_epi8(b, px3_g));
            const __m128i vb = _mm_or_si128(_mm_shuffle_epi8(a, sb0), _mm_shuffle_epi8(b, px23_b));
            const __m128i sum = _mm_add_epi32(_mm_add_epi32(vr, vg), vb);
            const __m128i qe = _mm_srli_epi64(_mm_mul_epu32(sum, vm), 33);
            const __m128i qo = _mm_srli_epi32(_mm_mul_epu32(_mm_srli_epi64(sum, 32), vm), 1);
            const __m128i q = _mm_blend_epi32(qe, qo, 0xA);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), _mm_packus_epi32(q, q));
        }
    }
    for (; x < w; ++x) {
        const uint16_t* p = sr + x * src_px_stride;
        const uint32_t sum = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        dst[x] = static_cast<uint16_t>(sum / 3u);
    }
}

void align_down_half_vert_row_avx2(
    const uint16_t* r0, const uint16_t* r1, const uint16_t* r2, const uint16_t* r3,
    uint32_t* scratch, int64_t w
) {
    const __m256i vw0 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW0));
    const __m256i vw1 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW1));
    const __m256i vw2 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW2));
    const __m256i vw3 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW3));
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256i u0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + x)));
        const __m256i u1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + x)));
        const __m256i u2 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r2 + x)));
        const __m256i u3 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r3 + x)));
        const __m256i t0 = _mm256_mullo_epi32(u0, vw0);
        const __m256i t1 = _mm256_mullo_epi32(u1, vw1);
        const __m256i t2 = _mm256_mullo_epi32(u2, vw2);
        const __m256i t3 = _mm256_mullo_epi32(u3, vw3);
        const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(t0, t1), _mm256_add_epi32(t2, t3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(scratch + x), div115_round_epu32_avx2(sum));
    }
    for (; x < w; ++x) {
        const uint32_t v = static_cast<uint32_t>(r0[x]) * kAlignDownW0
                         + static_cast<uint32_t>(r1[x]) * kAlignDownW1
                         + static_cast<uint32_t>(r2[x]) * kAlignDownW2
                         + static_cast<uint32_t>(r3[x]) * kAlignDownW3;
        scratch[x] = (v + 57u) / 115u;
    }
}

void align_down_half_horiz_row_avx2(
    const uint32_t* scratch, uint16_t* out, int64_t ow, int64_t w
) {
    if (ow <= 0) {
        return;
    }
    auto clamp_idx = [w](int64_t idx) -> int64_t {
        return idx < 0 ? 0 : (idx >= w ? w - 1 : idx);
    };
    {
        const uint32_t v = scratch[0] * kAlignDownW0
                         + scratch[0] * kAlignDownW1
                         + scratch[clamp_idx(1)] * kAlignDownW2
                         + scratch[clamp_idx(2)] * kAlignDownW3;
        out[0] = static_cast<uint16_t>((v + 57u) / 115u);
    }
    const __m256i vw0 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW0));
    const __m256i vw1 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW1));
    const __m256i vw2 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW2));
    const __m256i vw3 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW3));
    int64_t ox = 1;
    for (; ox + 8 <= ow && (2 * ox + 16 < w); ox += 8) {
        const uint32_t* sp = scratch + (2 * ox - 1);
        const __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp));
        const __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 8));
        const __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 2));
        const __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 10));
        const __m256i sx0 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0x88)), 0xD8);
        const __m256i sx1 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0xDD)), 0xD8);
        const __m256i sx2 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v2, v3, 0x88)), 0xD8);
        const __m256i sx3 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v2, v3, 0xDD)), 0xD8);
        const __m256i sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(sx0, vw0), _mm256_mullo_epi32(sx1, vw1)),
            _mm256_add_epi32(_mm256_mullo_epi32(sx2, vw2), _mm256_mullo_epi32(sx3, vw3)));
        store_u32x8_as_u16(div115_round_epu32_avx2(sum), out + ox);
    }
    for (; ox < ow; ++ox) {
        const uint32_t v = scratch[clamp_idx(2 * ox - 1)] * kAlignDownW0
                         + scratch[clamp_idx(2 * ox)] * kAlignDownW1
                         + scratch[clamp_idx(2 * ox + 1)] * kAlignDownW2
                         + scratch[clamp_idx(2 * ox + 2)] * kAlignDownW3;
        out[ox] = static_cast<uint16_t>((v + 57u) / 115u);
    }
}

void align_fast_downsize_u16_row_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* dst, int64_t ow, int64_t w
) {
    const __m256i m16 = _mm256_set1_epi32(0xFFFF);
    const __m256i v2 = _mm256_set1_epi32(2);
    int64_t ox = 0;
    for (; ox + 8 <= ow && (2 * ox + 16 <= w); ox += 8) {
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * ox));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * ox));
        const __m256i s0 = _mm256_add_epi32(_mm256_and_si256(p0, m16), _mm256_srli_epi32(p0, 16));
        const __m256i s1 = _mm256_add_epi32(_mm256_and_si256(p1, m16), _mm256_srli_epi32(p1, 16));
        const __m256i avg = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(s0, s1), v2), 2);
        store_u32x8_as_u16(avg, dst + ox);
    }
    for (; ox < ow; ++ox) {
        const int64_t x0 = 2 * ox;
        const int64_t x1 = (x0 + 1 < w) ? (x0 + 1) : (w - 1);
        dst[ox] = static_cast<uint16_t>(
            (static_cast<uint32_t>(r0[x0]) + r0[x1] + r1[x0] + r1[x1] + 2u) >> 2u);
    }
}

namespace {

// One output row of the similarity warp for one sample type, channel count and
// channel order. The coordinates follow the scalar sequence eight columns at a
// time; chunks whose taps are consecutive columns of one pair of rows take the
// dense path, the others the per-pixel one.
template <typename T, int Channels, bool BgrRev>
void warp_alignment_row(
    const T* img, int64_t img_s0, T* out_row,
    int64_t src_h, int64_t src_w, int64_t src_y0, int64_t x_off, int64_t rw,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy
) {
    const float wf = static_cast<float>(static_cast<int32_t>(src_w));
    const float hf = static_cast<float>(static_cast<int32_t>(src_h));
    const float w_m1 = static_cast<float>(static_cast<int32_t>(src_w) - 1);
    const float h_m1 = static_cast<float>(static_cast<int32_t>(src_h) - 1);
    const float u_clamp = wf - kAlignClampEdge;
    const float v_clamp = hf - kAlignClampEdge;

    const __m256 v_cx = _mm256_set1_ps(cx);
    const __m256 v_p08 = _mm256_set1_ps(p08);
    const __m256 v_p18 = _mm256_set1_ps(p18);
    const __m256 v_tx = _mm256_set1_ps(tx);
    const __m256 v_ty = _mm256_set1_ps(ty);
    const __m256 v_vdy = _mm256_set1_ps(v_dy);
    const __m256 v_udy = _mm256_set1_ps(u_dy);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_wm1 = _mm256_set1_ps(w_m1);
    const __m256 v_hm1 = _mm256_set1_ps(h_m1);
    const __m256 v_uc = _mm256_set1_ps(u_clamp);
    const __m256 v_vc = _mm256_set1_ps(v_clamp);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256 v_step = _mm256_set1_ps(8.0f);
    const __m256i vi_zero = _mm256_setzero_si256();
    const __m256i vi_one = _mm256_set1_epi32(1);
    const __m256i vi_wm1 = _mm256_set1_epi32(static_cast<int32_t>(src_w - 1));
    const __m256i vi_hm1 = _mm256_set1_epi32(static_cast<int32_t>(src_h - 1));

    const float x0_f = static_cast<float>(x_off);
    __m256 vx = _mm256_add_ps(
        _mm256_set1_ps(x0_f),
        _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));

    alignas(32) int32_t ix_buf[8], jx_buf[8], iy_buf[8], jy_buf[8];
    alignas(32) uint32_t fx_buf[8], fy_buf[8];
    const int32_t chunk_len_x = static_cast<int32_t>(src_w);

    int64_t ox = 0;
    for (; ox + 8 <= rw; ox += 8) {
        const __m256 dx = _mm256_sub_ps(vx, v_cx);
        const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx, v_p08)), v_ty);
        const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, v_p18), v_udy), v_tx);

        const __m256 inv_u = _mm256_or_ps(_mm256_cmp_ps(u, v_wf, _CMP_GE_OQ), _mm256_cmp_ps(u, v_zero, _CMP_LT_OQ));
        const __m256 inv_v = _mm256_or_ps(_mm256_cmp_ps(v, v_hf, _CMP_GE_OQ), _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        const int inv_mask = _mm256_movemask_ps(_mm256_or_ps(inv_u, inv_v));

        vx = _mm256_add_ps(vx, v_step);
        if (inv_mask == 0xFF) {
            std::memset(out_row + ox * Channels, 0, sizeof(T) * 8u * Channels);
            continue;
        }

        __m256 uc = _mm256_blendv_ps(u, v_uc, _mm256_cmp_ps(u, v_wm1, _CMP_GE_OQ));
        uc = _mm256_blendv_ps(uc, v_zero, _mm256_cmp_ps(uc, v_zero, _CMP_LT_OQ));
        const __m256i uidx = _mm256_cvttps_epi32(uc);
        const __m256i ufx = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(uc, _mm256_cvtepi32_ps(uidx)), v_256));
        const __m256i uix = _mm256_min_epi32(_mm256_max_epi32(uidx, vi_zero), vi_wm1);
        const __m256i ujx = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(uidx, vi_one), vi_zero), vi_wm1);

        __m256 vc = _mm256_blendv_ps(v, v_vc, _mm256_cmp_ps(v, v_hm1, _CMP_GE_OQ));
        vc = _mm256_blendv_ps(vc, v_zero, _mm256_cmp_ps(vc, v_zero, _CMP_LT_OQ));
        const __m256i vidx = _mm256_cvttps_epi32(vc);
        const __m256i vfy = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(vc, _mm256_cvtepi32_ps(vidx)), v_256));
        const __m256i viy = _mm256_min_epi32(_mm256_max_epi32(vidx, vi_zero), vi_hm1);
        const __m256i vjy = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vidx, vi_one), vi_zero), vi_hm1);

        if (inv_mask == 0 && dense_chunk(uix, ujx, viy, vjy)) {
            const int64_t first = static_cast<int64_t>(lane0_epi32(uix)) * Channels;
            const T* s0 = img + (static_cast<int64_t>(lane0_epi32(viy)) - src_y0) * img_s0 + first;
            const T* s1 = img + (static_cast<int64_t>(lane0_epi32(vjy)) - src_y0) * img_s0 + first;
            if constexpr (Channels == 1) {
                dense_gray8<T>(s0, s1, out_row + ox, ufx, vfy);
            } else {
                dense_rgb8<T, BgrRev, false>(s0, s1, out_row + ox * 3, ufx, vfy);
            }
            continue;
        }

        _mm256_store_si256(reinterpret_cast<__m256i*>(ix_buf), uix);
        _mm256_store_si256(reinterpret_cast<__m256i*>(jx_buf), ujx);
        _mm256_store_si256(reinterpret_cast<__m256i*>(iy_buf), viy);
        _mm256_store_si256(reinterpret_cast<__m256i*>(jy_buf), vjy);
        _mm256_store_si256(reinterpret_cast<__m256i*>(fx_buf), ufx);
        _mm256_store_si256(reinterpret_cast<__m256i*>(fy_buf), vfy);
        warp_chunk_8<T, Channels, BgrRev>(img, img_s0, src_y0, out_row, ox, ix_buf, jx_buf, iy_buf, jy_buf,
                                          fx_buf, fy_buf, inv_mask, chunk_len_x, ox + 8 < rw);
    }

    auto clamp_i32 = [](int32_t v, int32_t hi) -> int32_t {
        return v < 0 ? 0 : (v > hi ? hi : v);
    };
    const int32_t len_x = static_cast<int32_t>(src_w);
    const int32_t len_y = static_cast<int32_t>(src_h);
    for (; ox < rw; ++ox) {
        const float dx_ = static_cast<float>(x_off + ox) - cx;
        const float v = (v_dy - (dx_ * p08)) + ty;
        const float u = ((dx_ * p18) + u_dy) + tx;
        const bool invalid = (u >= wf || u < 0.0f || v >= hf || v < 0.0f);
        if (invalid) {
            for (int c = 0; c < Channels; ++c) {
                out_row[ox * Channels + c] = 0;
            }
            continue;
        }
        float uc = (u >= w_m1) ? u_clamp : u;
        if (uc < 0.0f) uc = 0.0f;
        const int32_t uidx = static_cast<int32_t>(uc);
        const uint32_t fxx = static_cast<uint32_t>((uc - static_cast<float>(uidx)) * 256.0f);
        const uint32_t gxx = 256u - fxx;
        const int64_t ixx = static_cast<int64_t>(clamp_i32(uidx, len_x - 1)) * Channels;
        const int64_t jxx = static_cast<int64_t>(clamp_i32(uidx + 1, len_x - 1)) * Channels;

        float vc = (v >= h_m1) ? v_clamp : v;
        if (vc < 0.0f) vc = 0.0f;
        const int32_t vidx = static_cast<int32_t>(vc);
        const uint32_t fyy = static_cast<uint32_t>((vc - static_cast<float>(vidx)) * 256.0f);
        const uint32_t gyy = 256u - fyy;
        const T* r0 = img + (static_cast<int64_t>(clamp_i32(vidx, len_y - 1)) - src_y0) * img_s0;
        const T* r1 = img + (static_cast<int64_t>(clamp_i32(vidx + 1, len_y - 1)) - src_y0) * img_s0;

        for (int c = 0; c < Channels; ++c) {
            const int sc = (BgrRev && Channels == 3) ? (2 - c) : c;
            const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
            const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
            out_row[ox * Channels + c] = static_cast<T>((top * gyy + bot * fyy) >> 16u);
        }
    }
}

}  // namespace

void align_warp_alignment_row_avx2(
    const void* img_data, int64_t img_s0, int64_t img_s1, bool is_u8,
    void* out_row, int64_t out_s1,
    int64_t src_h, int64_t src_w, int64_t src_y0, int64_t x_off, int64_t rw,
    int64_t channels, bool bgr_rev,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy
) {
    (void)img_s1;
    (void)out_s1;
    const auto run = [&](auto* src, auto* dst, auto row) {
        row(src, img_s0, dst, src_h, src_w, src_y0, x_off, rw, cx, p08, p18, tx, ty, v_dy, u_dy);
    };
    if (is_u8) {
        const auto* src = static_cast<const uint8_t*>(img_data);
        auto* dst = static_cast<uint8_t*>(out_row);
        if (channels == 1) {
            run(src, dst, warp_alignment_row<uint8_t, 1, false>);
        } else if (bgr_rev) {
            run(src, dst, warp_alignment_row<uint8_t, 3, true>);
        } else {
            run(src, dst, warp_alignment_row<uint8_t, 3, false>);
        }
    } else {
        const auto* src = static_cast<const uint16_t*>(img_data);
        auto* dst = static_cast<uint16_t*>(out_row);
        if (channels == 1) {
            run(src, dst, warp_alignment_row<uint16_t, 1, false>);
        } else if (bgr_rev) {
            run(src, dst, warp_alignment_row<uint16_t, 3, true>);
        } else {
            run(src, dst, warp_alignment_row<uint16_t, 3, false>);
        }
    }
}

namespace {

inline void score_block_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx_k, float ty_k, float cx_k, float cy_k,
    float p08_k, float p0c_k, float p18_k, float p1c_k,
    int64_t& out_sum, int64_t& out_sq, int64_t& out_cnt
) {
    const int32_t len_x = static_cast<int32_t>(w);
    const int32_t len_y = static_cast<int32_t>(h);
    const float wf = static_cast<float>(len_x);
    const float hf = static_cast<float>(len_y);
    const float w_m1 = static_cast<float>(len_x - 1);
    const float h_m1 = static_cast<float>(len_y - 1);
    const float u_clamp = wf - kAlignClampEdge;
    const float v_clamp = hf - kAlignClampEdge;

    const __m256 v_cx = _mm256_set1_ps(cx_k);
    const __m256 v_p08 = _mm256_set1_ps(p08_k);
    const __m256 v_p18 = _mm256_set1_ps(p18_k);
    const __m256 v_tx = _mm256_set1_ps(tx_k);
    const __m256 v_ty = _mm256_set1_ps(ty_k);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_wm1 = _mm256_set1_ps(w_m1);
    const __m256 v_hm1 = _mm256_set1_ps(h_m1);
    const __m256 v_uc = _mm256_set1_ps(u_clamp);
    const __m256 v_vc = _mm256_set1_ps(v_clamp);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256i vi_zero = _mm256_setzero_si256();
    const __m256i vi_one = _mm256_set1_epi32(1);
    const __m256i vi_256 = _mm256_set1_epi32(256);
    const __m256i vi_wm1 = _mm256_set1_epi32(len_x - 1);
    const __m256i vi_hm1 = _mm256_set1_epi32(len_y - 1);

    __m256i v_sum_lo = _mm256_setzero_si256();
    __m256i v_sum_hi = _mm256_setzero_si256();
    __m256i v_sq_lo = _mm256_setzero_si256();
    __m256i v_sq_hi = _mm256_setzero_si256();
    int64_t scalar_sum = 0;
    int64_t scalar_sq = 0;
    int64_t acc_cnt = 0;

    alignas(32) int32_t ix_buf[8], jx_buf[8], iy_buf[8], jy_buf[8];
    alignas(32) int32_t fx_buf[8], fy_buf[8], a0_buf[8], a1_buf[8], b0_buf[8], b1_buf[8];

    auto clamp_i32 = [](int32_t v, int32_t hi) -> int32_t {
        return v < 0 ? 0 : (v > hi ? hi : v);
    };

    for (int64_t iy_i = 0; iy_i < nyy; ++iy_i) {
        const float dy_ = ys[iy_i] - cy_k;
        const float v_dy = dy_ * p1c_k;
        const float u_dy = dy_ * p0c_k;
        const __m256 v_vdy = _mm256_set1_ps(v_dy);
        const __m256 v_udy = _mm256_set1_ps(u_dy);
        const uint16_t* rrow = rb + iy_i * ref_sy;

        int64_t ix_i = 0;
        for (; ix_i + 8 <= nxx; ix_i += 8) {
            const __m256 dx_ = _mm256_sub_ps(_mm256_loadu_ps(xs + ix_i), v_cx);
            const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx_, v_p08)), v_ty);
            const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx_, v_p18), v_udy), v_tx);

            const __m256 ok_u = _mm256_and_ps(
                _mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_wf, _CMP_LT_OQ));
            const __m256 ok_v = _mm256_and_ps(
                _mm256_cmp_ps(v, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(v, v_hf, _CMP_LT_OQ));
            const int valid_mask = _mm256_movemask_ps(_mm256_and_ps(ok_u, ok_v));
            if (valid_mask == 0) {
                continue;
            }

            __m256 uc = _mm256_blendv_ps(u, v_uc, _mm256_cmp_ps(u, v_wm1, _CMP_GE_OQ));
            uc = _mm256_blendv_ps(uc, v_zero, _mm256_cmp_ps(uc, v_zero, _CMP_LT_OQ));
            const __m256i uidx = _mm256_cvttps_epi32(uc);
            const __m256i vfx = _mm256_cvttps_epi32(
                _mm256_mul_ps(_mm256_sub_ps(uc, _mm256_cvtepi32_ps(uidx)), v_256));
            const __m256i vix = _mm256_min_epi32(_mm256_max_epi32(uidx, vi_zero), vi_wm1);
            const __m256i vjx = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(uidx, vi_one), vi_zero), vi_wm1);

            __m256 vc = _mm256_blendv_ps(v, v_vc, _mm256_cmp_ps(v, v_hm1, _CMP_GE_OQ));
            vc = _mm256_blendv_ps(vc, v_zero, _mm256_cmp_ps(vc, v_zero, _CMP_LT_OQ));
            const __m256i vidx = _mm256_cvttps_epi32(vc);
            const __m256i vfy = _mm256_cvttps_epi32(
                _mm256_mul_ps(_mm256_sub_ps(vc, _mm256_cvtepi32_ps(vidx)), v_256));
            const __m256i viy = _mm256_min_epi32(_mm256_max_epi32(vidx, vi_zero), vi_hm1);
            const __m256i vjy = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vidx, vi_one), vi_zero), vi_hm1);

            _mm256_store_si256(reinterpret_cast<__m256i*>(ix_buf), vix);
            _mm256_store_si256(reinterpret_cast<__m256i*>(jx_buf), vjx);
            _mm256_store_si256(reinterpret_cast<__m256i*>(iy_buf), viy);
            _mm256_store_si256(reinterpret_cast<__m256i*>(jy_buf), vjy);

            if (valid_mask == 0xFF) {
                for (int m = 0; m < 8; ++m) {
                    const uint16_t* br0 = bb + static_cast<int64_t>(iy_buf[m]) * b_sy;
                    const uint16_t* br1 = bb + static_cast<int64_t>(jy_buf[m]) * b_sy;
                    a0_buf[m] = static_cast<int32_t>(br0[ix_buf[m]]);
                    a1_buf[m] = static_cast<int32_t>(br0[jx_buf[m]]);
                    b0_buf[m] = static_cast<int32_t>(br1[ix_buf[m]]);
                    b1_buf[m] = static_cast<int32_t>(br1[jx_buf[m]]);
                }
                const __m256i vgx = _mm256_sub_epi32(vi_256, vfx);
                const __m256i vgy = _mm256_sub_epi32(vi_256, vfy);
                const __m256i top = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(a0_buf)), vgx),
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(a1_buf)), vfx));
                const __m256i bot = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(b0_buf)), vgx),
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(b1_buf)), vfx));
                const __m256i val = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(top, vgy), _mm256_mullo_epi32(bot, vfy)), 16);
                const __m256i vref = _mm256_cvtepu16_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(rrow + ix_i)));
                const __m256i diff = _mm256_sub_epi32(vref, val);

                v_sum_lo = _mm256_add_epi64(
                    v_sum_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(diff)));
                v_sum_hi = _mm256_add_epi64(
                    v_sum_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(diff, 1)));

                const __m256i diff_odd = _mm256_srli_epi64(diff, 32);
                v_sq_lo = _mm256_add_epi64(v_sq_lo, _mm256_mul_epi32(diff, diff));
                v_sq_hi = _mm256_add_epi64(v_sq_hi, _mm256_mul_epi32(diff_odd, diff_odd));
                acc_cnt += 8;
            } else {
                _mm256_store_si256(reinterpret_cast<__m256i*>(fx_buf), vfx);
                _mm256_store_si256(reinterpret_cast<__m256i*>(fy_buf), vfy);
                for (int m = 0; m < 8; ++m) {
                    if (!((valid_mask >> m) & 1)) {
                        continue;
                    }
                    const uint32_t fx = static_cast<uint32_t>(fx_buf[m]);
                    const uint32_t fy = static_cast<uint32_t>(fy_buf[m]);
                    const uint32_t gx = 256u - fx;
                    const uint32_t gy = 256u - fy;
                    const uint16_t* br0 = bb + static_cast<int64_t>(iy_buf[m]) * b_sy;
                    const uint16_t* br1 = bb + static_cast<int64_t>(jy_buf[m]) * b_sy;
                    const uint32_t top = static_cast<uint32_t>(br0[ix_buf[m]]) * gx
                                       + static_cast<uint32_t>(br0[jx_buf[m]]) * fx;
                    const uint32_t bot = static_cast<uint32_t>(br1[ix_buf[m]]) * gx
                                       + static_cast<uint32_t>(br1[jx_buf[m]]) * fx;
                    const uint32_t val = (top * gy + bot * fy) >> 16u;
                    const int64_t diff = static_cast<int64_t>(rrow[ix_i + m]) - static_cast<int64_t>(val);
                    scalar_sum += diff;
                    scalar_sq += diff * diff;
                    acc_cnt += 1;
                }
            }
        }

        for (; ix_i < nxx; ++ix_i) {
            const float dx_ = xs[ix_i] - cx_k;
            const float v = (v_dy - (dx_ * p08_k)) + ty_k;
            const float u = ((dx_ * p18_k) + u_dy) + tx_k;
            if (!(u >= 0.0f && u < wf && v >= 0.0f && v < hf)) {
                continue;
            }
            float uc = (u >= w_m1) ? u_clamp : u;
            if (uc < 0.0f) uc = 0.0f;
            const int32_t uidx = static_cast<int32_t>(uc);
            const uint32_t fx = static_cast<uint32_t>((uc - static_cast<float>(uidx)) * 256.0f);
            const int32_t ix = clamp_i32(uidx, len_x - 1);
            const int32_t jx = clamp_i32(uidx + 1, len_x - 1);

            float vc = (v >= h_m1) ? v_clamp : v;
            if (vc < 0.0f) vc = 0.0f;
            const int32_t vidx = static_cast<int32_t>(vc);
            const uint32_t fy = static_cast<uint32_t>((vc - static_cast<float>(vidx)) * 256.0f);
            const int32_t iy = clamp_i32(vidx, len_y - 1);
            const int32_t jy = clamp_i32(vidx + 1, len_y - 1);

            const uint32_t gx = 256u - fx;
            const uint32_t gy = 256u - fy;
            const uint16_t* br0 = bb + static_cast<int64_t>(iy) * b_sy;
            const uint16_t* br1 = bb + static_cast<int64_t>(jy) * b_sy;
            const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
            const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
            const uint32_t val = (top * gy + bot * fy) >> 16u;
            const int64_t diff = static_cast<int64_t>(rrow[ix_i]) - static_cast<int64_t>(val);
            scalar_sum += diff;
            scalar_sq += diff * diff;
            acc_cnt += 1;
        }
    }

    alignas(32) int64_t red_sum[4], red_sq[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sum), _mm256_add_epi64(v_sum_lo, v_sum_hi));
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sq), _mm256_add_epi64(v_sq_lo, v_sq_hi));
    out_sum = scalar_sum + red_sum[0] + red_sum[1] + red_sum[2] + red_sum[3];
    out_sq = scalar_sq + red_sq[0] + red_sq[1] + red_sq[2] + red_sq[3];
    out_cnt = acc_cnt;
}

// The sample grid flattened row by row and padded to whole vectors, so every
// candidate transform is scored without a per-row scalar remainder.
struct FlatScoreGrid {
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<uint16_t> ref;
    int64_t count = 0;
    bool usable = false;
};

inline FlatScoreGrid make_flat_score_grid(
    const uint16_t* rb, int64_t ref_sy, const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy
) {
    (void)bb;
    FlatScoreGrid grid;
    const int64_t count = nxx * nyy;
    // Bilinear taps are read as adjacent pairs through int32 offsets, and
    // per-lane sums stay in 32 bits.
    const float u_clamp = static_cast<float>(static_cast<int32_t>(w)) - kAlignClampEdge;
    grid.usable = w >= 2 && h >= 1 && b_sy >= w && count >= 1 && count <= 131072
        && h * b_sy < (int64_t{1} << 30)
        && static_cast<int64_t>(u_clamp) <= w - 2;
    if (!grid.usable) {
        return grid;
    }
    const size_t padded = static_cast<size_t>((count + 7) & ~int64_t{7});
    grid.count = count;
    grid.xs.assign(padded, 0.0f);
    grid.ys.assign(padded, 0.0f);
    grid.ref.assign(padded, uint16_t{0});
    for (int64_t j = 0; j < nyy; ++j) {
        for (int64_t i = 0; i < nxx; ++i) {
            const size_t n = static_cast<size_t>(j * nxx + i);
            grid.xs[n] = xs[i];
            grid.ys[n] = ys[j];
            grid.ref[n] = rb[j * ref_sy + i];
        }
    }
    return grid;
}

inline int32_t load_u16_pair(const uint16_t* p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Same sums as score_block_packed_avx2. With gx + fx == 256 the horizontal
// blend a0 * gx + a1 * fx is one signed multiply-add of the biased sample pair,
// and top * gy + bot * fy equals (top << 8) + (bot - top) * fy modulo 2^32.
inline void score_block_flat_avx2(
    const FlatScoreGrid& grid,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    float tx_k, float ty_k, float cx_k, float cy_k,
    float p08_k, float p0c_k, float p18_k, float p1c_k,
    int64_t& out_sum, int64_t& out_sq, int64_t& out_cnt
) {
    const int32_t len_x = static_cast<int32_t>(w);
    const int32_t len_y = static_cast<int32_t>(h);
    const float wf = static_cast<float>(len_x);
    const float hf = static_cast<float>(len_y);

    const __m256 v_cx = _mm256_set1_ps(cx_k);
    const __m256 v_cy = _mm256_set1_ps(cy_k);
    const __m256 v_p08 = _mm256_set1_ps(p08_k);
    const __m256 v_p0c = _mm256_set1_ps(p0c_k);
    const __m256 v_p18 = _mm256_set1_ps(p18_k);
    const __m256 v_p1c = _mm256_set1_ps(p1c_k);
    const __m256 v_tx = _mm256_set1_ps(tx_k);
    const __m256 v_ty = _mm256_set1_ps(ty_k);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_wm1 = _mm256_set1_ps(static_cast<float>(len_x - 1));
    const __m256 v_hm1 = _mm256_set1_ps(static_cast<float>(len_y - 1));
    const __m256 v_uc = _mm256_set1_ps(wf - kAlignClampEdge);
    const __m256 v_vc = _mm256_set1_ps(hf - kAlignClampEdge);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256i vi_zero = _mm256_setzero_si256();
    const __m256i vi_one = _mm256_set1_epi32(1);
    const __m256i vi_256 = _mm256_set1_epi32(256);
    const __m256i vi_wm2 = _mm256_set1_epi32(len_x - 2);
    const __m256i vi_hm1 = _mm256_set1_epi32(len_y - 1);
    const __m256i vi_stride = _mm256_set1_epi32(static_cast<int32_t>(b_sy));
    const __m256i vi_bias = _mm256_set1_epi16(static_cast<int16_t>(0x8000));
    const __m256i vi_unbias = _mm256_set1_epi32(1 << 23);
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    __m256i v_sum = _mm256_setzero_si256();
    __m256i v_cnt = _mm256_setzero_si256();
    __m256i v_sq_even = _mm256_setzero_si256();
    __m256i v_sq_odd = _mm256_setzero_si256();
    alignas(32) int32_t off0[8], off1[8];

    const float* gx_ptr = grid.xs.data();
    const float* gy_ptr = grid.ys.data();
    const uint16_t* ref_ptr = grid.ref.data();
    const int64_t count = grid.count;

    for (int64_t n = 0; n < count; n += 8) {
        const __m256 dx_ = _mm256_sub_ps(_mm256_loadu_ps(gx_ptr + n), v_cx);
        const __m256 dy_ = _mm256_sub_ps(_mm256_loadu_ps(gy_ptr + n), v_cy);
        const __m256 v_dy = _mm256_mul_ps(dy_, v_p1c);
        const __m256 u_dy = _mm256_mul_ps(dy_, v_p0c);
        const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_dy, _mm256_mul_ps(dx_, v_p08)), v_ty);
        const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx_, v_p18), u_dy), v_tx);

        __m256 ok = _mm256_and_ps(
            _mm256_and_ps(_mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_wf, _CMP_LT_OQ)),
            _mm256_and_ps(_mm256_cmp_ps(v, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(v, v_hf, _CMP_LT_OQ)));
        if (n + 8 > count) {
            const __m256i real = _mm256_cmpgt_epi32(
                _mm256_set1_epi32(static_cast<int32_t>(count - n)), lane_ids);
            ok = _mm256_and_ps(ok, _mm256_castsi256_ps(real));
        }
        if (_mm256_movemask_ps(ok) == 0) {
            continue;
        }

        __m256 uc = _mm256_blendv_ps(u, v_uc, _mm256_cmp_ps(u, v_wm1, _CMP_GE_OQ));
        uc = _mm256_blendv_ps(uc, v_zero, _mm256_cmp_ps(uc, v_zero, _CMP_LT_OQ));
        const __m256i uidx = _mm256_cvttps_epi32(uc);
        const __m256i vfx = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(uc, _mm256_cvtepi32_ps(uidx)), v_256));
        // Valid samples already lie in [0, w - 2]; the clamp keeps the paired
        // load of a rejected lane inside its row.
        const __m256i vix = _mm256_min_epi32(_mm256_max_epi32(uidx, vi_zero), vi_wm2);

        __m256 vc = _mm256_blendv_ps(v, v_vc, _mm256_cmp_ps(v, v_hm1, _CMP_GE_OQ));
        vc = _mm256_blendv_ps(vc, v_zero, _mm256_cmp_ps(vc, v_zero, _CMP_LT_OQ));
        const __m256i vidx = _mm256_cvttps_epi32(vc);
        const __m256i vfy = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(vc, _mm256_cvtepi32_ps(vidx)), v_256));
        const __m256i viy = _mm256_min_epi32(_mm256_max_epi32(vidx, vi_zero), vi_hm1);
        const __m256i vjy = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(vidx, vi_one), vi_zero), vi_hm1);

        _mm256_store_si256(reinterpret_cast<__m256i*>(off0),
                           _mm256_add_epi32(_mm256_mullo_epi32(viy, vi_stride), vix));
        _mm256_store_si256(reinterpret_cast<__m256i*>(off1),
                           _mm256_add_epi32(_mm256_mullo_epi32(vjy, vi_stride), vix));
        const __m256i pair0 = _mm256_setr_epi32(
            load_u16_pair(bb + off0[0]), load_u16_pair(bb + off0[1]),
            load_u16_pair(bb + off0[2]), load_u16_pair(bb + off0[3]),
            load_u16_pair(bb + off0[4]), load_u16_pair(bb + off0[5]),
            load_u16_pair(bb + off0[6]), load_u16_pair(bb + off0[7]));
        const __m256i pair1 = _mm256_setr_epi32(
            load_u16_pair(bb + off1[0]), load_u16_pair(bb + off1[1]),
            load_u16_pair(bb + off1[2]), load_u16_pair(bb + off1[3]),
            load_u16_pair(bb + off1[4]), load_u16_pair(bb + off1[5]),
            load_u16_pair(bb + off1[6]), load_u16_pair(bb + off1[7]));

        const __m256i weights = _mm256_or_si256(_mm256_sub_epi32(vi_256, vfx), _mm256_slli_epi32(vfx, 16));
        const __m256i top = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_xor_si256(pair0, vi_bias), weights), vi_unbias);
        const __m256i bot = _mm256_add_epi32(
            _mm256_madd_epi16(_mm256_xor_si256(pair1, vi_bias), weights), vi_unbias);
        const __m256i val = _mm256_srli_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(top, 8),
                             _mm256_mullo_epi32(_mm256_sub_epi32(bot, top), vfy)), 16);
        const __m256i vref = _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(ref_ptr + n)));
        const __m256i oki = _mm256_castps_si256(ok);
        const __m256i diff = _mm256_and_si256(_mm256_sub_epi32(vref, val), oki);

        v_sum = _mm256_add_epi32(v_sum, diff);
        v_cnt = _mm256_sub_epi32(v_cnt, oki);
        const __m256i diff_odd = _mm256_srli_epi64(diff, 32);
        v_sq_even = _mm256_add_epi64(v_sq_even, _mm256_mul_epi32(diff, diff));
        v_sq_odd = _mm256_add_epi64(v_sq_odd, _mm256_mul_epi32(diff_odd, diff_odd));
    }

    alignas(32) int32_t red_sum[8], red_cnt[8];
    alignas(32) int64_t red_sq[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sum), v_sum);
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_cnt), v_cnt);
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sq), _mm256_add_epi64(v_sq_even, v_sq_odd));
    int64_t total = 0, samples = 0;
    for (int m = 0; m < 8; ++m) {
        total += red_sum[m];
        samples += red_cnt[m];
    }
    out_sum = total;
    out_sq = red_sq[0] + red_sq[1] + red_sq[2] + red_sq[3];
    out_cnt = samples;
}

}  // namespace

void align_score_chunk_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    const float* p08, const float* p0c, const float* p18, const float* p1c,
    const float* tx, const float* ty, const float* cx, const float* cy,
    const int64_t* idx, int64_t k_start, int64_t k_stop,
    int64_t* ssum, int64_t* ssq, int64_t* cnt
) {
    const FlatScoreGrid grid = make_flat_score_grid(rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy);
    for (int64_t k = k_start; k < k_stop; ++k) {
        const int64_t ci = idx[k];
        int64_t acc_sum = 0, acc_sq = 0, acc_cnt = 0;
        if (grid.usable) {
            score_block_flat_avx2(
                grid, bb, b_sy, h, w,
                tx[ci], ty[ci], cx[ci], cy[ci],
                p08[ci], p0c[ci], p18[ci], p1c[ci],
                acc_sum, acc_sq, acc_cnt);
        } else {
            score_block_packed_avx2(
                rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy,
                tx[ci], ty[ci], cx[ci], cy[ci],
                p08[ci], p0c[ci], p18[ci], p1c[ci],
                acc_sum, acc_sq, acc_cnt);
        }
        ssum[ci] = acc_sum;
        ssq[ci] = acc_sq;
        cnt[ci] = acc_cnt;
    }
}

void align_score_grid_4d_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx0, float ty0,
    const float* cx_tab, const float* cy_tab,
    const float* p08b, int64_t p08_sy, int64_t p08_sx,
    const float* p18b, int64_t p18_sy, int64_t p18_sx,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    int64_t k_start, int64_t k_stop,
    int64_t* out_best_k, float* out_best_sc, int64_t slot
) {
    (void)n3;
    const float hf = static_cast<float>(h);
    const float wf = static_cast<float>(w);
    const float x_lo = xs[0], x_hi = xs[nxx - 1];
    const float y_lo = ys[0], y_hi = ys[nyy - 1];
    float best_sc = std::numeric_limits<float>::infinity();
    int64_t best_k = -1;
    const FlatScoreGrid grid = make_flat_score_grid(rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy);

    for (int64_t k = k_start; k < k_stop; ++k) {
        int64_t rem = k;
        const int64_t t0 = rem % n0;
        rem /= n0;
        const int64_t t1 = rem % n1;
        rem /= n1;
        const int64_t t2 = rem % n2;
        const int64_t t3 = rem / n2;

        const float cx_k = cx_tab[t0];
        const float cy_k = cy_tab[t1];
        const float p08_k = p08b[t3 * p08_sy + t2 * p08_sx];
        const float p18_k = p18b[t3 * p18_sy + t2 * p18_sx];
        const float p0c_k = p08_k;
        const float p1c_k = p18_k;

        const float c_u = (tx0 - (cx_k * p18_k)) - (cy_k * p0c_k);
        const float c_v = (ty0 - (cy_k * p1c_k)) - (cx_k * p08_k);

        const float aux0 = p18_k * x_lo, aux1 = p18_k * x_hi;
        const float buy0 = p0c_k * y_lo, buy1 = p0c_k * y_hi;
        const float u_lo = ((aux0 < aux1 ? aux0 : aux1) + (buy0 < buy1 ? buy0 : buy1)) + c_u;
        const float u_hi = ((aux0 > aux1 ? aux0 : aux1) + (buy0 > buy1 ? buy0 : buy1)) + c_u;

        const float avx0 = p1c_k * x_lo, avx1 = p1c_k * x_hi;
        const float bvy0 = p08_k * y_lo, bvy1 = p08_k * y_hi;
        const float v_lo = ((avx0 < avx1 ? avx0 : avx1) + (bvy0 < bvy1 ? bvy0 : bvy1)) + c_v;
        const float v_hi = ((avx0 > avx1 ? avx0 : avx1) + (bvy0 > bvy1 ? bvy0 : bvy1)) + c_v;

        if (!(u_hi >= 0.0f && u_lo < wf && v_hi >= 0.0f && v_lo < hf)) {
            continue;
        }

        int64_t acc_sum = 0, acc_sq = 0, acc_cnt = 0;
        if (grid.usable) {
            score_block_flat_avx2(
                grid, bb, b_sy, h, w,
                tx0, ty0, cx_k, cy_k,
                p08_k, p0c_k, p18_k, p1c_k,
                acc_sum, acc_sq, acc_cnt);
        } else {
            score_block_packed_avx2(
                rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy,
                tx0, ty0, cx_k, cy_k,
                p08_k, p0c_k, p18_k, p1c_k,
                acc_sum, acc_sq, acc_cnt);
        }
        if (acc_cnt >= 2) {
            const double denom = static_cast<double>(static_cast<float>(acc_cnt));
            const float msq_f = static_cast<float>(static_cast<double>(acc_sq) / denom);
            const float msu_f = static_cast<float>(static_cast<double>(acc_sum) / denom);
            const float var_f = msq_f - (msu_f * msu_f);
            if (var_f >= 0.0f) {
                const float sc = std::sqrt(var_f);
                if (sc < best_sc) {
                    best_sc = sc;
                    best_k = k;
                }
            }
        }
    }
    out_best_k[slot] = best_k;
    out_best_sc[slot] = best_sc;
}

namespace {

constexpr std::array<uint32_t, 101> make_activity_recip_u32() {
    std::array<uint32_t, 101> arr{};
    for (int i = 0; i < 101; ++i) {
        const uint64_t d = static_cast<uint64_t>(10 + i);
        arr[static_cast<size_t>(i)] = static_cast<uint32_t>(((uint64_t{1} << 32) + d - 1u) / d);
    }
    return arr;
}

constexpr std::array<uint32_t, 101> kActivityRecipU32 = make_activity_recip_u32();

// min(65535, 110 * |v - m| / (10 + 100 * max(v, m) / 65535)), integer division.
// The scalar kernel multiplies by a rounded-up 2^32 reciprocal from a table,
// which equals the floor quotient for these operand ranges (numerator below
// 2^23, divisor 10..110). A float division is at most one above that quotient,
// so a single exact compare restores it without a table gather, which is slow
// on processors that mitigate gather data sampling.
inline __m256i normalized_diff_8_avx2(__m256i v_val, __m256i v_mean) {
    const __m256i vmax = _mm256_max_epu32(v_val, v_mean);
    const __m256i v100 = _mm256_mullo_epi32(vmax, _mm256_set1_epi32(100));
    // v100 / 65535 for v100 <= 6553500.
    const __m256i denom = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(v100, _mm256_set1_epi32(1)), _mm256_srli_epi32(v100, 16)), 16);
    const __m256 vd = _mm256_cvtepi32_ps(_mm256_add_epi32(denom, _mm256_set1_epi32(10)));
    const __m256i vdiff = _mm256_abs_epi32(_mm256_sub_epi32(v_val, v_mean));
    const __m256 vn = _mm256_cvtepi32_ps(_mm256_mullo_epi32(vdiff, _mm256_set1_epi32(110)));
    const __m256 q = _mm256_floor_ps(_mm256_div_ps(vn, vd));
    // q * vd is an integer below 2^24, so the product and the compare are exact.
    const __m256i over = _mm256_castps_si256(_mm256_cmp_ps(_mm256_mul_ps(q, vd), vn, _CMP_GT_OQ));
    const __m256i res = _mm256_add_epi32(_mm256_cvttps_epi32(q), over);
    return _mm256_min_epu32(res, _mm256_set1_epi32(65535));
}

}  // namespace

// Only callers that checked FMA support may enter these two functions.
#if !defined(_MSC_VER)
__attribute__((target("fma")))
#endif
void resample_u16_vert_acc_row_avx2_fma(
    const uint16_t* sline, float* row_buf, int64_t sw, float wval
) {
    const __m256 vw = _mm256_set1_ps(wval);
    int64_t x = 0;
    for (; x + 8 <= sw; x += 8) {
        const __m256 vs = _mm256_cvtepi32_ps(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sline + x))));
        const __m256 acc = _mm256_loadu_ps(row_buf + x);
        _mm256_storeu_ps(row_buf + x, _mm256_fmadd_ps(vw, vs, acc));
    }
    for (; x < sw; ++x) {
        row_buf[x] = std::fmaf(wval, static_cast<float>(sline[x]), row_buf[x]);
    }
}

#if !defined(_MSC_VER)
__attribute__((target("fma")))
#endif
void resample_u16_horiz_rep_row_avx2_fma(
    const float* row_buf, float* sums_buf, uint16_t* orow,
    int64_t first, int64_t nrep,
    const int32_t* h_idx_first, const float* h_w_first,
    int64_t h_taps, int64_t hi_sy, int64_t hw_sy
) {
    (void)hi_sy;
    (void)hw_sy;
    std::memset(sums_buf, 0, static_cast<size_t>(nrep) * sizeof(float));
    for (int64_t tap = 0; tap < h_taps; tap += 2) {
        const __m256 v_even = _mm256_set1_ps(h_w_first[tap]);
        const __m256 v_odd = _mm256_set1_ps(h_w_first[tap + 1]);
        const float even_s = h_w_first[tap];
        const float odd_s = h_w_first[tap + 1];
        const float* line = row_buf + h_idx_first[tap];
        int64_t x = 0;
        for (; x + 8 <= nrep; x += 8) {
            const __m256 a = _mm256_loadu_ps(line + 2 * x);
            const __m256 b = _mm256_loadu_ps(line + 2 * x + 8);
            const __m256 even_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
            const __m256 odd_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
            const __m256 prev = _mm256_loadu_ps(sums_buf + x);
            const __m256 next = _mm256_fmadd_ps(
                v_odd, odd_v, _mm256_fmadd_ps(v_even, even_v, prev));
            _mm256_storeu_ps(sums_buf + x, next);
        }
        for (; x < nrep; ++x) {
            sums_buf[x] = std::fmaf(
                odd_s, line[2 * x + 1], std::fmaf(even_s, line[2 * x], sums_buf[x]));
        }
    }
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(65535.0f);
    uint16_t* dst = orow + first;
    int64_t x = 0;
    for (; x + 8 <= nrep; x += 8) {
        const __m256 r = _mm256_add_ps(_mm256_loadu_ps(sums_buf + x), v_half);
        __m256 c = _mm256_blendv_ps(r, v_zero, _mm256_cmp_ps(r, v_zero, _CMP_LE_OQ));
        c = _mm256_blendv_ps(c, v_max, _mm256_cmp_ps(c, v_max, _CMP_GE_OQ));
        store_u32x8_as_u16(_mm256_cvttps_epi32(c), dst + x);
    }
    for (; x < nrep; ++x) {
        const float rounded = sums_buf[x] + 0.5f;
        if (rounded <= 0.0f) {
            dst[x] = 0;
        } else if (rounded >= 65535.0f) {
            dst[x] = 65535;
        } else {
            dst[x] = static_cast<uint16_t>(static_cast<int32_t>(rounded));
        }
    }
}

#if !defined(_MSC_VER)
__attribute__((target("fma")))
#endif
int64_t resample_u16_horiz_taps_row_avx2_fma(
    const float* row_buf, uint16_t* orow, int64_t x0, int64_t x1,
    const int32_t* h_idx, int64_t hi_sy, const float* h_w, int64_t hw_sy,
    const int32_t* h_cnt, int64_t h_taps
) {
    const __m256i v_taps = _mm256_set1_epi32(static_cast<int32_t>(h_taps));
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(65535.0f);
    int64_t x = x0;
    for (; x + 8 <= x1; x += 8) {
        const __m256i counts = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h_cnt + x));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(counts, v_taps)) != -1) {
            break;
        }
        const int32_t* idx = h_idx + x * hi_sy;
        const float* wgt = h_w + x * hw_sy;
        __m256 acc = v_zero;
        for (int64_t tap = 0; tap < h_taps; ++tap) {
            // Element loads instead of gathers: see normalized_diff_8_avx2.
            const __m256 vw = _mm256_setr_ps(
                wgt[tap], wgt[hw_sy + tap], wgt[2 * hw_sy + tap], wgt[3 * hw_sy + tap],
                wgt[4 * hw_sy + tap], wgt[5 * hw_sy + tap], wgt[6 * hw_sy + tap], wgt[7 * hw_sy + tap]);
            const __m256 vv = _mm256_setr_ps(
                row_buf[idx[tap]], row_buf[idx[hi_sy + tap]],
                row_buf[idx[2 * hi_sy + tap]], row_buf[idx[3 * hi_sy + tap]],
                row_buf[idx[4 * hi_sy + tap]], row_buf[idx[5 * hi_sy + tap]],
                row_buf[idx[6 * hi_sy + tap]], row_buf[idx[7 * hi_sy + tap]]);
            acc = _mm256_fmadd_ps(vw, vv, acc);
        }
        const __m256 r = _mm256_add_ps(acc, v_half);
        __m256 c = _mm256_blendv_ps(r, v_zero, _mm256_cmp_ps(r, v_zero, _CMP_LE_OQ));
        c = _mm256_blendv_ps(c, v_max, _mm256_cmp_ps(c, v_max, _CMP_GE_OQ));
        store_u32x8_as_u16(_mm256_cvttps_epi32(c), orow + x);
    }
    return x;
}

void activity_row_u16_avx2(
    const uint16_t* gr, const uint16_t* fr, uint16_t* orow, int64_t w
) {
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256i vg = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + x)));
        const __m256i vf = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fr + x)));
        store_u32x8_as_u16(normalized_diff_8_avx2(vg, vf), orow + x);
    }
    for (; x < w; ++x) {
        const int64_t value = gr[x];
        const int64_t mean = fr[x];
        const int64_t maximum = value > mean ? value : mean;
        const int64_t denom = (maximum * 100LL * int64_t{0x80008001LL}) >> 47;
        const int64_t num = 110LL * std::abs(value - mean);
        const int64_t res = (num * static_cast<int64_t>(kActivityRecipU32[static_cast<size_t>(denom)])) >> 32;
        orow[x] = static_cast<uint16_t>(res < 65535LL ? res : 65535LL);
    }
}

void detail_row_u16_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* orow, int64_t full_w
) {
    const __m256i m16 = _mm256_set1_epi32(0xFFFF);
    int64_t x = 0;
    for (; x + 8 <= full_w; x += 8) {
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x));
        const __m256i va = _mm256_and_si256(p0, m16);
        const __m256i vb = _mm256_srli_epi32(p0, 16);
        const __m256i vc = _mm256_and_si256(p1, m16);
        const __m256i vd = _mm256_srli_epi32(p1, 16);
        const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(va, vb), _mm256_add_epi32(vc, vd));
        const __m256i vmean = _mm256_srli_epi32(sum, 2);
        const __m256i ea = normalized_diff_8_avx2(va, vmean);
        const __m256i eb = normalized_diff_8_avx2(vb, vmean);
        const __m256i ec = normalized_diff_8_avx2(vc, vmean);
        const __m256i ed = normalized_diff_8_avx2(vd, vmean);
        const __m256i etot = _mm256_add_epi32(_mm256_add_epi32(ea, eb), _mm256_add_epi32(ec, ed));
        store_u32x8_as_u16(_mm256_srli_epi32(etot, 2), orow + x);
    }
    for (; x < full_w; ++x) {
        const int64_t x0 = 2 * x;
        const int64_t a = r0[x0], b = r0[x0 + 1], c = r1[x0], d = r1[x0 + 1];
        const int64_t mean = (a + b + c + d) >> 2;
        auto ndiff = [mean](int64_t v) -> int64_t {
            const int64_t maximum = v > mean ? v : mean;
            const int64_t denom = (maximum * 100LL * int64_t{0x80008001LL}) >> 47;
            const int64_t num = 110LL * std::abs(v - mean);
            const int64_t res = (num * static_cast<int64_t>(kActivityRecipU32[static_cast<size_t>(denom)])) >> 32;
            return res < 65535LL ? res : 65535LL;
        };
        orow[x] = static_cast<uint16_t>((ndiff(a) + ndiff(b) + ndiff(c) + ndiff(d)) >> 2);
    }
}

void select_activity_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* dr, uint16_t* vr,
    int64_t w, uint16_t code, uint32_t reciprocal_u32
) {
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i v_code = _mm256_set1_epi16(static_cast<int16_t>(code));
    const __m256i v_recip = _mm256_set1_epi32(static_cast<int32_t>(reciprocal_u32));
    int64_t x = 0;
    for (; x + 16 <= w; x += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ar + x));
        if (vr != nullptr) {
            const __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(va));
            const __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(va, 1));
            auto mul_r = [&](__m256i v) -> __m256i {
                const __m256i qe = _mm256_srli_epi64(_mm256_mul_epu32(v, v_recip), 32);
                const __m256i qo = _mm256_mul_epu32(_mm256_srli_epi64(v, 32), v_recip);
                return _mm256_blend_epi32(qe, qo, 0xAA);
            };
            const __m256i add16 = _mm256_permute4x64_epi64(
                _mm256_packus_epi32(mul_r(lo32), mul_r(hi32)), 0xD8);
            const __m256i prev_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vr + x));
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(vr + x), _mm256_add_epi16(prev_v, add16));
        }
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i le_b = _mm256_cmpeq_epi16(_mm256_max_epu16(va, vb), vb);
        if (!_mm256_testc_si256(le_b, all_ones)) {
            const __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dr + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(br + x), _mm256_blendv_epi8(va, vb, le_b));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dr + x), _mm256_blendv_epi8(v_code, vd, le_b));
        }
    }
    for (; x < w; ++x) {
        const uint16_t val = ar[x];
        if (vr != nullptr) {
            vr[x] = static_cast<uint16_t>(
                vr[x] + ((static_cast<uint64_t>(val) * reciprocal_u32) >> 32u));
        }
        if (val > br[x]) {
            br[x] = val;
            dr[x] = code;
        }
    }
}

void select_detail_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* sr, uint16_t* wr,
    int64_t w, uint16_t code
) {
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i v_code = _mm256_set1_epi16(static_cast<int16_t>(code));
    int64_t x = 0;
    for (; x + 16 <= w; x += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ar + x));
        const __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x));
        const __m256i max_s = _mm256_max_epu16(va, vs);
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i le_b = _mm256_cmpeq_epi16(_mm256_max_epu16(va, vb), vb);
        if (_mm256_testc_si256(le_b, all_ones)) {
            const __m256i le_s = _mm256_cmpeq_epi16(max_s, vs);
            if (!_mm256_testc_si256(le_s, all_ones)) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(sr + x), max_s);
            }
            continue;
        }
        const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
        const __m256i new_sr = _mm256_blendv_epi8(vb, max_s, le_b);
        const __m256i new_br = _mm256_blendv_epi8(va, vb, le_b);
        const __m256i new_wr = _mm256_blendv_epi8(v_code, vw, le_b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sr + x), new_sr);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(br + x), new_br);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wr + x), new_wr);
    }
    for (; x < w; ++x) {
        const uint16_t val = ar[x];
        if (val > br[x]) {
            sr[x] = br[x];
            br[x] = val;
            wr[x] = code;
        } else if (val > sr[x]) {
            sr[x] = val;
        }
    }
}

void output_u16_to_u8_row_avx2(
    const uint16_t* cr, uint8_t* orow, int64_t elems, bool e37
) {
    const __m256i v_one = _mm256_set1_epi16(1);
    int64_t x = 0;
    for (; x + 16 <= elems; x += 16) {
        __m256i v = _mm256_srli_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cr + x)), 8);
        if (e37) {
            v = _mm256_subs_epu16(v, v_one);
        }
        const __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0xD8);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + x), _mm256_castsi256_si128(p));
    }
    for (; x < elems; ++x) {
        uint16_t v = cr[x] >> 8u;
        if (e37 && v > 0) {
            v -= 1;
        }
        orow[x] = static_cast<uint8_t>(v);
    }
}

void output_u16_to_f32_row_avx2(
    const uint16_t* cr, float* orow, int64_t elems, float factor
) {
    const __m256 v_fac = _mm256_set1_ps(factor);
    int64_t x = 0;
    for (; x + 8 <= elems; x += 8) {
        const __m256 vf = _mm256_cvtepi32_ps(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cr + x))));
        _mm256_storeu_ps(orow + x, _mm256_mul_ps(vf, v_fac));
    }
    for (; x < elems; ++x) {
        orow[x] = static_cast<float>(cr[x]) * factor;
    }
}

void tile_accumulate_3c_row_avx2(
    const void* tr, int src_kind, const float* wr,
    float* ar, float* war, int64_t fw
) {
    const __m256i perm0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i perm1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i perm2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    int64_t dx = 0;
    for (; dx + 8 <= fw; dx += 8) {
        const __m256 w = _mm256_loadu_ps(wr + dx);
        _mm256_storeu_ps(war + dx, _mm256_add_ps(_mm256_loadu_ps(war + dx), w));
        const __m256 w_blk0 = _mm256_permutevar8x32_ps(w, perm0);
        const __m256 w_blk1 = _mm256_permutevar8x32_ps(w, perm1);
        const __m256 w_blk2 = _mm256_permutevar8x32_ps(w, perm2);

        __m256 t0, t1, t2;
        if (src_kind == 0) {
            const float* src = static_cast<const float*>(tr) + dx * 3;
            t0 = _mm256_loadu_ps(src);
            t1 = _mm256_loadu_ps(src + 8);
            t2 = _mm256_loadu_ps(src + 16);
        } else if (src_kind == 1) {
            const uint8_t* src = static_cast<const uint8_t*>(tr) + dx * 3;
            const __m128i b01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            const __m128i b2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 16));
            t0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b01));
            t1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b01, 8)));
            t2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b2));
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(tr) + dx * 3;
            t0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src))));
            t1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8))));
            t2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16))));
        }

        float* dst = ar + dx * 3;
        _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(dst), _mm256_mul_ps(t0, w_blk0)));
        _mm256_storeu_ps(dst + 8, _mm256_add_ps(_mm256_loadu_ps(dst + 8), _mm256_mul_ps(t1, w_blk1)));
        _mm256_storeu_ps(dst + 16, _mm256_add_ps(_mm256_loadu_ps(dst + 16), _mm256_mul_ps(t2, w_blk2)));
    }
    for (; dx < fw; ++dx) {
        const float w = wr[dx];
        war[dx] += w;
        const int64_t o = dx * 3;
        if (src_kind == 0) {
            const float* src = static_cast<const float*>(tr) + o;
            ar[o] += src[0] * w;
            ar[o + 1] += src[1] * w;
            ar[o + 2] += src[2] * w;
        } else if (src_kind == 1) {
            const uint8_t* src = static_cast<const uint8_t*>(tr) + o;
            ar[o] += static_cast<float>(src[0]) * w;
            ar[o + 1] += static_cast<float>(src[1]) * w;
            ar[o + 2] += static_cast<float>(src[2]) * w;
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(tr) + o;
            ar[o] += static_cast<float>(src[0]) * w;
            ar[o + 1] += static_cast<float>(src[1]) * w;
            ar[o + 2] += static_cast<float>(src[2]) * w;
        }
    }
}

namespace {

inline double hsum_pd4(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(s, s));
}

template <int P>
inline void ecc_prepare_level_row_impl(
    const float* rm1, const float* r0, const float* rp1,
    int x0, int rw, double cx, double inv_s, double mean_t, double g_scale, double v,
    bool sample_h_row, bool stride2,
    float* dst_t0, float* dst_gx, float* dst_gy,
    double& norm_T2, double* js, double* tv, double* H_tri
) {
    const __m256d v_mean = _mm256_set1_pd(mean_t);
    const __m256d v_gs = _mm256_set1_pd(g_scale);
    const __m256d v_two = _mm256_set1_pd(2.0);
    const __m256d v_vv = _mm256_set1_pd(v);
    const __m256d v_cx = _mm256_set1_pd(cx);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_step = _mm256_set1_pd(4.0);
    __m256d vx = _mm256_setr_pd(
        static_cast<double>(x0), static_cast<double>(x0 + 1),
        static_cast<double>(x0 + 2), static_cast<double>(x0 + 3));

    __m256d v_norm = _mm256_setzero_pd();
    __m256d v_js[P];
    __m256d v_tv[P];
    for (int r = 0; r < P; ++r) {
        v_js[r] = _mm256_setzero_pd();
        v_tv[r] = _mm256_setzero_pd();
    }
    constexpr int kTri = (P * (P + 1)) / 2;
    __m256d v_h256[kTri];
    __m128d v_h128[kTri];
    if (sample_h_row) {
        for (int k = 0; k < kTri; ++k) {
            v_h256[k] = _mm256_setzero_pd();
            v_h128[k] = _mm_setzero_pd();
        }
    }

    int rx = 0;
    for (; rx + 4 <= rw; rx += 4) {
        const int x = x0 + rx;
        const __m256d u = _mm256_mul_pd(_mm256_sub_pd(vx, v_cx), v_invs);
        vx = _mm256_add_pd(vx, v_step);

        const __m256d tl = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x - 1));
        const __m256d tc = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x));
        const __m256d tr = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x + 1));

        const __m256d ml = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x - 1));
        const __m256d c0 = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x));
        const __m256d mr = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x + 1));

        const __m256d bl = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x - 1));
        const __m256d bc = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x));
        const __m256d br = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x + 1));

        const __m256d t0 = _mm256_sub_pd(c0, v_mean);
        const __m256d gx_pos = _mm256_add_pd(_mm256_add_pd(tr, _mm256_mul_pd(v_two, mr)), br);
        const __m256d gx_neg = _mm256_add_pd(_mm256_add_pd(tl, _mm256_mul_pd(v_two, ml)), bl);
        const __m256d gx = _mm256_mul_pd(_mm256_sub_pd(gx_pos, gx_neg), v_gs);

        const __m256d gy_pos = _mm256_add_pd(_mm256_add_pd(bl, _mm256_mul_pd(v_two, bc)), br);
        const __m256d gy_neg = _mm256_add_pd(_mm256_add_pd(tl, _mm256_mul_pd(v_two, tc)), tr);
        const __m256d gy = _mm256_mul_pd(_mm256_sub_pd(gy_pos, gy_neg), v_gs);

        _mm_storeu_ps(dst_t0 + rx, _mm256_cvtpd_ps(t0));
        _mm_storeu_ps(dst_gx + rx, _mm256_cvtpd_ps(gx));
        _mm_storeu_ps(dst_gy + rx, _mm256_cvtpd_ps(gy));

        v_norm = _mm256_add_pd(v_norm, _mm256_mul_pd(t0, t0));

        __m256d J[P];
        if constexpr (P == 6) {
            J[0] = _mm256_mul_pd(gx, u);
            J[1] = _mm256_mul_pd(gy, u);
            J[2] = _mm256_mul_pd(gx, v_vv);
            J[3] = _mm256_mul_pd(gy, v_vv);
            J[4] = gx;
            J[5] = gy;
        } else {
            const __m256d guv = _mm256_add_pd(_mm256_mul_pd(gx, u), _mm256_mul_pd(gy, v_vv));
            const __m256d neg_guv = _mm256_sub_pd(_mm256_setzero_pd(), guv);
            J[0] = _mm256_mul_pd(gx, u);
            J[1] = _mm256_mul_pd(gy, u);
            J[2] = _mm256_mul_pd(u, neg_guv);
            J[3] = _mm256_mul_pd(gx, v_vv);
            J[4] = _mm256_mul_pd(gy, v_vv);
            J[5] = _mm256_mul_pd(v_vv, neg_guv);
            J[6] = gx;
            J[7] = gy;
        }

        for (int r = 0; r < P; ++r) {
            v_js[r] = _mm256_add_pd(v_js[r], J[r]);
            v_tv[r] = _mm256_add_pd(v_tv[r], _mm256_mul_pd(J[r], t0));
        }

        if (sample_h_row) {
            if (!stride2) {
                int idx = 0;
                for (int r = 0; r < P; ++r) {
                    const __m256d jr = J[r];
                    for (int c = r; c < P; ++c) {
                        v_h256[idx] = _mm256_add_pd(v_h256[idx], _mm256_mul_pd(jr, J[c]));
                        ++idx;
                    }
                }
            } else {
                // Sample even rx lanes (0 and 2) when stride2 is active.
                __m128d j_even[P];
                for (int r = 0; r < P; ++r) {
                    j_even[r] = _mm_shuffle_pd(
                        _mm256_castpd256_pd128(J[r]), _mm256_extractf128_pd(J[r], 1), 0);
                }
                int idx = 0;
                for (int r = 0; r < P; ++r) {
                    const __m128d jr = j_even[r];
                    for (int c = r; c < P; ++c) {
                        v_h128[idx] = _mm_add_pd(v_h128[idx], _mm_mul_pd(jr, j_even[c]));
                        ++idx;
                    }
                }
            }
        }
    }

    double s_norm = hsum_pd4(v_norm);
    double s_js[P];
    double s_tv[P];
    for (int r = 0; r < P; ++r) {
        s_js[r] = hsum_pd4(v_js[r]);
        s_tv[r] = hsum_pd4(v_tv[r]);
    }
    double s_h[kTri]{};
    if (sample_h_row) {
        if (!stride2) {
            for (int k = 0; k < kTri; ++k) {
                s_h[k] = hsum_pd4(v_h256[k]);
            }
        } else {
            for (int k = 0; k < kTri; ++k) {
                s_h[k] = _mm_cvtsd_f64(_mm_hadd_pd(v_h128[k], v_h128[k]));
            }
        }
    }

    for (; rx < rw; ++rx) {
        const int x = x0 + rx;
        const double u = (static_cast<double>(x) - cx) * inv_s;
        const double t0 = static_cast<double>(r0[x]) - mean_t;
        const double tl = static_cast<double>(rm1[x - 1]);
        const double tc = static_cast<double>(rm1[x]);
        const double tr = static_cast<double>(rm1[x + 1]);
        const double ml = static_cast<double>(r0[x - 1]);
        const double mr = static_cast<double>(r0[x + 1]);
        const double bl = static_cast<double>(rp1[x - 1]);
        const double bc = static_cast<double>(rp1[x]);
        const double br = static_cast<double>(rp1[x + 1]);
        const double gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * g_scale;
        const double gy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * g_scale;
        dst_t0[rx] = static_cast<float>(t0);
        dst_gx[rx] = static_cast<float>(gx);
        dst_gy[rx] = static_cast<float>(gy);
        s_norm += t0 * t0;

        double J[P];
        if constexpr (P == 6) {
            J[0] = gx * u; J[1] = gy * u; J[2] = gx * v;
            J[3] = gy * v; J[4] = gx;     J[5] = gy;
        } else {
            const double guv = gx * u + gy * v;
            J[0] = gx * u; J[1] = gy * u; J[2] = -u * guv;
            J[3] = gx * v; J[4] = gy * v; J[5] = -v * guv;
            J[6] = gx;     J[7] = gy;
        }
        for (int r = 0; r < P; ++r) {
            s_js[r] += J[r];
            s_tv[r] += J[r] * t0;
        }
        if (sample_h_row && ((rx & 1) == 0 || !stride2)) {
            int idx = 0;
            for (int r = 0; r < P; ++r) {
                const double jr = J[r];
                for (int c = r; c < P; ++c) {
                    s_h[idx++] += jr * J[c];
                }
            }
        }
    }

    norm_T2 += s_norm;
    for (int r = 0; r < P; ++r) {
        js[r] += s_js[r];
        tv[r] += s_tv[r];
    }
    if (sample_h_row) {
        for (int k = 0; k < kTri; ++k) {
            H_tri[k] += s_h[k];
        }
    }
}

inline __m256d ecc_bilinear_4_avx2(
    const float* src_data, int w, int h, __m256d sx, __m256d sy
) {
    const __m128i vi_zero = _mm_setzero_si128();
    const __m128i vi_wm2 = _mm_set1_epi32(std::max(0, w - 2));
    const __m128i vi_hm2 = _mm_set1_epi32(std::max(0, h - 2));
    const __m128i ix_v = _mm_min_epi32(_mm_max_epi32(_mm256_cvttpd_epi32(sx), vi_zero), vi_wm2);
    const __m128i iy_v = _mm_min_epi32(_mm_max_epi32(_mm256_cvttpd_epi32(sy), vi_zero), vi_hm2);
    alignas(16) int32_t ix[4], iy[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(ix), ix_v);
    _mm_store_si128(reinterpret_cast<__m128i*>(iy), iy_v);

    const __m256d fx = _mm256_sub_pd(sx, _mm256_cvtepi32_pd(ix_v));
    const __m256d fy = _mm256_sub_pd(sy, _mm256_cvtepi32_pd(iy_v));

    // Each tap pair (left, right) is one 64-bit load; the pairs are shuffled
    // into lanes in registers, avoiding a store that a wider load must wait on.
    const float* p[4];
    for (int m = 0; m < 4; ++m) {
        p[m] = src_data + static_cast<int64_t>(iy[m]) * w + ix[m];
    }
    const auto load_pairs = [](const float* a, const float* b) {
        return _mm_castpd_ps(_mm_loadh_pd(
            _mm_load_sd(reinterpret_cast<const double*>(a)), reinterpret_cast<const double*>(b)));
    };
    const __m128 top01 = load_pairs(p[0], p[1]);
    const __m128 top23 = load_pairs(p[2], p[3]);
    const __m128 bot01 = load_pairs(p[0] + w, p[1] + w);
    const __m128 bot23 = load_pairs(p[2] + w, p[3] + w);
    const __m256d v00 = _mm256_cvtps_pd(_mm_shuffle_ps(top01, top23, 0x88));
    const __m256d v01 = _mm256_cvtps_pd(_mm_shuffle_ps(top01, top23, 0xDD));
    const __m256d v10 = _mm256_cvtps_pd(_mm_shuffle_ps(bot01, bot23, 0x88));
    const __m256d v11 = _mm256_cvtps_pd(_mm_shuffle_ps(bot01, bot23, 0xDD));

    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d top = _mm256_add_pd(v00, _mm256_mul_pd(fx, _mm256_sub_pd(v01, v00)));
    const __m256d bot = _mm256_add_pd(v10, _mm256_mul_pd(fx, _mm256_sub_pd(v11, v10)));
    return _mm256_add_pd(
        _mm256_mul_pd(top, _mm256_sub_pd(v_one, fy)),
        _mm256_mul_pd(bot, fy));
}

}  // namespace

void ecc_prepare_level_row_avx2(
    const float* rm1, const float* r0, const float* rp1,
    int x0, int rw, double cx, double inv_s, double mean_t, double g_scale, double v,
    int p_count, bool sample_h_row, bool stride2,
    float* dst_t0, float* dst_gx, float* dst_gy,
    double& norm_T2, double* js, double* tv, double* H_tri
) {
    if (p_count == 6) {
        ecc_prepare_level_row_impl<6>(
            rm1, r0, rp1, x0, rw, cx, inv_s, mean_t, g_scale, v,
            sample_h_row, stride2, dst_t0, dst_gx, dst_gy, norm_T2, js, tv, H_tri);
    } else {
        ecc_prepare_level_row_impl<8>(
            rm1, r0, rp1, x0, rw, cx, inv_s, mean_t, g_scale, v,
            sample_h_row, stride2, dst_t0, dst_gx, dst_gy, norm_T2, js, tv, H_tri);
    }
}

void ecc_ic_pass_all_in_row_p6_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double sx0, double sy0, double dsx, double dsy,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g
) {
    __m256d v_iw = _mm256_setzero_pd();
    __m256d v_iw2 = _mm256_setzero_pd();
    __m256d v_tiw = _mm256_setzero_pd();
    __m256d v_rg[6];
    for (int k = 0; k < 6; ++k) {
        v_rg[k] = _mm256_setzero_pd();
    }
    const __m256d v_offsets = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    const __m256d v_dsx = _mm256_set1_pd(dsx);
    const __m256d v_dsy = _mm256_set1_pd(dsy);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_vv = _mm256_set1_pd(v);
    __m256d v_sx = _mm256_add_pd(_mm256_set1_pd(sx0), _mm256_mul_pd(v_offsets, v_dsx));
    __m256d v_sy = _mm256_add_pd(_mm256_set1_pd(sy0), _mm256_mul_pd(v_offsets, v_dsy));
    __m256d v_u = _mm256_add_pd(_mm256_set1_pd(u0), _mm256_mul_pd(v_offsets, v_invs));
    const __m256d v_dsx4 = _mm256_set1_pd(dsx * 4.0);
    const __m256d v_dsy4 = _mm256_set1_pd(dsy * 4.0);
    const __m256d v_invs4 = _mm256_set1_pd(inv_s * 4.0);

    // Two passes per segment, as in the eight-parameter pass below.
    constexpr int kSegment = 512;
    alignas(32) double iw_seg[kSegment];
    const int vec_end = rw & ~3;
    int rx = 0;
    while (rx < vec_end) {
        const int seg_end = std::min(vec_end, rx + kSegment);
        for (int k = rx; k < seg_end; k += 4) {
            const __m256d iw = ecc_bilinear_4_avx2(src_data, w, h, v_sx, v_sy);
            const __m256d t0 = _mm256_cvtps_pd(_mm_loadu_ps(t0_row + k));
            v_iw = _mm256_add_pd(v_iw, iw);
            v_iw2 = _mm256_add_pd(v_iw2, _mm256_mul_pd(iw, iw));
            v_tiw = _mm256_add_pd(v_tiw, _mm256_mul_pd(t0, iw));
            _mm256_store_pd(iw_seg + (k - rx), iw);
            v_sx = _mm256_add_pd(v_sx, v_dsx4);
            v_sy = _mm256_add_pd(v_sy, v_dsy4);
        }
        for (int k = rx; k < seg_end; k += 4) {
            const __m256d iw = _mm256_load_pd(iw_seg + (k - rx));
            const __m256d gxw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gx_row + k)), iw);
            const __m256d gyw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gy_row + k)), iw);
            v_rg[0] = _mm256_add_pd(v_rg[0], _mm256_mul_pd(gxw, v_u));
            v_rg[1] = _mm256_add_pd(v_rg[1], _mm256_mul_pd(gyw, v_u));
            v_rg[2] = _mm256_add_pd(v_rg[2], _mm256_mul_pd(gxw, v_vv));
            v_rg[3] = _mm256_add_pd(v_rg[3], _mm256_mul_pd(gyw, v_vv));
            v_rg[4] = _mm256_add_pd(v_rg[4], gxw);
            v_rg[5] = _mm256_add_pd(v_rg[5], gyw);
            v_u = _mm256_add_pd(v_u, v_invs4);
        }
        rx = seg_end;
    }

    double s_iw = hsum_pd4(v_iw);
    double s_iw2 = hsum_pd4(v_iw2);
    double s_tiw = hsum_pd4(v_tiw);
    double s_rg[6];
    for (int k = 0; k < 6; ++k) {
        s_rg[k] = hsum_pd4(v_rg[k]);
    }

    double sx = sx0 + static_cast<double>(rx) * dsx;
    double sy = sy0 + static_cast<double>(rx) * dsy;
    double u = u0 + static_cast<double>(rx) * inv_s;
    for (; rx < rw; ++rx) {
        int ix = static_cast<int>(sx);
        int iy = static_cast<int>(sy);
        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
        const double fx = sx - static_cast<double>(ix);
        const double fy = sy - static_cast<double>(iy);
        const float* p0 = src_data + static_cast<int64_t>(iy) * w + ix;
        const float* p1 = p0 + w;
        const double v00 = p0[0], v01 = p0[1], v10 = p1[0], v11 = p1[1];
        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                        + (v10 + fx * (v11 - v10)) * fy;
        const double t0 = t0_row[rx];
        const double gxw = static_cast<double>(gx_row[rx]) * iw;
        const double gyw = static_cast<double>(gy_row[rx]) * iw;
        s_iw += iw;
        s_iw2 += iw * iw;
        s_tiw += t0 * iw;
        s_rg[0] += gxw * u;
        s_rg[1] += gyw * u;
        s_rg[2] += gxw * v;
        s_rg[3] += gyw * v;
        s_rg[4] += gxw;
        s_rg[5] += gyw;
        sx += dsx;
        sy += dsy;
        u += inv_s;
    }

    rs_iw += s_iw;
    rs_iw2 += s_iw2;
    rs_tiw += s_tiw;
    for (int k = 0; k < 6; ++k) {
        raw_g[k] += s_rg[k];
    }
}

void ecc_ic_pass_all_in_row_p8_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double nsx0, double nsy0, double den0,
    double d_nsx, double d_nsy, double d_den,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g
) {
    __m256d v_iw = _mm256_setzero_pd();
    __m256d v_iw2 = _mm256_setzero_pd();
    __m256d v_tiw = _mm256_setzero_pd();
    __m256d v_rg[8];
    for (int k = 0; k < 8; ++k) {
        v_rg[k] = _mm256_setzero_pd();
    }
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_offsets = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    const __m256d v_dnsx = _mm256_set1_pd(d_nsx);
    const __m256d v_dnsy = _mm256_set1_pd(d_nsy);
    const __m256d v_dden = _mm256_set1_pd(d_den);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_vv = _mm256_set1_pd(v);
    __m256d v_nsx = _mm256_add_pd(_mm256_set1_pd(nsx0), _mm256_mul_pd(v_offsets, v_dnsx));
    __m256d v_nsy = _mm256_add_pd(_mm256_set1_pd(nsy0), _mm256_mul_pd(v_offsets, v_dnsy));
    __m256d v_den = _mm256_add_pd(_mm256_set1_pd(den0), _mm256_mul_pd(v_offsets, v_dden));
    __m256d v_u = _mm256_add_pd(_mm256_set1_pd(u0), _mm256_mul_pd(v_offsets, v_invs));
    const __m256d v_dnsx4 = _mm256_set1_pd(d_nsx * 4.0);
    const __m256d v_dnsy4 = _mm256_set1_pd(d_nsy * 4.0);
    const __m256d v_dden4 = _mm256_set1_pd(d_den * 4.0);
    const __m256d v_invs4 = _mm256_set1_pd(inv_s * 4.0);

    // Eleven running sums do not fit in the vector registers next to the
    // coordinates, so the row runs in segments of two passes: the first warps
    // and samples, keeping the three image sums and the samples, the second
    // adds the eight Jacobian sums. Every lane adds its pixels in the same
    // order as one combined loop.
    constexpr int kSegment = 512;
    alignas(32) double iw_seg[kSegment];
    const int vec_end = rw & ~3;
    int rx = 0;
    while (rx < vec_end) {
        const int seg_end = std::min(vec_end, rx + kSegment);
        for (int k = rx; k < seg_end; k += 4) {
            const __m256d inv_d = _mm256_div_pd(v_one, v_den);
            const __m256d sx = _mm256_mul_pd(v_nsx, inv_d);
            const __m256d sy = _mm256_mul_pd(v_nsy, inv_d);
            const __m256d iw = ecc_bilinear_4_avx2(src_data, w, h, sx, sy);
            const __m256d t0 = _mm256_cvtps_pd(_mm_loadu_ps(t0_row + k));
            v_iw = _mm256_add_pd(v_iw, iw);
            v_iw2 = _mm256_add_pd(v_iw2, _mm256_mul_pd(iw, iw));
            v_tiw = _mm256_add_pd(v_tiw, _mm256_mul_pd(t0, iw));
            _mm256_store_pd(iw_seg + (k - rx), iw);
            v_nsx = _mm256_add_pd(v_nsx, v_dnsx4);
            v_nsy = _mm256_add_pd(v_nsy, v_dnsy4);
            v_den = _mm256_add_pd(v_den, v_dden4);
        }
        for (int k = rx; k < seg_end; k += 4) {
            const __m256d iw = _mm256_load_pd(iw_seg + (k - rx));
            const __m256d gxw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gx_row + k)), iw);
            const __m256d gyw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gy_row + k)), iw);
            const __m256d guvw = _mm256_add_pd(_mm256_mul_pd(gxw, v_u), _mm256_mul_pd(gyw, v_vv));
            v_rg[0] = _mm256_add_pd(v_rg[0], _mm256_mul_pd(gxw, v_u));
            v_rg[1] = _mm256_add_pd(v_rg[1], _mm256_mul_pd(gyw, v_u));
            v_rg[2] = _mm256_sub_pd(v_rg[2], _mm256_mul_pd(v_u, guvw));
            v_rg[3] = _mm256_add_pd(v_rg[3], _mm256_mul_pd(gxw, v_vv));
            v_rg[4] = _mm256_add_pd(v_rg[4], _mm256_mul_pd(gyw, v_vv));
            v_rg[5] = _mm256_sub_pd(v_rg[5], _mm256_mul_pd(v_vv, guvw));
            v_rg[6] = _mm256_add_pd(v_rg[6], gxw);
            v_rg[7] = _mm256_add_pd(v_rg[7], gyw);
            v_u = _mm256_add_pd(v_u, v_invs4);
        }
        rx = seg_end;
    }

    double s_iw = hsum_pd4(v_iw);
    double s_iw2 = hsum_pd4(v_iw2);
    double s_tiw = hsum_pd4(v_tiw);
    double s_rg[8];
    for (int k = 0; k < 8; ++k) {
        s_rg[k] = hsum_pd4(v_rg[k]);
    }

    double nsx = nsx0 + static_cast<double>(rx) * d_nsx;
    double nsy = nsy0 + static_cast<double>(rx) * d_nsy;
    double den = den0 + static_cast<double>(rx) * d_den;
    double u = u0 + static_cast<double>(rx) * inv_s;
    for (; rx < rw; ++rx) {
        const double inv_d = 1.0 / den;
        const double sx = nsx * inv_d;
        const double sy = nsy * inv_d;
        int ix = static_cast<int>(sx);
        int iy = static_cast<int>(sy);
        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
        const double fx = sx - static_cast<double>(ix);
        const double fy = sy - static_cast<double>(iy);
        const float* p0 = src_data + static_cast<int64_t>(iy) * w + ix;
        const float* p1 = p0 + w;
        const double v00 = p0[0], v01 = p0[1], v10 = p1[0], v11 = p1[1];
        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                        + (v10 + fx * (v11 - v10)) * fy;
        const double t0 = t0_row[rx];
        const double gxw = static_cast<double>(gx_row[rx]) * iw;
        const double gyw = static_cast<double>(gy_row[rx]) * iw;
        const double guvw = gxw * u + gyw * v;
        s_iw += iw;
        s_iw2 += iw * iw;
        s_tiw += t0 * iw;
        s_rg[0] += gxw * u;
        s_rg[1] += gyw * u;
        s_rg[2] -= u * guvw;
        s_rg[3] += gxw * v;
        s_rg[4] += gyw * v;
        s_rg[5] -= v * guvw;
        s_rg[6] += gxw;
        s_rg[7] += gyw;
        nsx += d_nsx;
        nsy += d_nsy;
        den += d_den;
        u += inv_s;
    }

    rs_iw += s_iw;
    rs_iw2 += s_iw2;
    rs_tiw += s_tiw;
    for (int k = 0; k < 8; ++k) {
        raw_g[k] += s_rg[k];
    }
}

#else

void pyr_down_vertical_row_f32_avx2(const float*, const float*, const float*, const float*, float*, int64_t, float, float) {}
void pyr_reconstruct_row_f32_avx2(const float*, const float*, const float*, const float*, const float*, float*, int64_t, float, float, float, float, float, float) {}
void pyr_detail_weight_3c_f32_avx2(const float*, const float*, const float*, const float*, const float*, float*, uint16_t*, int64_t, float, float, float, float, float, float, float, bool) {}
void pyr_level0_weights_3c_f32_from_raw_avx2(const float*, const float*, const float*, const float*, const float*, const void*, bool, uint16_t*, int64_t, float, float, float, float, float, float, float, float, bool) {}
void depth_axis_iir_y_unguided_avx2(const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, uint16_t*, uint32_t*) {}
void depth_axis_iir_y_guided_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, uint32_t*, uint32_t*) {}
void depth_axis_iir_x_unguided_avx2(const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, uint16_t*) {}
void depth_axis_iir_x_guided_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t, uint16_t*) {}
void select_a_combine_m2_to_m5_avx2(const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, int64_t) {}
void select_a_response_pairs_avx2(const void*, bool, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, int32_t*, int64_t) {}
uint64_t sum_u8_row_avx2(const uint8_t*, int64_t) { return 0; }
uint64_t sum_u16_row_avx2(const uint16_t*, int64_t) { return 0; }
bool sum_u16_normalized_row_avx2(const uint16_t*, int64_t, double*) { return false; }
void quantize_f32_to_u16_row_avx2(const float*, uint16_t*, int64_t, float) {}
void quantize_f32_to_u8_row_avx2(const float*, uint8_t*, int64_t) {}
void exposure_gain_u16_row_avx2(const uint16_t*, uint16_t*, int64_t, float, float) {}
void exposure_gain_u8_row_avx2(const uint8_t*, uint8_t*, int64_t, float, float) {}
void exposure_gain_f32_row_avx2(const float*, float*, int64_t, float, float) {}
void down2_row_u16_to_i32_avx2(const uint16_t*, const uint16_t*, int32_t*, int64_t, int32_t) {}
void gray_rgb_u8_row_avx2(const uint8_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {}
void gray_rgb_u16_row_avx2(const uint16_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {}
void merge_recip_packed_row_avx2(const uint16_t*, const void*, bool, void*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t, uint64_t, int32_t) {}
int select_a_no_bg_packed_row_avx2(int32_t*, const int32_t*, int64_t, const void*, bool, void*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int state, int64_t, int, int) { return state; }
int select_a_background_row_avx2(int32_t*, const int32_t*, int64_t, const void*, bool, bool, uint16_t*, float*, bool, int state, int64_t* done, int32_t, int32_t, int, int, int) { *done = 0; return state; }
void tile_accumulate_f32_1c_row_avx2(const float*, const float*, float*, float*, int64_t) {}
void tile_finalize_3c_row_avx2(const float*, const float*, uint8_t*, float*, int64_t, int64_t, bool) {}
bool pyr_any_gt_u16_avx2(const uint16_t*, const uint16_t*, int64_t) { return false; }
void pyr_level0_merge_3c_row_avx2(const uint16_t*, uint16_t*, uint16_t*, const float*, const float*, const float*, const float*, const float*, const void*, bool, float*, int64_t, float, uint16_t, bool, float, float, float, float, float, float) {}
void pyr_merge_f32_3c_row_avx2(const uint16_t*, uint16_t*, uint16_t*, const float*, float*, int64_t, uint16_t) {}
void pyr_output_f32_to_u8_row_avx2(const float*, uint8_t*, int64_t) {}
void pyr_output_f32_to_f32_row_avx2(const float*, float*, int64_t) {}
void pyr_up_horiz_1c_row_avx2(const float*, float*, int64_t, int64_t, float, float, float, float) {}
void pyr_up_horiz_3c_row_avx2(const float*, float*, int64_t, int64_t, float, float, float, float) {}
void raw_row_to_f32_scaled_avx2(const void*, bool, float*, int64_t, float) {}
void pyr_down_horiz_f32_3c_row_avx2(const float*, float*, int64_t, float, float) {}
void align_warp_translation_rgb_row_avx2(const void*, const void*, bool, void*, int64_t, const int32_t*, const int32_t*, const uint32_t*, const uint8_t*, uint32_t, bool, int64_t) {}
void align_warp_translation_gray_row_avx2(const void*, const void*, bool, void*, int64_t, const int32_t*, const int32_t*, const uint32_t*, const uint8_t*, uint32_t) {}
void align_refine_dense_row_avx2(const uint16_t*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, uint32_t, int64_t*, int64_t*) {}
void align_valid_mask_row_avx2(uint8_t*, int64_t, float, float, float, float, float, float, float, float, float) {}
void align_gray16_rgb_row_avx2(const void*, bool, uint16_t*, int64_t, int64_t) {}
void align_down_half_vert_row_avx2(const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint32_t*, int64_t) {}
void align_down_half_horiz_row_avx2(const uint32_t*, uint16_t*, int64_t, int64_t) {}
void align_fast_downsize_u16_row_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t) {}
void align_warp_alignment_row_avx2(const void*, int64_t, int64_t, bool, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, float, float, float, float, float, float, float) {}
void align_score_chunk_packed_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const int64_t*, int64_t, int64_t, int64_t*, int64_t*, int64_t*) {}
void align_score_grid_4d_packed_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, float, float, const float*, const float*, const float*, int64_t, int64_t, const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t*, float*, int64_t) {}
void resample_u16_vert_acc_row_avx2_fma(const uint16_t*, float*, int64_t, float) {}
void resample_u16_horiz_rep_row_avx2_fma(const float*, float*, uint16_t*, int64_t, int64_t, const int32_t*, const float*, int64_t, int64_t, int64_t) {}
int64_t resample_u16_horiz_taps_row_avx2_fma(const float*, uint16_t*, int64_t x0, int64_t, const int32_t*, int64_t, const float*, int64_t, const int32_t*, int64_t) { return x0; }
void activity_row_u16_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t) {}
void detail_row_u16_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t) {}
void select_activity_row_u16_avx2(const uint16_t*, uint16_t*, uint16_t*, uint16_t*, int64_t, uint16_t, uint32_t) {}
void select_detail_row_u16_avx2(const uint16_t*, uint16_t*, uint16_t*, uint16_t*, int64_t, uint16_t) {}
void output_u16_to_u8_row_avx2(const uint16_t*, uint8_t*, int64_t, bool) {}
void output_u16_to_f32_row_avx2(const uint16_t*, float*, int64_t, float) {}
void tile_accumulate_3c_row_avx2(const void*, int, const float*, float*, float*, int64_t) {}
void ecc_prepare_level_row_avx2(const float*, const float*, const float*, int, int, double, double, double, double, double, int, bool, bool, float*, float*, float*, double&, double*, double*, double*) {}
void ecc_ic_pass_all_in_row_p6_avx2(const float*, int, int, const float*, const float*, const float*, int, double, double, double, double, double, double, double, double&, double&, double&, double*) {}
void ecc_ic_pass_all_in_row_p8_avx2(const float*, int, int, const float*, const float*, const float*, int, double, double, double, double, double, double, double, double, double, double&, double&, double&, double*) {}

#endif

}  // namespace simd
}  // namespace openfocus
