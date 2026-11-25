// AVX-512 (F, DQ, BW, VL) row kernels. This file alone is compiled with the
// AVX-512 flags; the kernels run only when use_avx512() reports both CPU and
// operating-system support, and each produces exactly the output of the AVX2
// and scalar rows it replaces.
//
// The file includes no standard-library templates or inline functions, so no
// code generated with AVX-512 enabled can be shared with other files through
// the linker's merging of identical inline definitions.

#include "simd_dispatch.hpp"

#include <cstdint>

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__) && defined(__AVX512BW__) \
    && defined(__AVX512DQ__) && defined(__AVX512VL__)
#define OPENFOCUS_AVX512 1
#include <immintrin.h>
#else
#define OPENFOCUS_AVX512 0
#endif

namespace openfocus {
namespace simd {

#if OPENFOCUS_AVX512

bool avx512_compiled() {
    return true;
}

namespace {

// Sixteen-bit lane 2k of channel m's index vector selects sample 3k + m of a
// 48-sample span held in two registers (indices 32 and up address the second);
// the odd lanes are zeroed, so each 32-bit lane holds one sample.
inline __m512i gray_channel_index(int m) {
    alignas(64) uint16_t idx[32];
    for (int k = 0; k < 16; ++k) {
        idx[2 * k] = static_cast<uint16_t>(3 * k + m);
        idx[2 * k + 1] = 0;
    }
    return _mm512_load_si512(idx);
}

// r + g + b of sixteen packed pixels whose 48 samples are lo (0..31) and the
// first sixteen lanes of hi (32..47), as sixteen uint32 lanes.
inline __m512i rgb_sum16(__m512i lo, __m512i hi, __m512i i0, __m512i i1, __m512i i2) {
    const __mmask32 even = 0x55555555u;
    const __m512i c0 = _mm512_maskz_permutex2var_epi16(even, lo, i0, hi);
    const __m512i c1 = _mm512_maskz_permutex2var_epi16(even, lo, i1, hi);
    const __m512i c2 = _mm512_maskz_permutex2var_epi16(even, lo, i2, hi);
    return _mm512_add_epi32(_mm512_add_epi32(c0, c1), c2);
}

// The gray code of each lane: (t * scale * 85) >> 8 for depth colour, else
// (t / 3) * scale, in uint32 arithmetic as the scalar row.
inline __m512i gray_code16(__m512i t, bool depth_color, __m512i v_scale, __m512i v_s85, __m512i v_magic) {
    if (depth_color) {
        return _mm512_srli_epi32(_mm512_mullo_epi32(t, v_s85), 8);
    }
    const __m512i q_even = _mm512_srli_epi64(_mm512_mul_epu32(t, v_magic), 33);
    const __m512i q_odd = _mm512_slli_epi64(
        _mm512_srli_epi64(_mm512_mul_epu32(_mm512_srli_epi64(t, 32), v_magic), 33), 32);
    return _mm512_mullo_epi32(_mm512_mask_blend_epi32(0xAAAA, q_even, q_odd), v_scale);
}

// Only the destination in use is offset; the other may be null.
inline void store_gray16(__m512i code, uint16_t* dst16, int32_t* dst32, int64_t x, bool dst_is_u16) {
    if (dst_is_u16) {
        // vpmovdw keeps the low sixteen bits, as the uint16 cast does.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst16 + x), _mm512_cvtepi32_epi16(code));
    } else {
        _mm512_storeu_si512(dst32 + x, code);
    }
}

}  // namespace

int64_t gray_rgb_u16_row_avx512(
    const uint16_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m512i i0 = gray_channel_index(0), i1 = gray_channel_index(1), i2 = gray_channel_index(2);
    const __m512i v_magic = _mm512_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m512i v_scale = _mm512_set1_epi32(scale);
    const __m512i v_s85 = _mm512_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 16 <= width; x += 16) {
        const uint16_t* p = sr + 3 * x;
        const __m512i lo = _mm512_loadu_si512(p);
        const __m512i hi = _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)));
        const __m512i code = gray_code16(rgb_sum16(lo, hi, i0, i1, i2), depth_color, v_scale, v_s85, v_magic);
        store_gray16(code, dst16, dst32, x, dst_is_u16);
    }
    return x;
}

int64_t gray_rgb_u8_row_avx512(
    const uint8_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m512i i0 = gray_channel_index(0), i1 = gray_channel_index(1), i2 = gray_channel_index(2);
    const __m512i v_magic = _mm512_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m512i v_scale = _mm512_set1_epi32(scale);
    const __m512i v_s85 = _mm512_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 16 <= width; x += 16) {
        const uint8_t* p = sr + 3 * x;
        // Bytes 0..31 and 32..47, widened to sixteen bits; both loads stay in the 48 bytes.
        const __m512i lo = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
        const __m512i hi = _mm512_castsi256_si512(
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32))));
        const __m512i code = gray_code16(rgb_sum16(lo, hi, i0, i1, i2), depth_color, v_scale, v_s85, v_magic);
        store_gray16(code, dst16, dst32, x, dst_is_u16);
    }
    return x;
}

#else

bool avx512_compiled() {
    return false;
}

int64_t gray_rgb_u16_row_avx512(const uint16_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {
    return 0;
}

int64_t gray_rgb_u8_row_avx512(const uint8_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {
    return 0;
}

#endif

}  // namespace simd
}  // namespace openfocus
