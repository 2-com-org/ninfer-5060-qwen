#pragma once

// The NVFP4 activation encoder of the GGUF two-pass products (ggml_bridge.h, LtForm::Nvfp4) and the
// cuBLASLt scale layout, free of the vendored ggml headers: the bridge includes it, and so do
// producers outside it that encode their output in lt_fp4_encode_kernel's form (rmsnorm_lt.cu).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf::detail {

// Byte offset of the UE8M0 exponent of 32-value group `group` of row `row` in cuBLASLt's
// VEC32_UE8M0 scale layout for rows of `groups` groups: 512-byte tiles of 128 rows x 4 groups,
// row-major over (row / 128, group / 4), inside a tile (row % 32) * 16 + (row % 128 / 32) * 4 +
// group % 4.
__host__ __device__ constexpr std::size_t lt_scale_offset(int row, int group, int groups) {
    return (std::size_t(row / 128) * std::size_t((groups + 3) / 4) + std::size_t(group / 4)) * 512 +
           std::size_t(row % 32) * 16 + std::size_t(row % 128 / 32) * 4 + std::size_t(group % 4);
}

// The power of two g with max |x| g <= 6 x 448 = 2688, the largest NVFP4 value: an NVFP4
// operand's global (1 for an all-zero operand; within 2^+-60, so alpha stays a normal FP32).
__host__ __device__ inline float lt_fp4_global(float amax) {
    if (!(amax > 0.0f)) { return 1.0f; }
    if (amax < 1e-15f) { return ldexpf(1.0f, 60); }
    int e = 0;
    (void)frexpf(2688.0f / amax, &e); // 2688 / amax = f 2^e, f in [0.5, 1)
    return ldexpf(1.0f, e - 1 < -60 ? -60 : e - 1);
}

// The 16 values of group `group` of column `column` of x [columns][k], input columns gathered
// (value j of the column is x[input_columns[j]]) when given. x may be global or shared memory.
__device__ __forceinline__ void lt_fp4_load16(const __nv_bfloat16* __restrict__ x,
                                              const std::int32_t* __restrict__ input_columns,
                                              int k, int column, int group, float* v) {
    const __nv_bfloat16* row = x + std::int64_t(column) * k;
    if (input_columns == nullptr) {
        const auto* source = reinterpret_cast<const uint4*>(row + 16 * group);
        const uint4 w[2]   = {source[0], source[1]};
        const auto* h      = reinterpret_cast<const __nv_bfloat162*>(w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            v[2 * j]       = f.x;
            v[2 * j + 1]   = f.y;
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) { v[j] = __bfloat162float(row[input_columns[16 * group + j]]); }
}

#if defined(NINFER_SM120_FP8)
__device__ __forceinline__ unsigned short fp4_scale_code(float value) {
    unsigned short code;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(code) : "f"(0.0f), "f"(value));
    return static_cast<unsigned short>(code & 0xFFu);
}

__device__ __forceinline__ float fp4_scale_value(unsigned short code) {
    unsigned h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(code));
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xFFFFu)));
}

// 1 / scale, __fdividef(1, scale) for an E4M3 scale (normal in FP32, so no subnormal guard), 0
// for scale 0 (code 0).
__device__ __forceinline__ float fp4_inverse(float scale) {
    float inv;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(scale));
    return scale > 0.0f ? inv : 0.0f;
}

// 16 values -> their e2m1 codes and E4M3 scale code: of the candidates RN(amax / 6), RN(amax / 4)
// and the next codes above each (indices 0 to 3), the one with the least squared error (the first
// on a tie; a nonzero group takes at least code 1, 2^-9, which an underflowed RN would zero), its
// index to *which when given. On the 27B (1m quick perplexity corpus, head on MXFP8) it costs +1.7%
// perplexity against MXFP8 where RN(amax / 6) alone costs +2.2%.
__device__ __forceinline__ unsigned short fp4_encode16(const float* v, uint2& out,
                                                      unsigned* which = nullptr) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { amax = fmaxf(amax, fabsf(v[i])); }
    const unsigned short least = amax > 0.0f ? 1 : 0;
    const unsigned short s6    = max(fp4_scale_code(amax * (1.0f / 6.0f)), least);
    const unsigned short s4    = max(fp4_scale_code(amax * 0.25f), least);
    // The values as half2 pairs once, for every candidate's error.
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]); }
    const unsigned short candidates[4] = {s6, s4,
                                          static_cast<unsigned short>(s6 < 0x7Eu ? s6 + 1 : s6),
                                          static_cast<unsigned short>(s4 < 0x7Eu ? s4 + 1 : s4)};
    // Each candidate's squared error: h times 1 / its scale minus the values of the e2m1 codes of
    // v times 1 / its scale, accumulated in half2 in units of the scale, then times its scale^2.
    // The four sums run side by side, each in its own fixed order.
    float scale[4], inv[4];
    __half2 inv2[4], sum[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        scale[c] = fp4_scale_value(candidates[c]);
        inv[c]   = fp4_inverse(scale[c]);
        inv2[c]  = __float2half2_rn(inv[c]);
        sum[c]   = __float2half2_rn(0.0f);
    }
    uint2 codes[4];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const float* x = v + 8 * w;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned word, d[4];
            asm("{\n .reg .b8 q0, q1, q2, q3;\n"
                " cvt.rn.satfinite.e2m1x2.f32 q0, %6, %5;\n"
                " cvt.rn.satfinite.e2m1x2.f32 q1, %8, %7;\n"
                " cvt.rn.satfinite.e2m1x2.f32 q2, %10, %9;\n"
                " cvt.rn.satfinite.e2m1x2.f32 q3, %12, %11;\n"
                " cvt.rn.f16x2.e2m1x2 %1, q0;\n"
                " cvt.rn.f16x2.e2m1x2 %2, q1;\n"
                " cvt.rn.f16x2.e2m1x2 %3, q2;\n"
                " cvt.rn.f16x2.e2m1x2 %4, q3;\n"
                " mov.b32 %0, {q0, q1, q2, q3};\n}"
                : "=r"(word), "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                : "f"(x[0] * inv[c]), "f"(x[1] * inv[c]), "f"(x[2] * inv[c]), "f"(x[3] * inv[c]),
                  "f"(x[4] * inv[c]), "f"(x[5] * inv[c]), "f"(x[6] * inv[c]), "f"(x[7] * inv[c]));
            (w == 0 ? codes[c].x : codes[c].y) = word;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const __half2 e = __hfma2(h[4 * w + b], inv2[c],
                                          __hneg2(*reinterpret_cast<const __half2*>(&d[b])));
                sum[c]          = __hfma2(e, e, sum[c]);
            }
        }
    }
    unsigned short best_code = s6;
    float best               = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float2 pair = __half22float2(sum[c]);
        const float error = (pair.x + pair.y) * scale[c] * scale[c];
        if (c == 0 || error < best) {
            best      = error;
            best_code = candidates[c];
            out       = codes[c];
            if (which != nullptr) { *which = unsigned(c); }
        }
    }
    return best_code;
}

// fp4_encode16's result for its candidate `which` by the same operations: when `which` is the
// index fp4_encode16 reports for these values, the same codes and scale code, without the search.
__device__ __forceinline__ unsigned short fp4_encode16_pick(const float* v, unsigned which,
                                                           uint2& out) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { amax = fmaxf(amax, fabsf(v[i])); }
    const unsigned short least = amax > 0.0f ? 1 : 0;
    unsigned short code =
        max(fp4_scale_code(amax * ((which & 1u) != 0 ? 0.25f : (1.0f / 6.0f))), least);
    if ((which & 2u) != 0 && code < 0x7Eu) { ++code; }
    const float inv = fp4_inverse(fp4_scale_value(code));
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const float* x = v + 8 * w;
        unsigned word;
        asm("{\n .reg .b8 q0, q1, q2, q3;\n"
            " cvt.rn.satfinite.e2m1x2.f32 q0, %2, %1;\n"
            " cvt.rn.satfinite.e2m1x2.f32 q1, %4, %3;\n"
            " cvt.rn.satfinite.e2m1x2.f32 q2, %6, %5;\n"
            " cvt.rn.satfinite.e2m1x2.f32 q3, %8, %7;\n"
            " mov.b32 %0, {q0, q1, q2, q3};\n}"
            : "=r"(word)
            : "f"(x[0] * inv), "f"(x[1] * inv), "f"(x[2] * inv), "f"(x[3] * inv),
              "f"(x[4] * inv), "f"(x[5] * inv), "f"(x[6] * inv), "f"(x[7] * inv));
        (w == 0 ? out.x : out.y) = word;
    }
    return code;
}

// The shift of a group lt_fp4_scale_kernel encodes at the global: its max |x| lies below 2^-119,
// so its own power of two leaves FP32's normal range.
constexpr int kFp4Reencode = -128;

constexpr int kFp4EncodeThreads = 256;

// The NVFP4 activation of x [columns][k] before its global is known, for group `group` of column
// `column` (values v, max |v| amax): fp4_encode16 of the group times its own power of two 2^n, the
// n that takes amax into [128, 256), gives its e2m1 codes and E4M3 scale code, and n goes to
// `shifts` at the scale's place (kFp4Reencode where 2^n is not a normal FP32).
__device__ __forceinline__ void lt_fp4_encode_group(float* v, float amax,
                                                    std::uint8_t* __restrict__ codes,
                                                    std::uint8_t* __restrict__ scales,
                                                    std::int8_t* __restrict__ shifts, int k,
                                                    int column, int group) {
    uint2 c              = make_uint2(0u, 0u);
    unsigned short scale = 0;
    int shift            = 0;
    if (amax > 0.0f) {
        // amax = f 2^(biased - 126), f in [0.5, 1): times 2^(134 - biased) it is f 2^8.
        const int biased = int((__float_as_uint(amax) >> 23) & 0xFFu);
        shift            = 134 - biased;
        if (shift > 127) {
            shift = kFp4Reencode;
        } else {
            const float norm = __uint_as_float(unsigned(127 + shift) << 23);
#pragma unroll
            for (int j = 0; j < 16; ++j) { v[j] *= norm; }
            scale = fp4_encode16(v, c);
        }
    }
    *reinterpret_cast<uint2*>(codes + std::int64_t(column) * (k / 2) + 8 * group) = c;
    const std::size_t at = lt_scale_offset(column, group, k / 16);
    scales[at]           = static_cast<std::uint8_t>(scale & 0xFFu);
    shifts[at]           = static_cast<std::int8_t>(shift);
}

// Max of every amax of a Block-thread block into the slot (FP32 bits), one atomic.
template <int Block = kFp4EncodeThreads>
__device__ __forceinline__ void lt_fp4_block_max(float amax, unsigned* __restrict__ slot) {
    __shared__ float warp_amax[Block / 32];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    }
    if ((threadIdx.x & 31) == 0) { warp_amax[threadIdx.x / 32] = amax; }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int w = 1; w < Block / 32; ++w) { amax = fmaxf(amax, warp_amax[w]); }
        atomicMax(slot, __float_as_uint(amax));
    }
}
#endif

} // namespace ninfer::ops::gguf::detail
