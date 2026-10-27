#pragma once

// The tile kernel: the prefill product of the one-scale (IQ3_XXS, IQ3_S, IQ2_XXS, IQ4_XS) and
// two-scale (IQ2_XS, IQ2_S) i-quants on sm_120, and in its MXFP8 form also of Q2_K and Q4_K. A CTA
// owns 128 weight rows x 128 tokens with 384 threads in two roles. Four producer warps keep
// cp.async rings of the raw blocks (two blocks per row) and of 64-value activation stages (four),
// and decode each stage with the vector kernels' Decoder<type> (a thread a weight row) into a
// three-slot 8-bit ring swizzled for ldmatrix. Eight consumer warps (2 x 4, a warp 64 rows x 32
// tokens) run ldmatrix and the tensor-core products. Named barriers hand the stages over: full[s]
// (producers arrive, consumers wait) and empty[s] (consumers arrive, producers wait). Results go
// straight through the product's epilogue.
//
// Two arithmetics, chosen by the activation's form (TileArgs::mxfp8):
// - q8_1, as the matrix kernel does it: int8 weights, one mma.m16n8k32 per 32 values (two m16n8k16
//   for per-16 scales), every 32-value integer dot scaled by its weight and activation factors in
//   FP32.
// - MXFP8 (a 120a build): the producers write each 32-value weight slice as e4m3 under one
//   power-of-two (UE8M0) scale and the consumers run the block-scaled e4m3 mma.m16n8k32, which
//   applies both operands' scales and accumulates in FP32. The i-quants and Q2_K decode through
//   byte-permute lookups (fp8_lut_decode): their codes take a few values (grid magnitudes, IQ4_XS's
//   16 levels, Q2_K's 4), each slice rounds those values times its scale to e4m3 once (FP32), and
//   a word of four weights is a PRMT or two of them. Q4_K decodes to int8 and converts in FP16
//   (the values times their factors, plus the minimum terms).
//   The tensor pipe shares issue with the producers, so the decode's instruction count sets the
//   rate. On the RTX 5060 Ti the form runs 1.26-1.41x the q8_1 form's rate for the one-scale
//   i-quants and about 2x for IQ2_XS and IQ2_S, at about 4x its per-product error (3.4-3.8%
//   relative RMS against 0.9%).
//
// The activation is planar: 8-bit values [columns][K] and FP32 d [K / 32][columns rounded up to
// 128] (ggml_bridge.cu quantize_tile_activation).

#include "ggml_bridge_vec.cuh"

#include <stdexcept>

namespace ninfer::ops::gguf::detail {

namespace tile {

constexpr int kBM = 128, kBN = 128, kStepK = 64, kSubs = kStepK / 32;
constexpr int kConsumers = 256, kProducers = 128, kThreads = kConsumers + kProducers;
constexpr int kASlots = 3, kQSlots = 4;
constexpr int kActSlotQ = kBN * kStepK;
constexpr int kActSlot  = kActSlotQ + kSubs * kBN * 4;
constexpr int kAPlane   = kBM * 32;
constexpr int kFull = 1, kEmpty = 1 + kASlots, kProducerBar = 1 + 2 * kASlots;

template <ggml_type type>
inline constexpr int kScales =
    type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_Q2_K ? 2 : 1;

// The K-quants (minimum terms, Q2_K's per-16 scales) have only the MXFP8 form. Q6_K's 210-byte
// blocks overflow the raw ring's shared memory and stay on the matrix kernel.
template <ggml_type type>
inline constexpr bool kFp8Only = type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K;

// Raw-ring copy unit and the guaranteed alignment of a block's first byte: IQ4_XS decodes with
// 8-byte loads (its 136-byte blocks are 8-byte aligned), every other type with 2-byte ones. A copy
// covers the block's unit-aligned superset; 8-byte units halve the producers' copy instructions
// against 4-byte ones (+4-7% kernel rate); 16-byte ones gain less outside Q4_K (bank conflicts
// in the decode at 16-byte row strides). The over-read stays inside the 256-byte payload padding.
template <ggml_type type>
inline constexpr int kCopy = type == GGML_TYPE_Q4_K ? 16 : 8;
// How the producers copy a block from each of their rows. A row per thread puts 32 rows behind
// every copy instruction. In the MXFP8 form, blocks of 98 bytes and more go a row at a time across
// 16 lanes (32 for IQ4_XS's 17 units), so an instruction covers two rows' contiguous bytes; on the
// RTX 5060 Ti that measured IQ3_XXS +2-4%, IQ3_S +4-6%, IQ4_XS 0-3% and Q4_K +4-10%. The two-bit
// types (66-84 bytes, half those lanes idle) lost about 2% that way and the q8_1 form 1-7%.
template <ggml_type type>
inline constexpr bool kRowCopies = vec::Decoder<type>::kBlockBytes >= 98;
template <ggml_type type>
inline constexpr int kAlign = type == GGML_TYPE_Q4_K    ? 16
                              : type == GGML_TYPE_IQ4_XS ? 8
                              : type == GGML_TYPE_Q2_K   ? 4
                                                         : 2;

// One-scale types visit token tiles fastest, so the token tiles of a row block share its weights
// in L2 (the 27B's large matrices exceed it); the two-scale ones keep row tiles fastest, which
// measured faster for them (their 2.3-2.5 bits per weight matrices mostly fit L2).
template <ggml_type type>
inline constexpr bool kTokensFastest = kScales<type> == 1;

template <ggml_type type>
struct Layout {
    using D                     = vec::Decoder<type>;
    static constexpr int kTable = D::kTableWords; // Q4_K's layout fills 101376 bytes exactly
    // A block's copy-unit-aligned superset at an odd unit stride: a row stride of a 16-byte
    // multiple puts the producers' same-offset decode loads from 32 rows into one bank.
    static constexpr int kUnits =
        ((D::kBlockBytes + kCopy<type> - kAlign<type> + kCopy<type> - 1) / kCopy<type>) | 1;
    static constexpr int kRawRow  = kUnits * kCopy<type>;
    static constexpr int kRawSlot = kBM * kRawRow;
    static constexpr int kASlot   = kSubs * kAPlane + kSubs * kBM * kScales<type> * 4;
    static constexpr int kRawOff  = kTable * 4;
    static constexpr int kAOff    = kRawOff + 2 * kRawSlot;
    static constexpr int kActOff  = kAOff + kASlots * kASlot;
    static constexpr int kBytes   = kActOff + kQSlots * kActSlot;
};

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(int (&a)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                 "{%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                 "{%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// The MXFP8 product, block-scaled e4m3: lane 4g supplies A row g's scale, lane 4g + 1 A row g + 8's
// and lane 4g B column g's (thread and byte selector 0).
__device__ __forceinline__ void mma_e4m3_k32(float (&c)[4], const int (&a)[4], int b0, int b1,
                                             unsigned sa, unsigned sb) {
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
                 "{%13}, {%14,%15};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa),
                   "h"((unsigned short)0), "h"((unsigned short)0), "r"(sb),
                   "h"((unsigned short)0), "h"((unsigned short)0));
}

// Largest decoded integer magnitude per type (the e4m3 image's range is sized on it).
template <ggml_type type>
inline constexpr float kFp8Max = type == GGML_TYPE_IQ3_XXS   ? 62.0f
                                 : type == GGML_TYPE_IQ3_S   ? 15.0f
                                 : type == GGML_TYPE_IQ4_XS  ? 127.0f
                                 : type == GGML_TYPE_Q2_K    ? 3.0f
                                 : type == GGML_TYPE_Q4_K    ? 15.0f
                                                             : 43.0f; // the IQ2 grids

// Four decoded int8 values v as v * c + m in FP16 (one rounding), then rounded to e4m3.
__device__ __forceinline__ unsigned e4m3x4(int w, __half2 c, __half2 m) {
    const unsigned u  = unsigned(w) ^ 0x80808080u; // value + 128
    const unsigned lo = __byte_perm(u, 0x64646464u, 0x4140);
    const unsigned hi = __byte_perm(u, 0x64646464u, 0x4342);
    const __half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152
    const __half2 l = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo), bias), c, m);
    const __half2 h = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi), bias), c, m);
    unsigned short pl, ph;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
        : "=h"(pl)
        : "r"(*reinterpret_cast<const unsigned*>(&l)));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
        : "=h"(ph)
        : "r"(*reinterpret_cast<const unsigned*>(&h)));
    return unsigned(pl) | (unsigned(ph) << 16);
}

// The UE8M0 exponent E (biased) and multiplier 2^-E * f such that kFp8Max * |f| * 2^-E is in
// (224, 448]; a zero scale takes E = 127 (2^0), its values all zero.
template <ggml_type type>
__device__ __forceinline__ unsigned fp8_exponent(float fmax) {
    const unsigned bits = __float_as_uint(fmax * (kFp8Max<type> / 448.0f));
    if (fmax == 0.0f) { return 127u; }
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1u : 0u);
}

// The MXFP8 lookup decode of the grid i-quants. A grid entry's bytes take one of a few magnitudes:
// IQ3_XXS {4, 12, ..., 52, 62} (byte >> 3), IQ3_S {1, 3, ..., 15} (byte >> 1), the IQ2 grids
// {8, 25, 43} (byte >> 4). The shared-memory table holds per grid entry the PRMT selectors of its
// magnitude indices, four 4-bit selectors a word (the IQ2 entries' eight values in one word's two
// halves). Per slice the magnitudes times the slice's scale (2^-E folded in) are rounded to e4m3
// once into lookup words; a word of four values is one PRMT of them with the sign bits ORed in.
// IQ4_XS's 16 levels and Q2_K's 4 values per 16-value half look up in registers.
template <ggml_type type>
inline constexpr bool kFp8Lut = type != GGML_TYPE_Q4_K;
template <ggml_type type>
inline constexpr bool kFp8Lut8 = type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
// Selector words in shared memory: one per grid word (IQ3), one per two (IQ2's two-word entries),
// none for IQ4_XS and Q2_K.
template <ggml_type type>
inline constexpr int kLutWords = kFp8Lut8<type> ? vec::Decoder<type>::kTableWords
                                                : vec::Decoder<type>::kTableWords / 2;

__device__ __forceinline__ std::uint32_t lut_selectors(std::uint32_t w, int shift) {
    return ((w >> shift) & 7) | (((w >> (8 + shift)) & 7) << 4) | (((w >> (16 + shift)) & 7) << 8) |
           (((w >> (24 + shift)) & 7) << 12);
}

// Word i of the selector table, from the type's grid.
template <ggml_type type>
__device__ __forceinline__ std::uint32_t lut_table_word(const std::uint32_t* grid, int i) {
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return lut_selectors(grid[i], 3);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        return lut_selectors(grid[i], 1);
    } else {
        return lut_selectors(grid[2 * i], 4) | (lut_selectors(grid[2 * i + 1], 4) << 16);
    }
}

__device__ __forceinline__ unsigned e4m3_pair(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
// The e4m3 lookup words of eight magnitudes m * c: lo = m0..m3, hi = m4..m7.
__device__ __forceinline__ void lut8(float c, float m0, float m1, float m2, float m3, float m4,
                                     float m5, float m6, float m7, unsigned& lo, unsigned& hi) {
    lo = e4m3_pair(m1 * c, m0 * c) | (e4m3_pair(m3 * c, m2 * c) << 16);
    hi = e4m3_pair(m5 * c, m4 * c) | (e4m3_pair(m7 * c, m6 * c) << 16);
}
// The IQ2 grids' three magnitudes {8, 25, 43} * c.
__device__ __forceinline__ unsigned lut3(float c) {
    return e4m3_pair(25.0f * c, 8.0f * c) | (e4m3_pair(0.0f, 43.0f * c) << 16);
}
// The e4m3 values at the even (x) and odd (y) 4-bit indices of q in a 16-entry table, entries
// 4i..4i+3 in t[i] (vecdotq's get_int_from_table_16 with the table in registers).
__device__ __forceinline__ int2 lut16(std::uint32_t q, const unsigned (&t)[4]) {
    const std::uint32_t pick = 0x32103210u | ((q & 0x88888888u) >> 1);
    const std::uint32_t r = q >> 16;
    const unsigned w0 = __byte_perm(__byte_perm(t[0], t[1], q), __byte_perm(t[2], t[3], q), pick);
    const unsigned w1 =
        __byte_perm(__byte_perm(t[0], t[1], r), __byte_perm(t[2], t[3], r), pick >> 16);
    return make_int2(int(__byte_perm(w0, w1, 0x6420)), int(__byte_perm(w0, w1, 0x7531)));
}
// The e4m3 values at four 2-bit indices (one a byte, bits 0-1) in a 4-entry table t.
__device__ __forceinline__ int lut4(std::uint32_t q, unsigned t) {
    const std::uint32_t n = q | (q >> 4); // indices 0, 1 in byte 0, indices 2, 3 in byte 2
    return int(__byte_perm(t, 0, __byte_perm(n, 0, 0x0020)));
}
// Four e4m3 values: magnitudes looked up by sel, sign bit j of signs negating value j.
__device__ __forceinline__ int lut_word(unsigned lo, unsigned hi, std::uint32_t sel,
                                        std::uint32_t signs) {
    return int(__byte_perm(lo, hi, sel) | (((signs & 0xF) * 0x10204080u) & 0x80808080u));
}

// Decodes 32-value slice u of block b into eight e4m3 words (Decoder<type>'s value order);
// returns the slice's UE8M0 exponent.
template <ggml_type type>
__device__ __forceinline__ unsigned fp8_lut_decode(const std::uint8_t* b, int u,
                                                   const std::uint32_t* tab, int (&v)[8]) {
    const float d = vec::half_at(b);
    if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const std::uint32_t q0  = vec::u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1  = vec::u32_a2(b + 6 + 8 * u);
        const std::uint32_t aux = vec::u32_a2(b + 66 + 4 * u);
        const float f0          = d * (0.5f + float(aux >> 28)) * 0.5f;
        const unsigned e        = fp8_exponent<type>(fabsf(f0));
        unsigned lo, hi;
        lut8(f0 * __uint_as_float((254u - e) << 23), 4, 12, 20, 28, 36, 44, 52, 62, lo, hi);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t q  = (l < 2 ? q0 : q1) >> (16 * (l & 1));
            const std::uint32_t sg = vec::ksigns((aux >> (7 * l)) & 0x7F);
            v[2 * l]     = lut_word(lo, hi, tab[q & 0xFF], sg);
            v[2 * l + 1] = lut_word(lo, hi, tab[(q >> 8) & 0xFF], sg >> 4);
        }
        return e;
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const std::uint32_t q0 = vec::u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1 = vec::u32_a2(b + 6 + 8 * u);
        const std::uint32_t qh = b[66 + u];
        const std::uint32_t sg = vec::u32_a2(b + 74 + 4 * u);
        const int sc           = (b[106 + u / 2] >> (4 * (u & 1))) & 0xF;
        const float f0         = d * float(1 + 2 * sc);
        const unsigned e       = fp8_exponent<type>(fabsf(f0));
        unsigned lo, hi;
        lut8(f0 * __uint_as_float((254u - e) << 23), 1, 3, 5, 7, 9, 11, 13, 15, lo, hi);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const std::uint32_t idx =
                (((w < 4 ? q0 : q1) >> (8 * (w & 3))) & 0xFF) | ((qh << (8 - w)) & 0x100);
            v[w] = lut_word(lo, hi, tab[idx], sg >> (4 * w));
        }
        return e;
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const std::uint32_t idx = vec::u32_a2(b + 2 + 8 * u);
        const std::uint32_t aux = vec::u32_a2(b + 6 + 8 * u);
        const float f0          = d * (0.5f + float(aux >> 28)) * 0.25f;
        const unsigned e        = fp8_exponent<type>(fabsf(f0));
        const unsigned lut      = lut3(f0 * __uint_as_float((254u - e) << 23));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t t  = tab[(idx >> (8 * l)) & 0xFF];
            const std::uint32_t sg = vec::ksigns((aux >> (7 * l)) & 0x7F);
            v[2 * l]     = lut_word(lut, 0, t, sg);
            v[2 * l + 1] = lut_word(lut, 0, t >> 16, sg >> 4);
        }
        return e;
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        const std::uint32_t sc  = b[66 + u];
        const float f0          = d * (0.5f + float(sc & 0xF)) * 0.25f;
        const float f1          = d * (0.5f + float(sc >> 4)) * 0.25f;
        const unsigned e        = fp8_exponent<type>(fmaxf(fabsf(f0), fabsf(f1)));
        const float unscale     = __uint_as_float((254u - e) << 23);
        const unsigned lut0     = lut3(f0 * unscale), lut1 = lut3(f1 * unscale);
        const std::uint32_t q01 = vec::u32_a2(b + 2 + 8 * u), q23 = vec::u32_a2(b + 6 + 8 * u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t q  = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFF;
            const std::uint32_t t  = tab[q & 511];
            const std::uint32_t sg = vec::ksigns(q >> 9);
            v[2 * l]     = lut_word(l < 2 ? lut0 : lut1, 0, t, sg);
            v[2 * l + 1] = lut_word(l < 2 ? lut0 : lut1, 0, t >> 16, sg >> 4);
        }
        return e;
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const std::uint32_t sh = *reinterpret_cast<const std::uint16_t*>(b + 2);
        const int ls = ((b[4 + u / 2] >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
        const float f0   = d * float(ls - 32);
        const unsigned e = fp8_exponent<type>(fabsf(f0));
        const float c    = f0 * __uint_as_float((254u - e) << 23);
        unsigned t[4]; // kvalues_iq4nl * c
        lut8(c, -127, -104, -83, -65, -49, -35, -22, -10, t[0], t[1]);
        lut8(c, 1, 13, 25, 38, 53, 69, 89, 113, t[2], t[3]);
        const uint2 lo = *reinterpret_cast<const uint2*>(b + 8 + 16 * u);
        const uint2 hi = *reinterpret_cast<const uint2*>(b + 16 + 16 * u);
        const std::uint32_t q[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 w = lut16(q[j], t);
            v[j]         = w.x;
            v[j + 4]     = w.y;
        }
        return e;
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        // Values q * a - m (q in 0..3) per 16-value half; E from the larger half's bound.
        const int h = u / 4, j = u % 4;
        const float2 dm        = __half22float2(*reinterpret_cast<const half2*>(b + 80));
        const std::uint32_t s0 = b[8 * h + 2 * j], s1 = b[8 * h + 2 * j + 1];
        const float a0 = dm.x * float(s0 & 0xF), a1 = dm.x * float(s1 & 0xF);
        const float m0 = dm.y * float(s0 >> 4), m1 = dm.y * float(s1 >> 4);
        constexpr float kInv = 1.0f / kFp8Max<type>;
        const unsigned e =
            fp8_exponent<type>(fmaxf(fabsf(a0) + fabsf(m0) * kInv, fabsf(a1) + fabsf(m1) * kInv));
        const float unscale = __uint_as_float((254u - e) << 23);
        const float c0 = a0 * unscale, n0 = -m0 * unscale, c1 = a1 * unscale, n1 = -m1 * unscale;
        const unsigned t0 =
            e4m3_pair(c0 + n0, n0) | (e4m3_pair(fmaf(3.0f, c0, n0), fmaf(2.0f, c0, n0)) << 16);
        const unsigned t1 =
            e4m3_pair(c1 + n1, n1) | (e4m3_pair(fmaf(3.0f, c1, n1), fmaf(2.0f, c1, n1)) << 16);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t q = (vec::u32_a4(b + 16 + 32 * h + 4 * i) >> (2 * j)) & 0x03030303u;
            v[i]                  = lut4(q, i < 4 ? t0 : t1);
        }
        return e;
    } else {
        static_assert(type == GGML_TYPE_IQ2_S, "fp8_lut_decode: an i-quant or Q2_K");
        const std::uint32_t sc = b[74 + u];
        const float f0         = d * (0.5f + float(sc & 0xF)) * 0.25f;
        const float f1         = d * (0.5f + float(sc >> 4)) * 0.25f;
        const unsigned e       = fp8_exponent<type>(fmaxf(fabsf(f0), fabsf(f1)));
        const float unscale    = __uint_as_float((254u - e) << 23);
        const unsigned lut0    = lut3(f0 * unscale), lut1 = lut3(f1 * unscale);
        const std::uint32_t qs = vec::u32_a2(b + 2 + 4 * u);
        const std::uint32_t sg = vec::u32_a2(b + 34 + 4 * u);
        const std::uint32_t qh = b[66 + u];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t t = tab[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
            v[2 * l]     = lut_word(l < 2 ? lut0 : lut1, 0, t, sg >> (8 * l));
            v[2 * l + 1] = lut_word(l < 2 ? lut0 : lut1, 0, t >> 16, sg >> (8 * l + 4));
        }
        return e;
    }
}

// Slice u of block b as eight e4m3 words and its UE8M0 exponent, the MXFP8 form's weights: the
// lookup decode, or for Q4_K the values q * f0 - f1 in FP16 (E from the bound kFp8Max * |f0| +
// |f1|) rounded to e4m3. `tab` holds the lookup selectors, or the decoder's own table.
template <ggml_type type>
__device__ __forceinline__ unsigned fp8_decode(const std::uint8_t* b, int u,
                                               const std::uint32_t* tab, int (&v)[8]) {
    if constexpr (kFp8Lut<type>) {
        return fp8_lut_decode<type>(b, u, tab, v);
    } else {
        static_assert(type == GGML_TYPE_Q4_K, "fp8_decode: a lookup type or Q4_K");
        vec::Slice sl;
        vec::Decoder<type>::decode(b, u, tab, sl);
        const unsigned e =
            fp8_exponent<type>(fabsf(sl.f[0]) + fabsf(sl.f[1]) * (1.0f / kFp8Max<type>));
        const float unscale = __uint_as_float((254u - e) << 23); // 2^-(e - 127)
        const __half2 c     = __float2half2_rn(sl.f[0] * unscale);
        const __half2 m     = __float2half2_rn(-sl.f[1] * unscale);
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = int(e4m3x4(sl.w[j], c, m)); }
        return e;
    }
}

// Byte offset of half h (16 bytes) of weight row r in a 32-value plane, swizzled for ldmatrix.
__device__ __forceinline__ int a_offset(int r, int h) { return r * 32 + ((h ^ ((r >> 2) & 1)) << 4); }

// Chunk c (16 bytes) of token r in an activation stage (64 values per token), swizzled for ldmatrix.
__device__ __forceinline__ int act_offset(int r, int c) {
    return r * kStepK + ((c ^ ((r >> 1) & 3)) << 4);
}

// Row-scale slot of row r: rows r and r + 8 of a 16-row tile are adjacent (one load per tile).
__device__ __forceinline__ int scale_index(int r) {
    return (r & ~15) | ((r & 7) << 1) | ((r >> 3) & 1);
}

// Stages before the end at which the producers pull the epilogue's reads into L2.
constexpr int kPrefetchStages = 16;

template <ggml_type type, int KS, int EPI, bool FP8 = false>
__launch_bounds__(kThreads, 1) __global__ void kernel(TileArgs p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap(); // dispatched on sm_120 only
#else
    using D                       = vec::Decoder<type>;
    using L                       = Layout<type>;
    constexpr int kS              = kScales<type>;
    constexpr int kStagesPerBlock = D::kBlockElems / kStepK;
    static_assert(kSubs == 2, "tile kernel: two 32-value sub-blocks per stage");
    extern __shared__ __align__(16) std::uint8_t smem[];
    auto* table = reinterpret_cast<std::uint32_t*>(smem);
    if constexpr (FP8 && kFp8Lut<type>) {
        const std::uint32_t* src = D::table();
        vec::stage_words<kThreads, kLutWords<type>>(
            table, [&](int i) { return lut_table_word<type>(src, i); });
    } else if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        vec::stage_words<kThreads, D::kTableWords>(table, [&](int i) { return src[i]; });
    }
    __syncthreads();
    const int tid   = threadIdx.x;
    const int k     = KS != 0 ? KS : p.k;
    const int iters = k / kStepK;
    const int nb    = k / D::kBlockElems;
    const int row0  = (kTokensFastest<type> ? blockIdx.y : blockIdx.x) * kBM;
    const int tok0  = (kTokensFastest<type> ? blockIdx.x : blockIdx.y) * kBN;

    if (tid >= kConsumers) {
        // Producers: thread pt owns weight row pt (raw copies and decode) and a share of the
        // activation copies.
        const int pt             = tid - kConsumers;
        const int row            = min(row0 + pt, p.rows - 1);
        const std::uint8_t* wrow = p.weight + std::int64_t(row) * p.row_bytes;
        const auto issue_raw     = [&](int b) {
            if constexpr (FP8 && kRowCopies<type>) {
                constexpr int kLanes = L::kUnits <= 16 ? 16 : 32; // per row
                static_assert(L::kUnits <= 32);
                std::uint8_t* slot = smem + L::kRawOff + (b & 1) * L::kRawSlot;
                const int lane = pt & 31, unit = lane % kLanes;
#pragma unroll 4
                for (int j = 0; j < kLanes; ++j) {
                    const int r = (pt & ~31) + j * (32 / kLanes) + lane / kLanes;
                    const std::uint8_t* src =
                        p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes +
                        std::int64_t(b) * D::kBlockBytes;
                    const auto* base = reinterpret_cast<const std::uint8_t*>(
                        reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t(kCopy<type> - 1));
                    const int n =
                        int((src + D::kBlockBytes - base + kCopy<type> - 1) / kCopy<type>);
                    if (unit < n) {
                        std::uint8_t* dst = slot + r * L::kRawRow + kCopy<type> * unit;
                        if constexpr (kCopy<type> == 16) {
                            cp_async16(dst, base + 16 * unit);
                        } else {
                            cp_async8(dst, base + 8 * unit);
                        }
                    }
                }
                return;
            }
            const std::uint8_t* src = wrow + std::int64_t(b) * D::kBlockBytes;
            const auto* base        = reinterpret_cast<const std::uint8_t*>(
                reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t(kCopy<type> - 1));
            const int n       = int((src + D::kBlockBytes - base + kCopy<type> - 1) / kCopy<type>);
            std::uint8_t* dst = smem + L::kRawOff + (b & 1) * L::kRawSlot + pt * L::kRawRow;
            for (int j = 0; j < n; ++j) {
                if constexpr (kCopy<type> == 16) {
                    cp_async16(dst + 16 * j, base + 16 * j);
                } else if constexpr (kCopy<type> == 8) {
                    cp_async8(dst + 8 * j, base + 8 * j);
                } else {
                    cp_async4(dst + 4 * j, base + 4 * j);
                }
            }
        };
        const auto issue_act = [&](int i) {
            std::uint8_t* q = smem + L::kActOff + (i % kQSlots) * kActSlot;
#pragma unroll
            for (int j = 0; j < kActSlotQ / 16 / kProducers; ++j) {
                const int c = pt + j * kProducers;
                const int r = c >> 2, part = c & 3;
                const int source = min(tok0 + r, p.columns - 1);
                cp_async16(q + act_offset(r, part),
                           p.qs + std::int64_t(source) * k + i * kStepK + part * 16);
            }
            if (pt < kSubs * kBN / 4) {
                const int u = pt / (kBN / 4), c = pt % (kBN / 4);
                cp_async16(q + kActSlotQ + (u * kBN + 4 * c) * 4,
                           p.d + std::int64_t(i * kSubs + u) * p.columns_pad + tok0 + 4 * c);
            }
        };
        // The epilogue's reads (the residual it accumulates into, the gate it multiplies by) are
        // pulled into L2 some stages before the consumers need them: with one CTA per SM nothing
        // else hides their DRAM latency.
        const auto prefetch_epilogue = [&] {
            const bool gated = p.out.epilogue == Epilogue::GateProduct;
            if (!gated && p.out.epilogue != Epilogue::Accumulate) { return; }
            const int bytes         = gated || p.out.f32 != nullptr ? 4 : 2;
            const int rows_per_line = 128 / bytes;
            const int lines         = kBM / rows_per_line;
            for (int j = pt; j < kBN * lines; j += kProducers) {
                const int col = tok0 + j / lines, r = row0 + (j % lines) * rows_per_line;
                if (col >= p.columns || r >= p.rows) { continue; }
                const void* at =
                    gated ? static_cast<const void*>(p.out.gate + std::int64_t(col) * p.out.gate_column_stride + r)
                    : p.out.f32 != nullptr
                        ? static_cast<const void*>(p.out.f32 + std::int64_t(col) * p.out.column_stride + r)
                        : static_cast<const void*>(p.out.bf16 + std::int64_t(col) * p.out.column_stride + r);
                asm volatile("prefetch.global.L2 [%0];" ::"l"(at));
            }
        };
        const int prefetch_at = max(0, iters - kPrefetchStages);
        issue_raw(0);
        issue_act(0);
        cp_async_commit();
        for (int i = 0; i < iters; ++i) {
            const int s = i % kASlots;
            if (i >= kASlots) { bar_sync(kEmpty + s, kThreads); } // consumers are done with i - 3
            if constexpr (EPI == kEpiBf16Add || EPI == kEpiBf16Silu || EPI == kEpiGeneric) {
                if (i == prefetch_at) { prefetch_epilogue(); }
            }
            if (i + 1 < iters) { issue_act(i + 1); }
            if (i % kStagesPerBlock == 0 && i / kStagesPerBlock + 1 < nb) {
                issue_raw(i / kStagesPerBlock + 1);
            }
            cp_async_commit();
            cp_async_wait<1>();
            bar_sync(kProducerBar, kProducers);
            const int b  = i / kStagesPerBlock;
            const int ph = int((reinterpret_cast<std::uintptr_t>(wrow) +
                                std::uint64_t(b) * D::kBlockBytes) & (kCopy<type> - 1));
            const std::uint8_t* blk =
                smem + L::kRawOff + (b & 1) * L::kRawSlot + pt * L::kRawRow + ph;
            std::uint8_t* as = smem + L::kAOff + s * L::kASlot;
            auto* sc         = reinterpret_cast<float*>(as + kSubs * kAPlane);
#pragma unroll
            for (int us = 0; us < kSubs; ++us) {
                std::uint8_t* a = as + us * kAPlane;
                if constexpr (FP8) {
                    int v[8];
                    const unsigned e =
                        fp8_decode<type>(blk, (i % kStagesPerBlock) * kSubs + us, table, v);
                    *reinterpret_cast<int4*>(a + a_offset(pt, 0)) =
                        make_int4(v[0], v[1], v[2], v[3]);
                    *reinterpret_cast<int4*>(a + a_offset(pt, 1)) =
                        make_int4(v[4], v[5], v[6], v[7]);
                    reinterpret_cast<unsigned*>(sc)[us * kBM + scale_index(pt)] = e;
                    continue;
                }
                vec::Slice sl;
                D::decode(blk, (i % kStagesPerBlock) * kSubs + us, table, sl);
                *reinterpret_cast<int4*>(a + a_offset(pt, 0)) =
                    make_int4(sl.w[0], sl.w[1], sl.w[2], sl.w[3]);
                *reinterpret_cast<int4*>(a + a_offset(pt, 1)) =
                    make_int4(sl.w[4], sl.w[5], sl.w[6], sl.w[7]);
                if constexpr (kS == 1) {
                    sc[us * kBM + scale_index(pt)] = sl.f[0];
                } else {
                    *reinterpret_cast<float2*>(sc + (us * kBM + scale_index(pt)) * 2) =
                        make_float2(sl.f[0], sl.f[1]);
                }
            }
            bar_arrive(kFull + s, kThreads);
        }
        cp_async_wait<0>();
        return;
    }

    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int wm = warp >> 2, wn = warp & 3;
    const int lm_row = (lane & 7) + 8 * ((lane >> 3) & 1), lm_half = lane >> 4;
    float acc[4][4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[m][n][j] = 0.0f; }
        }
    }
    // Fragments of one 32-value sub-block: A for the warp's 4 row tiles, B for its 4 token tiles.
    struct Frag {
        int a[4][4];
        int b[2][4];
    };
    const auto load_frag = [&](Frag& f, int i, int su) {
        const std::uint8_t* as = smem + L::kAOff + (i % kASlots) * L::kASlot + su * kAPlane;
        const std::uint8_t* q  = smem + L::kActOff + (i % kQSlots) * kActSlot;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            ldmatrix_x4(f.a[m], as + a_offset(wm * 64 + m * 16 + lm_row, lm_half));
        }
#pragma unroll
        for (int np = 0; np < 2; ++np) {
            const int j   = lane >> 3;
            const int tok = wn * 32 + (2 * np + (j >> 1)) * 8 + (lane & 7);
            ldmatrix_x4(f.b[np], q + act_offset(tok, 2 * su + (j & 1)));
        }
    };
    const auto compute = [&](const Frag& f, int i, int su) {
        const float* rsc = reinterpret_cast<const float*>(smem + L::kAOff +
                                                          (i % kASlots) * L::kASlot +
                                                          kSubs * kAPlane);
        const float* dsm = reinterpret_cast<const float*>(smem + L::kActOff +
                                                          (i % kQSlots) * kActSlot + kActSlotQ);
        if constexpr (FP8) {
            const auto* rsu = reinterpret_cast<const unsigned*>(rsc);
            unsigned sb[4];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                sb[n] = __float_as_uint(dsm[su * kBN + wn * 32 + n * 8 + g]) >> 23;
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const unsigned sa = rsu[su * kBM + wm * 64 + m * 16 + 2 * g + (t & 1)];
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    mma_e4m3_k32(acc[m][n], f.a[m], f.b[n >> 1][2 * (n & 1)],
                                 f.b[n >> 1][2 * (n & 1) + 1], sa, sb[n]);
                }
            }
        } else {
            float2 dv[4];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                dv[n] = *reinterpret_cast<const float2*>(dsm + su * kBN + wn * 32 + n * 8 + 2 * t);
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                // Rows g and g + 8 (scale_index).
                const int rs = su * kBM + wm * 64 + m * 16 + 2 * g;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int b0 = f.b[n >> 1][2 * (n & 1)], b1 = f.b[n >> 1][2 * (n & 1) + 1];
                    if constexpr (kS == 1) {
                        const float2 sr = *reinterpret_cast<const float2*>(rsc + rs);
                        int c[4];
                        mma_k32(c, f.a[m], b0, b1);
                        acc[m][n][0] = fmaf(float(c[0]), sr.x * dv[n].x, acc[m][n][0]);
                        acc[m][n][1] = fmaf(float(c[1]), sr.x * dv[n].y, acc[m][n][1]);
                        acc[m][n][2] = fmaf(float(c[2]), sr.y * dv[n].x, acc[m][n][2]);
                        acc[m][n][3] = fmaf(float(c[3]), sr.y * dv[n].y, acc[m][n][3]);
                    } else {
                        const float4 sr = *reinterpret_cast<const float4*>(rsc + rs * 2);
                        int c0[4], c1[4];
                        mma_k16(c0, f.a[m][0], f.a[m][1], b0);
                        mma_k16(c1, f.a[m][2], f.a[m][3], b1);
                        acc[m][n][0] = fmaf(fmaf(float(c0[0]), sr.x, float(c1[0]) * sr.y), dv[n].x,
                                            acc[m][n][0]);
                        acc[m][n][1] = fmaf(fmaf(float(c0[1]), sr.x, float(c1[1]) * sr.y), dv[n].y,
                                            acc[m][n][1]);
                        acc[m][n][2] = fmaf(fmaf(float(c0[2]), sr.z, float(c1[2]) * sr.w), dv[n].x,
                                            acc[m][n][2]);
                        acc[m][n][3] = fmaf(fmaf(float(c0[3]), sr.z, float(c1[3]) * sr.w), dv[n].y,
                                            acc[m][n][3]);
                    }
                }
            }
        }
    };
    Frag f0, f1;
    bar_sync(kFull, kThreads);
    if constexpr (kS == 2 && !FP8) {
        // Two-scale types run twice the MMAs per value; their fragments are not double-buffered
        // (registers).
        for (int i = 0; i < iters; ++i) {
            if (i > 0) { bar_sync(kFull + i % kASlots, kThreads); }
#pragma unroll
            for (int su = 0; su < kSubs; ++su) {
                load_frag(f0, i, su);
                compute(f0, i, su);
            }
            bar_arrive(kEmpty + i % kASlots, kThreads);
        }
    } else {
        // The next sub-block's fragments load while this one multiplies.
        load_frag(f0, 0, 0);
        for (int i = 0; i < iters; ++i) {
            load_frag(f1, i, 1);
            compute(f0, i, 0);
            if (i + 1 < iters) {
                bar_sync(kFull + (i + 1) % kASlots, kThreads);
                load_frag(f0, i + 1, 0);
            }
            compute(f1, i, 1);
            bar_arrive(kEmpty + i % kASlots, kThreads);
        }
    }
    (void)f1;
    // Per 16-row tile, every operand the epilogue reads (the residual, the gate) is loaded before
    // any store, so the loads overlap instead of each waiting behind the stores it might alias.
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        float operand[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wm * 64 + m * 16 + g + 8 * (j >> 1);
                const int c = tok0 + wn * 32 + n * 8 + 2 * t + (j & 1);
                operand[n][j] = r < p.rows && c < p.columns ? epi_operand<EPI>(p.out, r, c) : 0.0f;
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wm * 64 + m * 16 + g + 8 * (j >> 1);
                const int c = tok0 + wn * 32 + n * 8 + 2 * t + (j & 1);
                if (r < p.rows && c < p.columns) {
                    epi_finish<EPI>(p.out, r, c, acc[m][n][j], operand[n][j]);
                }
            }
        }
    }
#endif
}

template <ggml_type type, int KS, int EPI, bool FP8>
void launch_k(const TileArgs& args, cudaStream_t stream) {
    auto* kern = kernel<type, KS, EPI, FP8>;
    static const bool raised = [kern] {
        check(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Layout<type>::kBytes),
              "tile shared memory");
        return true;
    }();
    (void)raised;
    const unsigned row_tiles   = (args.rows + kBM - 1) / kBM;
    const unsigned token_tiles = (args.columns + kBN - 1) / kBN;
    const dim3 grid = kTokensFastest<type> ? dim3(token_tiles, row_tiles) : dim3(row_tiles, token_tiles);
    kern<<<grid, kThreads, Layout<type>::kBytes, stream>>>(args);
    check(cudaGetLastError(), "tile product launch");
}

// The two-pass MXFP8 product's first pass (ggml_bridge.cu lt_product): rows [0, rows) decoded
// with fp8_decode, the tile kernel's MXFP8 arithmetic, into an image of e4m3 [rows][k] and its
// UE8M0 exponents in cuBLASLt's layout (lt_scale_offset). A thread decodes one 32-value slice, a
// warp 1 KB of a row; slice lanes 4j..4j+3 hold one 4-byte word of exponents.
constexpr int kImageThreads = 256;

template <ggml_type type>
__launch_bounds__(kImageThreads) __global__ void image_kernel(ImageArgs p) {
#if defined(NINFER_SM120_FP8)
    using D               = vec::Decoder<type>;
    constexpr int kSlices = D::kBlockElems / 32;
    constexpr int kWords  = kFp8Lut<type> ? kLutWords<type> : D::kTableWords;
    static_assert(kSlices % 4 == 0, "image kernel: whole exponent words per block");
    __shared__ std::uint32_t table[kWords > 0 ? kWords : 1];
    if constexpr (kWords > 0) {
        vec::stage_words<kImageThreads, kWords>(table, [](int i) -> std::uint32_t {
            if constexpr (kFp8Lut<type>) {
                return lut_table_word<type>(D::table(), i);
            } else {
                return D::table()[i];
            }
        });
        __syncthreads();
    }
    const int blocks       = p.k / D::kBlockElems;
    const std::int64_t at  = std::int64_t(blockIdx.x) * kImageThreads + threadIdx.x;
    const int u            = int(at % kSlices);
    const std::int64_t row_block = at / kSlices;
    const int b            = int(row_block % blocks);
    const int row          = int(row_block / blocks);
    const bool valid       = row < p.rows;
    int v[8]               = {};
    unsigned e             = 0;
    if (valid) {
        e = fp8_decode<type>(p.weight + std::int64_t(row) * p.row_bytes +
                                 std::int64_t(b) * D::kBlockBytes,
                             u, table, v);
    }
    const unsigned e1 = __shfl_down_sync(0xFFFFFFFF, e, 1);
    const unsigned e2 = __shfl_down_sync(0xFFFFFFFF, e, 2);
    const unsigned e3 = __shfl_down_sync(0xFFFFFFFF, e, 3);
    if (!valid) { return; }
    auto* out = reinterpret_cast<int4*>(p.image + std::int64_t(row) * p.k +
                                        std::int64_t(b) * D::kBlockElems + 32 * u);
    out[0]    = make_int4(v[0], v[1], v[2], v[3]);
    out[1]    = make_int4(v[4], v[5], v[6], v[7]);
    if (u % 4 == 0) {
        *reinterpret_cast<unsigned*>(p.scales + lt_scale_offset(row, b * kSlices + u, p.k / 32)) =
            (e & 0xFF) | ((e1 & 0xFF) << 8) | ((e2 & 0xFF) << 16) | ((e3 & 0xFF) << 24);
    }
#else
    __trap(); // a 120a build only
#endif
}

// The NVFP4 image pass and the weight's max |w|: a thread decodes one 32-value slice with the
// vector decoder (ggml_bridge_vec.cuh) into 32 floats, two NVFP4 groups.
template <ggml_type type>
__device__ __forceinline__ void slice_values(const vec::Slice& s, float (&v)[32]) {
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const unsigned byte = (unsigned(s.w[i / 4]) >> (8 * (i % 4))) & 0xFFu;
        if constexpr (type == GGML_TYPE_Q2_K) {
            v[i] = i < 16 ? s.f[0] * float(byte) - s.f[2] : s.f[1] * float(byte) - s.f[3];
        } else if constexpr (type == GGML_TYPE_IQ1_S) {
            v[i] = s.f[0] * (float(byte) + s.f[1]);
        } else if constexpr (type == GGML_TYPE_IQ1_M) {
            v[i] = (i < 16 ? s.f[0] : s.f[1]) * (float(byte) + s.f[2 + i / 8]);
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            v[i] = s.f[0] * float(byte) - s.f[1];
        } else {
            const float c = float(static_cast<signed char>(byte));
            if constexpr (type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
                          type == GGML_TYPE_Q6_K) {
                v[i] = (i < 16 ? s.f[0] : s.f[1]) * c;
            } else {
                v[i] = s.f[0] * c;
            }
        }
    }
}

// Slice `at` of a [rows][k] GGUF matrix (row-major over the rows' 32-value slices) as 32 floats,
// its row and its first value's column; false past the last row.
template <ggml_type type>
__device__ __forceinline__ bool decode_slice(const std::uint8_t* weight, std::int64_t row_bytes,
                                             int rows, int k, std::int64_t at,
                                             const std::uint32_t* table, int& row,
                                             std::int64_t& first, float (&v)[32]) {
    using D                      = vec::Decoder<type>;
    constexpr int kSlices        = D::kBlockElems / 32;
    const int blocks             = k / D::kBlockElems;
    const int u                  = int(at % kSlices);
    const std::int64_t row_block = at / kSlices;
    const int b                  = int(row_block % blocks);
    row                          = int(row_block / blocks);
    first                        = std::int64_t(b) * D::kBlockElems + 32 * u;
    if (row >= rows) { return false; }
    vec::Slice s;
    D::decode(weight + std::int64_t(row) * row_bytes + std::int64_t(b) * D::kBlockBytes, u, table,
              s);
    slice_values<type>(s, v);
    return true;
}

// The decoder's grid table in shared memory (every thread of the block calls this first).
template <ggml_type type>
__device__ __forceinline__ const std::uint32_t* stage_table() {
    using D = vec::Decoder<type>;
    __shared__ std::uint32_t table[D::kTableWords > 0 ? D::kTableWords : 1];
    if constexpr (D::kTableWords > 0) {
        vec::stage_words<kImageThreads, D::kTableWords>(table,
                                                         [](int i) { return D::table()[i]; });
        __syncthreads();
    }
    return table;
}

template <ggml_type type>
__launch_bounds__(kImageThreads) __global__ void amax_kernel(AmaxArgs p) {
    const std::uint32_t* table = stage_table<type>();
    const std::int64_t at      = std::int64_t(blockIdx.x) * kImageThreads + threadIdx.x;
    float v[32];
    int row            = 0;
    std::int64_t first = 0;
    float m            = 0.0f;
    if (decode_slice<type>(p.weight, p.row_bytes, p.rows, p.k, at, table, row, first, v)) {
#pragma unroll
        for (int i = 0; i < 32; ++i) { m = fmaxf(m, fabsf(v[i])); }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
    }
    if ((threadIdx.x & 31) == 0) { atomicMax(p.amax, __float_as_uint(m)); }
}

// How the NVFP4 image pass takes each group's scale: Search runs fp4_encode16's candidate search,
// Pick encodes the candidate the choice table names (Fp4ImageArgs::choices), Choices runs the
// search and writes only the table.
enum Fp4ImageMode : int { kFp4Search, kFp4Pick, kFp4Choices };

// A block covers 256 slices, whose choices are its 128 table bytes: eight 16-byte words, which
// Pick loads before the decode (the table is mapped host memory; the bus latency then overlaps the
// decode) and Choices stores after the search.
template <ggml_type type, int Mode>
__launch_bounds__(kImageThreads) __global__ void image_fp4_kernel(Fp4ImageArgs p,
                                                                   std::uint8_t* table_out) {
#if defined(NINFER_SM120_FP8)
    constexpr int kWords = kImageThreads / 32;
    __shared__ uint4 block_choices[kWords];
    uint4 loaded = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (Mode == kFp4Pick) {
        if (threadIdx.x < kWords) {
            loaded = reinterpret_cast<const uint4*>(p.choices +
                                                    std::int64_t(blockIdx.x) * (kImageThreads / 2))
                [threadIdx.x];
        }
    }
    const std::uint32_t* table = stage_table<type>();
    const std::int64_t at      = std::int64_t(blockIdx.x) * kImageThreads + threadIdx.x;
    if constexpr (Mode != kFp4Choices) {
        if (at == 0) {
            p.alpha[0] = 1.0f / (p.global * lt_fp4_global(__uint_as_float(*p.x_amax)));
            p.alpha[1] = 0.0f;
            p.alpha[2] = 1.0f;
        }
    }
    float v[32];
    int row            = 0;
    std::int64_t first = 0;
    const bool live =
        decode_slice<type>(p.weight, p.row_bytes, p.rows, p.k, at, table, row, first, v);
    if constexpr (Mode == kFp4Pick) {
        if (threadIdx.x < kWords) { block_choices[threadIdx.x] = loaded; }
        __syncthreads();
    }
    unsigned nibble = 0;
    if (live) {
#pragma unroll
        for (int i = 0; i < 32; ++i) { v[i] *= p.global; }
        uint2 c0, c1;
        unsigned short s0, s1;
        if constexpr (Mode == kFp4Pick) {
            nibble = (unsigned(reinterpret_cast<const std::uint8_t*>(block_choices)[threadIdx.x / 2]) >>
                      (4 * (threadIdx.x % 2))) &
                     0xFu;
            s0 = fp4_encode16_pick(v, nibble & 3u, c0);
            s1 = fp4_encode16_pick(v + 16, nibble >> 2, c1);
        } else {
            unsigned w0 = 0, w1 = 0;
            s0     = fp4_encode16(v, c0, Mode == kFp4Choices ? &w0 : nullptr);
            s1     = fp4_encode16(v + 16, c1, Mode == kFp4Choices ? &w1 : nullptr);
            nibble = w0 | (w1 << 2);
        }
        if constexpr (Mode != kFp4Choices) {
            *reinterpret_cast<uint4*>(p.image + std::int64_t(row) * (p.k / 2) + first / 2) =
                make_uint4(c0.x, c0.y, c1.x, c1.y);
            *reinterpret_cast<unsigned short*>(p.scales +
                                               lt_scale_offset(row, int(first / 16), p.k / 16)) =
                static_cast<unsigned short>((s0 & 0xFFu) | ((s1 & 0xFFu) << 8));
        }
    }
    if constexpr (Mode == kFp4Choices) {
        // A slice past the rows writes 0.
        const unsigned next = __shfl_down_sync(0xFFFFFFFFu, nibble, 1);
        if (threadIdx.x % 2 == 0) {
            reinterpret_cast<std::uint8_t*>(block_choices)[threadIdx.x / 2] =
                static_cast<std::uint8_t>(nibble | (next << 4));
        }
        __syncthreads();
        if (threadIdx.x < kWords) {
            reinterpret_cast<uint4*>(table_out + std::int64_t(blockIdx.x) * (kImageThreads / 2))
                [threadIdx.x] = block_choices[threadIdx.x];
        }
    }
#else
    __trap(); // a 120a build only
#endif
}

// In the q8_1 form one-scale types run a loop bound by K (measured 1-3.5% faster than static K
// there) and two-scale ones take static-K forms at the 27B's widths (2-3% faster for them). The
// MXFP8 form runs the K-bound loop for every type.
template <ggml_type type, int EPI>
void launch_e(const TileArgs& args, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    if (args.mxfp8) { return launch_k<type, 0, EPI, true>(args, stream); }
#endif
    if constexpr (kFp8Only<type>) {
        throw std::invalid_argument("gguf tile product: this block type has only the MXFP8 form");
    } else {
        if constexpr (kScales<type> == 2 && EPI != kEpiGeneric) {
            switch (args.k) {
            case 5120: return launch_k<type, 5120, EPI, false>(args, stream);
            case 6144: return launch_k<type, 6144, EPI, false>(args, stream);
            case 17408: return launch_k<type, 17408, EPI, false>(args, stream);
            default: break;
            }
        }
        launch_k<type, 0, EPI, false>(args, stream);
    }
}

} // namespace tile

template <ggml_type type>
void tile_launch(const TileArgs& args, cudaStream_t stream) {
    switch (store_epi(args.out)) {
    case kEpiF32: return tile::launch_e<type, kEpiF32>(args, stream);
    case kEpiBf16: return tile::launch_e<type, kEpiBf16>(args, stream);
    case kEpiBf16Add: return tile::launch_e<type, kEpiBf16Add>(args, stream);
    case kEpiBf16Silu: return tile::launch_e<type, kEpiBf16Silu>(args, stream);
    default: return tile::launch_e<type, kEpiGeneric>(args, stream);
    }
}

template <ggml_type type>
void image_launch(const ImageArgs& args, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    const std::int64_t slices = std::int64_t(args.rows) * (args.k / 32);
    const auto grid =
        static_cast<unsigned>((slices + tile::kImageThreads - 1) / tile::kImageThreads);
    tile::image_kernel<type><<<grid, tile::kImageThreads, 0, stream>>>(args);
    check(cudaGetLastError(), "MXFP8 image launch");
#else
    (void)args;
    (void)stream;
    throw std::invalid_argument("gguf MXFP8 image: this build has no MXFP8 form");
#endif
}

template <ggml_type type>
void image_fp4_launch(const Fp4ImageArgs& args, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    const std::int64_t slices = std::int64_t(args.rows) * (args.k / 32);
    const auto grid =
        static_cast<unsigned>((slices + tile::kImageThreads - 1) / tile::kImageThreads);
    if (args.choices != nullptr) {
        tile::image_fp4_kernel<type, tile::kFp4Pick>
            <<<grid, tile::kImageThreads, 0, stream>>>(args, nullptr);
    } else {
        tile::image_fp4_kernel<type, tile::kFp4Search>
            <<<grid, tile::kImageThreads, 0, stream>>>(args, nullptr);
    }
    check(cudaGetLastError(), "NVFP4 image launch");
#else
    (void)args;
    (void)stream;
    throw std::invalid_argument("gguf NVFP4 image: this build has no NVFP4 form");
#endif
}

template <ggml_type type>
void fp4_choices_launch(const Fp4ImageArgs& args, std::uint8_t* table, cudaStream_t stream) {
#if defined(NINFER_SM120_FP8)
    const std::int64_t slices = std::int64_t(args.rows) * (args.k / 32);
    const auto grid =
        static_cast<unsigned>((slices + tile::kImageThreads - 1) / tile::kImageThreads);
    tile::image_fp4_kernel<type, tile::kFp4Choices>
        <<<grid, tile::kImageThreads, 0, stream>>>(args, table);
    check(cudaGetLastError(), "NVFP4 choice table launch");
#else
    (void)args;
    (void)table;
    (void)stream;
    throw std::invalid_argument("gguf NVFP4 choice table: this build has no NVFP4 form");
#endif
}

template <ggml_type type>
void amax_launch(const AmaxArgs& args, cudaStream_t stream) {
    const std::int64_t slices = std::int64_t(args.rows) * (args.k / 32);
    const auto grid =
        static_cast<unsigned>((slices + tile::kImageThreads - 1) / tile::kImageThreads);
    tile::amax_kernel<type><<<grid, tile::kImageThreads, 0, stream>>>(args);
    check(cudaGetLastError(), "GGUF amax launch");
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_TILE_INSTANCE(TYPE)                                                            \
    template void ninfer::ops::gguf::detail::tile_launch<TYPE>(                                    \
        const ninfer::ops::gguf::detail::TileArgs&, cudaStream_t);                                 \
    template void ninfer::ops::gguf::detail::image_launch<TYPE>(                                   \
        const ninfer::ops::gguf::detail::ImageArgs&, cudaStream_t);                                \
    NINFER_GGUF_FP4_INSTANCE(TYPE)

// The NVFP4 image and max |w| of a type, which the types without a tile kernel instantiate alone.
#define NINFER_GGUF_FP4_INSTANCE(TYPE)                                                             \
    template void ninfer::ops::gguf::detail::image_fp4_launch<TYPE>(                               \
        const ninfer::ops::gguf::detail::Fp4ImageArgs&, cudaStream_t);                             \
    template void ninfer::ops::gguf::detail::fp4_choices_launch<TYPE>(                             \
        const ninfer::ops::gguf::detail::Fp4ImageArgs&, std::uint8_t*, cudaStream_t);              \
    template void ninfer::ops::gguf::detail::amax_launch<TYPE>(                                    \
        const ninfer::ops::gguf::detail::AmaxArgs&, cudaStream_t)
