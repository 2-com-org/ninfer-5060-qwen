#pragma once

// The narrow kernel: products of a few columns -- decode verification of several sequences or
// drafts, short prompt pieces -- of IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S and Q2_K on integer
// tensor cores (sm_120). The vector kernel's dp4a work grows with the columns past four, and the
// matrix kernel's 128-row tiles waste most of their work below 32 columns; here a weight row is read
// and decoded once per 16 rows x 32 values and multiplied against every column with
// mma.m16n8k32, NT = ceil(columns / 8) n-tiles of 8 columns.
//
// A CTA of TW x KS warps owns units of 16 TW rows (units blockIdx.x, + gridDim.x, ...) and walks
// their K in steps of CB blocks as one cp.async pipeline of S stages: each stage holds the 16-byte
// aligned superset of every row's CB blocks, the step's activation values and its scales. Warp
// (tw, ks) owns rows [16 tw, 16 tw + 16) of the unit and slices [ks NS, ks NS + NS) of the step
// (NS = 8 CB / KS: half or all of one block); the KS partial sums meet in shared memory after a
// unit's last step, in a fixed order, and go through the product's epilogue (vec_store).
//
// Fragments: lane (g, t) = (lane / 4, lane % 4) holds A rows g and g + 8 at values [8t, 8t + 8) of a
// 32-value slice (two words) and the B fragment of column 8j + g at the same values. A slice's
// integer dot is exact in int32 and scaled by the weight's slice factor and the column's d in FP32,
// as the vector kernel scales its dp4a sums. Per-16 scales (IQ2_XS, IQ2_S) take two k32 MMAs over
// lane-masked A (lanes t < 2 hold a slice's first 16 values) combined as integers with their 2s + 1
// scales. Q2_K folds its 4-bit scale into A (q sc <= 45) and its minimum into a second MMA
// (A = min replicated, against the quantized activation as the vector kernel's dp4a takes it).
// Grids are a shared copy of ggml's unsigned tables (8-byte entries for IQ2) signed in the ALU.
//
// The activation (ggml_bridge.cu quantize_narrow_activation) is the vector kernel's q8_1 -- d =
// max|x| / 127 per 32 values, each value x / d rounded, d rounded to FP16 -- in this kernel's
// order: per column, per pair p of slices, lane t's 8 values of slice 2p and then of slice 2p + 1
// (a lane's B fragments of both slices are one LDS.128), and d, held as FP32, at [p][column,
// padded to 8 NT][slice parity] (a lane's two columns of both slices one LDS.128). Columns past
// the product's read its last column's values (their scales are zero, their outputs not stored).
//
// The kernel is issue-bound: a thread's cp.async copies are planned once, as 32-bit offsets from
// the issuing unit's start, rather than derived per step.

#include "ggml_bridge_internal.cuh"

#include <atomic>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::gguf::detail {

namespace narrow {

enum class Kind { K32, Split16, Q2K };

__device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%10,%10,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

// 16 bytes to shared memory, of which the first `bytes` are read and the rest zeroed.
__device__ __forceinline__ void copy16(void* smem, const void* gmem, unsigned bytes) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void copy_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void copy_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ std::uint32_t prmt(std::uint32_t a, std::uint32_t b, std::uint32_t sel) {
    std::uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ std::uint32_t load_u16(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint16_t*>(p);
}
__device__ __forceinline__ float load_half(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(load_u16(p))));
}
// N little-endian words from a 2-byte aligned address in N + 1 aligned loads. The aligned pointer
// derives from p itself, so a shared p keeps its address space.
template <int N>
__device__ __forceinline__ void load_words(const std::uint8_t* p, std::uint32_t (&w)[N]) {
    const unsigned mis = static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(p)) & 3u;
    const auto* q      = reinterpret_cast<const std::uint32_t*>(p - mis);
    std::uint32_t x[N + 1];
#pragma unroll
    for (int i = 0; i <= N; ++i) { x[i] = q[i]; }
#pragma unroll
    for (int i = 0; i < N; ++i) { w[i] = __funnelshift_r(x[i], x[i + 1], 8 * mis); }
}
// ggml's 7-bit sign code with its parity bit (ksigns_iq2xs).
__device__ __forceinline__ std::uint32_t signs8(std::uint32_t code) {
    return code | ((__popc(code) & 1u) << 7);
}
// The bytes of g (each in 1..127) negated where nib has a bit: (g - m) ^ (m * 255) = ~(g - 1).
__device__ __forceinline__ int apply_signs(std::uint32_t g, std::uint32_t nib) {
    const std::uint32_t m1 = (nib * 0x00204081u) & 0x01010101u;
    return int((g - m1) ^ (m1 * 0xFFu));
}

// One slice of rows g or g + 8 in lane t's A layout.
struct Frag {
    int lo, hi;   // values [8t, 8t + 4), [8t + 4, 8t + 8)
    float f;      // the slice's factor (K32), the block's (Split16), d (Q2K)
    int slo, shi; // Split16: the halves' 2s + 1
    int m;        // Q2K: the minimum, replicated
    float fm;     // Q2K: dmin
};

// A type's decoder: Head reads what a group of four slices [g0, g0 + 4) of a block shares,
// decode<U> slice g0 + U.
template <ggml_type type>
struct Decoder;

template <>
struct Decoder<GGML_TYPE_IQ3_S> {
    static constexpr int kBlockBytes = sizeof(block_iq3_s);
    static constexpr Kind kKind      = Kind::K32;
    static constexpr int kLutWords   = 512;
    static __device__ __forceinline__ const std::uint32_t* lut() { return iq3s_grid; }
    struct Head {
        float d, d2;
        std::uint32_t sc, ha, hb;
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int t) {
        Head h;
        h.d  = load_half(b);
        h.d2 = 2.f * h.d;
        h.sc = load_u16(b + 106 + g0 / 2);
        std::uint32_t qh[1];
        load_words<1>(b + 66 + g0, qh);
        h.ha = (qh[0] >> (2 * t)) & 0x01010101u;
        h.hb = (qh[0] >> (2 * t + 1)) & 0x01010101u;
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t* b, const Head& h, int g0,
                                                  int t, const std::uint32_t* lut, Frag& fr) {
        const std::uint32_t q  = load_u16(b + 2 + 8 * (g0 + U) + 2 * t);
        const std::uint32_t s  = b[74 + 4 * (g0 + U) + t];
        const std::uint32_t i0 = prmt(q, h.ha, 0xCC40 + (U << 4));
        const std::uint32_t i1 = prmt(q, h.hb, 0xCC41 + (U << 4));
        fr.lo = apply_signs(lut[i0], s & 0xFu);
        fr.hi = apply_signs(lut[i1], s >> 4);
        fr.f  = fmaf(h.d2, float((h.sc >> (4 * U)) & 0xFu), h.d);
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_XXS> {
    static constexpr int kBlockBytes = sizeof(block_iq3_xxs);
    static constexpr Kind kKind    = Kind::K32;
    static constexpr int kLutWords = 256;
    static __device__ __forceinline__ const std::uint32_t* lut() { return iq3xxs_grid; }
    struct Head {
        float d4, d2;
        std::uint32_t aux[4];
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int) {
        Head h;
        const float d = load_half(b);
        h.d4 = 0.25f * d;
        h.d2 = 0.5f * d;
        load_words<4>(b + 66 + 4 * g0, h.aux);
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t* b, const Head& h, int g0,
                                                  int t, const std::uint32_t* lut, Frag& fr) {
        const std::uint32_t q  = load_u16(b + 2 + 8 * (g0 + U) + 2 * t);
        const std::uint32_t s8 = signs8((h.aux[U] >> (7 * t)) & 127u);
        fr.lo = apply_signs(lut[q & 0xFFu], s8 & 0xFu);
        fr.hi = apply_signs(lut[q >> 8], s8 >> 4);
        fr.f  = fmaf(h.d2, float(h.aux[U] >> 28), h.d4);
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XXS> {
    static constexpr int kBlockBytes = sizeof(block_iq2_xxs);
    static constexpr Kind kKind    = Kind::K32;
    static constexpr int kLutWords = 512; // 256 8-byte entries
    static __device__ __forceinline__ const std::uint32_t* lut() {
        return reinterpret_cast<const std::uint32_t*>(iq2xxs_grid);
    }
    struct Head {
        float d8, d4;
        std::uint32_t w[8]; // per slice: its index word, its sign/scale word
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int) {
        Head h;
        const float d = load_half(b);
        h.d8 = 0.125f * d;
        h.d4 = 0.25f * d;
        load_words<8>(b + 2 + 8 * g0, h.w);
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t*, const Head& h, int, int t,
                                                  const std::uint32_t* lut, Frag& fr) {
        const std::uint32_t idx = (h.w[2 * U] >> (8 * t)) & 0xFFu;
        const std::uint32_t s8  = signs8((h.w[2 * U + 1] >> (7 * t)) & 127u);
        const uint2 g           = reinterpret_cast<const uint2*>(lut)[idx];
        fr.lo = apply_signs(g.x, s8 & 0xFu);
        fr.hi = apply_signs(g.y, s8 >> 4);
        fr.f  = fmaf(h.d4, float(h.w[2 * U + 1] >> 28), h.d8);
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XS> {
    static constexpr int kBlockBytes = sizeof(block_iq2_xs);
    static constexpr Kind kKind    = Kind::Split16;
    static constexpr int kLutWords = 1024; // 512 8-byte entries
    static __device__ __forceinline__ const std::uint32_t* lut() {
        return reinterpret_cast<const std::uint32_t*>(iq2xs_grid);
    }
    struct Head {
        float d8;
        std::uint32_t sc;
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int) {
        Head h;
        h.d8 = 0.125f * load_half(b);
        std::uint32_t sc[1];
        load_words<1>(b + 66 + g0, sc);
        h.sc = sc[0];
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t* b, const Head& h, int g0,
                                                  int t, const std::uint32_t* lut, Frag& fr) {
        const std::uint32_t c  = load_u16(b + 2 + 8 * (g0 + U) + 2 * t);
        const std::uint32_t s8 = signs8(c >> 9);
        const uint2 g          = reinterpret_cast<const uint2*>(lut)[c & 511u];
        fr.lo = apply_signs(g.x, s8 & 0xFu);
        fr.hi = apply_signs(g.y, s8 >> 4);
        const std::uint32_t sb = (h.sc >> (8 * U)) & 0xFFu;
        fr.slo = int(((sb & 0xFu) << 1) | 1u);
        fr.shi = int(((sb >> 3) & 0x1Eu) | 1u);
        fr.f   = h.d8;
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_S> {
    static constexpr int kBlockBytes = sizeof(block_iq2_s);
    static constexpr Kind kKind    = Kind::Split16;
    static constexpr int kLutWords = 2048; // 1024 8-byte entries
    static __device__ __forceinline__ const std::uint32_t* lut() {
        return reinterpret_cast<const std::uint32_t*>(iq2s_grid);
    }
    struct Head {
        float d8;
        std::uint32_t qh, sc;
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int) {
        Head h;
        h.d8 = 0.125f * load_half(b);
        std::uint32_t qh[1], sc[1];
        load_words<1>(b + 66 + g0, qh);
        load_words<1>(b + 74 + g0, sc);
        h.qh = qh[0];
        h.sc = sc[0];
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t* b, const Head& h, int g0,
                                                  int t, const std::uint32_t* lut, Frag& fr) {
        const int u              = g0 + U;
        const std::uint32_t qhb  = (h.qh >> (8 * U)) & 0xFFu;
        const std::uint32_t idx  = b[2 + 4 * u + t] | (((qhb >> (2 * t)) & 3u) << 8);
        const std::uint32_t sg   = b[34 + 4 * u + t];
        const uint2 g            = reinterpret_cast<const uint2*>(lut)[idx];
        fr.lo = apply_signs(g.x, sg & 0xFu);
        fr.hi = apply_signs(g.y, sg >> 4);
        const std::uint32_t sb = (h.sc >> (8 * U)) & 0xFFu;
        fr.slo = int(((sb & 0xFu) << 1) | 1u);
        fr.shi = int(((sb >> 3) & 0x1Eu) | 1u);
        fr.f   = h.d8;
    }
};

template <>
struct Decoder<GGML_TYPE_Q2_K> {
    static constexpr int kBlockBytes = sizeof(block_q2_K);
    static constexpr Kind kKind    = Kind::Q2K;
    static constexpr int kLutWords = 0;
    static __device__ __forceinline__ const std::uint32_t* lut() { return nullptr; }
    struct Head {
        float d, dm;
        std::uint32_t sc[2]; // the scale/min bytes of the group's 8 16-value sub-blocks
        std::uint32_t q[2];  // the group's 128-value half: bytes [8t, 8t + 8)
    };
    static __device__ __forceinline__ Head head(const std::uint8_t* b, int g0, int t) {
        Head h;
        h.d  = load_half(b + 80);
        h.dm = load_half(b + 82);
        load_words<2>(b + 2 * g0, h.sc);
        load_words<2>(b + 16 + 32 * (g0 / 4) + 8 * t, h.q);
        return h;
    }
    template <int U>
    static __device__ __forceinline__ void decode(const std::uint8_t*, const Head& h, int, int t,
                                                  const std::uint32_t*, Frag& fr) {
        const std::uint32_t sb = (h.sc[U / 2] >> (16 * (U & 1) + 8 * (t >> 1))) & 0xFFu;
        const std::uint32_t s  = sb & 0xFu;
        fr.lo = int(((h.q[0] >> (2 * U)) & 0x03030303u) * s);
        fr.hi = int(((h.q[1] >> (2 * U)) & 0x03030303u) * s);
        fr.m  = int((sb >> 4) * 0x01010101u);
        fr.f  = h.d;
        fr.fm = h.dm;
    }
};

template <ggml_type type, int NT, int TW, int KS, int S, int CB>
struct Shape {
    static constexpr int kBlockBytes  = Decoder<type>::kBlockBytes;
    static constexpr int kChunkValues = 256 * CB;
    static constexpr int NS           = 8 * CB / KS; // slices of a warp per step
    static constexpr int TP           = 8 * NT;      // scale columns, padded to the n-tiles
    static constexpr int kThreads     = TW * KS * 32;
    static constexpr int kChunkBytes  = CB * kBlockBytes;
    // Blocks start 2-byte aligned: the 16-byte aligned superset of a row's step.
    static constexpr int kRowCopies   = (kChunkBytes + 14 + 15) / 16;
    static constexpr int kRowStride   = 16 * (kRowCopies | 1);
    static constexpr int kWeightBytes = TW * 16 * kRowStride;
    // kChunkValues + 64: a quarter warp's LDS.128 (2 columns x 4 lanes) hits all 32 banks.
    static constexpr int kActStride    = kChunkValues + 64;
    static constexpr int kActCopies    = kChunkValues / 16; // a column's per step
    static constexpr int kScaleBytes   = kChunkValues / 64 * TP * 8;
    static constexpr int kScaleCopies  = kScaleBytes / 16;
    static constexpr int kWeightCopies = TW * 16 * kRowCopies;
    static constexpr int kPartialBytes = (KS - 1) * TW * 32 * NT * 16;
    static_assert(NS == 8 || NS == 4, "a warp owns 4 or 8 slices of one block");
    static_assert(kBlockBytes % 2 == 0, "blocks must start 2-byte aligned");
    [[nodiscard]] __host__ __device__ static constexpr int stage_bytes(int columns) {
        return kWeightBytes + columns * kActStride + kScaleBytes;
    }
    [[nodiscard]] static constexpr int bytes(int columns) {
        return S * stage_bytes(columns) + kPartialBytes;
    }
};

template <ggml_type type, int NT, int TW, int KS, int S, int CB>
__global__ void __launch_bounds__(TW * KS * 32) kernel(const NarrowArgs p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using D  = Decoder<type>;
    using Sh = Shape<type, NT, TW, KS, S, CB>;
    constexpr int NS = Sh::NS, TP = Sh::TP, kThreads = Sh::kThreads, kBB = Sh::kBlockBytes;
    constexpr int kLut = D::kLutWords > 0 ? D::kLutWords : 4;
    __shared__ __align__(16) std::uint32_t lut[kLut];
    extern __shared__ __align__(16) std::uint8_t ring[];
    const int T     = p.columns;
    const int rows  = p.rows;
    const int stage = Sh::stage_bytes(T);
    float* partial  = reinterpret_cast<float*>(ring + S * stage);
    if constexpr (D::kLutWords > 0) {
        for (int i = threadIdx.x; i < kLut / 4; i += kThreads) { copy16(lut + 4 * i, D::lut() + 4 * i, 16); }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tw = warp % TW, ks = warp / TW;
    const int g = lane >> 2, t = lane & 3;
    const int blk = (ks * NS) >> 3, u0 = (ks * NS) & 7;
    // Row r's step c starts at byte base + r * row_bytes + c * kChunkBytes of the aligned `wa`; a
    // copy reads no byte at or past `end`.
    const std::int64_t base = std::int64_t(reinterpret_cast<std::uintptr_t>(p.weight) & 15);
    const std::uint8_t* wa  = p.weight - base;
    const std::int64_t end  = base + std::int64_t(rows) * p.row_bytes;
    const int steps_per_unit = p.k / Sh::kChunkValues;
    const int units          = (rows + 16 * TW - 1) / (16 * TW);
    const int my_units =
        int(blockIdx.x) < units ? (units - 1 - int(blockIdx.x)) / int(gridDim.x) + 1 : 0;
    const int steps = my_units * steps_per_unit;

    // A thread's copies are the same every step: copy n of the weights (thread + n kThreads of the
    // stage's) is row w_row[n]'s 16 bytes at w_src[n] of the step's aligned superset, to w_dst[n];
    // of the activation, a_src[n] to a_dst[n] (-1: none); scale copy n at s_src[n] (-1: none).
    constexpr int kWC = (Sh::kWeightCopies + kThreads - 1) / kThreads;
    constexpr int kAC = (8 * NT * Sh::kActCopies + kThreads - 1) / kThreads;
    constexpr int kSC = (Sh::kScaleCopies + kThreads - 1) / kThreads;
    const int tid     = threadIdx.x;
    int w_row[kWC], w_dst[kWC], w_src[kWC], a_dst[kAC], a_src[kAC], s_src[kSC];
#pragma unroll
    for (int n = 0; n < kWC; ++n) {
        const int i = tid + n * kThreads;
        const int r = i / Sh::kRowCopies, c = i - r * Sh::kRowCopies;
        w_row[n]    = i < Sh::kWeightCopies ? r : -1;
        w_dst[n]    = r * Sh::kRowStride + 16 * c;
        w_src[n]    = 16 * c;
    }
#pragma unroll
    for (int n = 0; n < kAC; ++n) {
        const int i = tid + n * kThreads;
        const int col = i / Sh::kActCopies, c = i - col * Sh::kActCopies;
        a_dst[n] = i < T * Sh::kActCopies ? Sh::kWeightBytes + col * Sh::kActStride + 16 * c : -1;
        a_src[n] = col * p.k + 16 * c;
    }
#pragma unroll
    for (int n = 0; n < kSC; ++n) {
        const int i = tid + n * kThreads;
        s_src[n]    = i < Sh::kScaleCopies ? 16 * i : -1;
    }
    const int s_dst = Sh::kWeightBytes + T * Sh::kActStride;
    const auto* scales = reinterpret_cast<const std::uint8_t*>(p.d);
    // The issuing unit's first row from `w_unit`, its start rounded down to 16 bytes: copy n's row
    // starts at w_at[n], and `end` at w_end (at most INT_MAX).
    const std::uint8_t* w_unit = wa;
    int w_at[kWC];
    int w_end = 0;
    const auto plan_unit = [&](int u) {
        if (u >= units) { return; }
        const int first          = u * 16 * TW;
        const std::int64_t start = base + std::int64_t(first) * p.row_bytes;
        const std::int64_t down  = start & ~std::int64_t(15);
        w_unit = wa + down;
        w_end  = end - down < INT_MAX ? int(end - down) : INT_MAX;
#pragma unroll
        for (int n = 0; n < kWC; ++n) {
            w_at[n] = int(start - down) + (min(first + max(w_row[n], 0), rows - 1) - first) * int(p.row_bytes);
        }
    };
    int issue_unit = blockIdx.x, issue_step = 0, issue_slot = 0, issued = 0;
    int at_w = 0, at_a = 0, at_s = 0; // the issuing step's offsets
    plan_unit(issue_unit);
    const auto issue = [&]() {
        if (issued < steps) {
            std::uint8_t* st = ring + issue_slot * stage;
#pragma unroll
            for (int n = 0; n < kWC; ++n) {
                if (w_row[n] >= 0) {
                    const int at = ((w_at[n] + at_w) & ~15) + w_src[n];
                    copy16(st + w_dst[n], w_unit + at, unsigned(min(16, max(0, w_end - at))));
                }
            }
#pragma unroll
            for (int n = 0; n < kAC; ++n) {
                if (a_dst[n] >= 0) { copy16(st + a_dst[n], p.qs + a_src[n] + at_a, 16); }
            }
#pragma unroll
            for (int n = 0; n < kSC; ++n) {
                if (s_src[n] >= 0) { copy16(st + s_dst + s_src[n], scales + at_s + s_src[n], 16); }
            }
            ++issued;
            at_w += Sh::kChunkBytes;
            at_a += Sh::kChunkValues;
            at_s += Sh::kScaleBytes;
            if (++issue_step == steps_per_unit) {
                issue_step = 0;
                at_w = at_a = at_s = 0;
                issue_unit += gridDim.x;
                plan_unit(issue_unit);
            }
            if (++issue_slot == S) { issue_slot = 0; }
        }
        copy_commit();
    };
#pragma unroll
    for (int i = 0; i < S - 1; ++i) { issue(); }

    float acc[NT][4] = {};
    int aoff[NT]; // column 8j + g's activation, clamped to the last column
#pragma unroll
    for (int j = 0; j < NT; ++j) { aoff[j] = min(8 * j + g, T - 1) * Sh::kActStride; }
    int unit = blockIdx.x, step_in_unit = 0, slot = 0;
    for (int step = 0; step < steps; ++step) {
        issue();
        copy_wait<S - 1>();
        __syncthreads();
        const int ra = unit * 16 * TW + tw * 16 + g;
        const int rb = ra + 8;
        const std::uint8_t* st = ring + slot * stage;
        // A row's step starts at its byte offset modulo 16 in the stage's aligned superset.
        const unsigned at_step = unsigned(base) + unsigned(step_in_unit) * Sh::kChunkBytes;
        const int ma = int((at_step + unsigned(min(ra, rows - 1)) * unsigned(p.row_bytes)) & 15u);
        const int mb = int((at_step + unsigned(min(rb, rows - 1)) * unsigned(p.row_bytes)) & 15u);
        const std::uint8_t* pa = st + (tw * 16 + g) * Sh::kRowStride + ma + blk * kBB;
        const std::uint8_t* pb = st + (tw * 16 + g + 8) * Sh::kRowStride + mb + blk * kBB;
        const std::uint8_t* act0 = st + Sh::kWeightBytes + 16 * t + 32 * ks * NS;
        const std::uint8_t* ds0  = st + s_dst + 8 * (ks * NS / 2) * TP;
        const auto group = [&](auto gc) {
            constexpr int G = decltype(gc)::value;
            const int g0    = u0 + 4 * G;
            const auto ha   = D::head(pa, g0, t);
            const auto hb   = D::head(pb, g0, t);
            const auto pair = [&](auto pc) {
                constexpr int P = decltype(pc)::value; // slices 2P, 2P + 1 of the group
                Frag fa[2], fb[2];
                D::template decode<2 * P>(pa, ha, g0, t, lut, fa[0]);
                D::template decode<2 * P>(pb, hb, g0, t, lut, fb[0]);
                D::template decode<2 * P + 1>(pa, ha, g0, t, lut, fa[1]);
                D::template decode<2 * P + 1>(pb, hb, g0, t, lut, fb[1]);
                constexpr int PP = 2 * G + P; // the pair among the warp's slices
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int c0    = 8 * j + 2 * t;
                    const uint4 v   = *reinterpret_cast<const uint4*>(act0 + aoff[j] + 64 * PP);
                    const float4 dv = *reinterpret_cast<const float4*>(ds0 + 8 * (PP * TP + c0));
                    // (c0, slice 2P), (c0, 2P + 1); (c1, 2P), (c1, 2P + 1)
                    const float2 d0 = make_float2(dv.x, dv.y);
                    const float2 d1 = make_float2(dv.z, dv.w);
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int b0 = int(q == 0 ? v.x : v.z), b1 = int(q == 0 ? v.y : v.w);
                        const float e0 = q == 0 ? d0.x : d0.y, e1 = q == 0 ? d1.x : d1.y;
                        const Frag& xa = fa[q];
                        const Frag& xb = fb[q];
                        if constexpr (D::kKind == Kind::K32) {
                            const int a[4] = {xa.lo, xb.lo, xa.hi, xb.hi};
                            int c[4];
                            mma_k32(c, a, b0, b1);
                            acc[j][0] = fmaf(float(c[0]) * e0, xa.f, acc[j][0]);
                            acc[j][1] = fmaf(float(c[1]) * e1, xa.f, acc[j][1]);
                            acc[j][2] = fmaf(float(c[2]) * e0, xb.f, acc[j][2]);
                            acc[j][3] = fmaf(float(c[3]) * e1, xb.f, acc[j][3]);
                        } else if constexpr (D::kKind == Kind::Split16) {
                            const bool first = t < 2;
                            const int al[4]  = {first ? xa.lo : 0, first ? xb.lo : 0,
                                                first ? xa.hi : 0, first ? xb.hi : 0};
                            const int ah[4]  = {first ? 0 : xa.lo, first ? 0 : xb.lo,
                                                first ? 0 : xa.hi, first ? 0 : xb.hi};
                            int cl[4], ch[4];
                            mma_k32(cl, al, b0, b1);
                            mma_k32(ch, ah, b0, b1);
                            acc[j][0] = fmaf(float(cl[0] * xa.slo + ch[0] * xa.shi) * e0, xa.f, acc[j][0]);
                            acc[j][1] = fmaf(float(cl[1] * xa.slo + ch[1] * xa.shi) * e1, xa.f, acc[j][1]);
                            acc[j][2] = fmaf(float(cl[2] * xb.slo + ch[2] * xb.shi) * e0, xb.f, acc[j][2]);
                            acc[j][3] = fmaf(float(cl[3] * xb.slo + ch[3] * xb.shi) * e1, xb.f, acc[j][3]);
                        } else {
                            const int aq[4] = {xa.lo, xb.lo, xa.hi, xb.hi};
                            const int am[4] = {xa.m, xb.m, xa.m, xb.m};
                            int cq[4], cm[4];
                            mma_k32(cq, aq, b0, b1);
                            mma_k32(cm, am, b0, b1);
                            acc[j][0] = fmaf(fmaf(float(cq[0]), xa.f, -float(cm[0]) * xa.fm), e0, acc[j][0]);
                            acc[j][1] = fmaf(fmaf(float(cq[1]), xa.f, -float(cm[1]) * xa.fm), e1, acc[j][1]);
                            acc[j][2] = fmaf(fmaf(float(cq[2]), xb.f, -float(cm[2]) * xb.fm), e0, acc[j][2]);
                            acc[j][3] = fmaf(fmaf(float(cq[3]), xb.f, -float(cm[3]) * xb.fm), e1, acc[j][3]);
                        }
                    }
                }
            };
            pair(std::integral_constant<int, 0>{});
            pair(std::integral_constant<int, 1>{});
        };
        group(std::integral_constant<int, 0>{});
        if constexpr (NS == 8) { group(std::integral_constant<int, 1>{}); }
        if (step_in_unit == steps_per_unit - 1) {
            if constexpr (KS > 1) {
                if (ks > 0) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        *reinterpret_cast<float4*>(
                            partial + ((((ks - 1) * TW + tw) * 32 + lane) * NT + j) * 4) =
                            make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
                    }
                }
                __syncthreads();
                if (ks == 0) {
                    for (int s = 0; s < KS - 1; ++s) {
#pragma unroll
                        for (int j = 0; j < NT; ++j) {
                            const float4 q = *reinterpret_cast<const float4*>(
                                partial + (((s * TW + tw) * 32 + lane) * NT + j) * 4);
                            acc[j][0] += q.x;
                            acc[j][1] += q.y;
                            acc[j][2] += q.z;
                            acc[j][3] += q.w;
                        }
                    }
                }
            }
            if (ks == 0) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int c0 = 8 * j + 2 * t, c1 = c0 + 1;
                    if (ra < rows && c0 < T) { vec_store(p.out, ra, c0, acc[j][0]); }
                    if (ra < rows && c1 < T) { vec_store(p.out, ra, c1, acc[j][1]); }
                    if (rb < rows && c0 < T) { vec_store(p.out, rb, c0, acc[j][2]); }
                    if (rb < rows && c1 < T) { vec_store(p.out, rb, c1, acc[j][3]); }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[j][i] = 0.f; }
            }
        }
        __syncthreads();
        if (++step_in_unit == steps_per_unit) {
            step_in_unit = 0;
            unit += gridDim.x;
        }
        if (++slot == S) { slot = 0; }
    }
    copy_wait<0>();
#else
    (void)p;
#endif
}

// A type's configuration at NT n-tiles and TW warps of rows, TW * 1000 + KS * 100 + S * 10 + CB: of
// 14 measured on the RTX 5060 Ti over the 27B's shapes (rows 1024-34816, K 5120-17408) at 6-32
// columns, per (NT, TW) the one of least count-weighted time over the shapes row_warps gives that
// TW, of those whose CTA fits at 8 NT columns (count-weighted, 2.2% over each shape's fastest
// configuration); a (NT, TW) no 27B shape reaches (IQ2_XS and Q2_K at TW 2) the TW's least over all
// the shapes. IQ3_S keeps 2424 at one n-tile and TW 2 (its 1024 x 5120 product at 5 columns: 11.0
// us against 2422's 13.1). 512-value steps (CB 2) leave the activation stages of 17-32 columns
// room for two CTAs per SM, or for a second stage.
struct Configs {
    int c[4][3]; // [NT - 1][TW - 2]
};
template <ggml_type type>
inline constexpr Configs kConfigs{};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_IQ3_S>{
    {{2424, 3424, 4232}, {2424, 3424, 4222}, {2422, 3432, 4232}, {2432, 3422, 4222}}};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_IQ3_XXS>{
    {{2432, 3424, 4424}, {2422, 3424, 4232}, {2222, 3422, 4232}, {2222, 3422, 4222}}};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_IQ2_XXS>{
    {{2432, 3434, 4222}, {2424, 3424, 4424}, {2424, 3432, 4232}, {2422, 3422, 4222}}};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_IQ2_XS>{
    {{2424, 3422, 4222}, {2222, 3424, 4424}, {2222, 3432, 4222}, {2432, 3422, 4422}}};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_IQ2_S>{
    {{2824, 3434, 4424}, {2824, 3424, 4232}, {2424, 3422, 4232}, {2422, 3422, 4222}}};
template <>
inline constexpr Configs kConfigs<GGML_TYPE_Q2_K>{
    {{2222, 3434, 4222}, {2222, 3422, 4222}, {2222, 3422, 4232}, {2432, 3422, 4222}}};

template <ggml_type type, int NT, int TW, int KS, int S, int CB>
void launch_k(const NarrowArgs& args, cudaStream_t stream) {
    using Sh   = Shape<type, NT, TW, KS, S, CB>;
    auto* kern = kernel<type, NT, TW, KS, S, CB>;
    static const bool raised = [kern] {
        check(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   Sh::bytes(8 * NT)),
              "narrow shared memory");
        return true;
    }();
    (void)raised;
    // CTAs resident per SM at each column count of this NT (0: not yet measured).
    static std::atomic<int> resident[8] = {};
    const int bytes = Sh::bytes(args.columns);
    std::atomic<int>& slot = resident[args.columns - (8 * NT - 7)];
    int per_sm = slot.load(std::memory_order_relaxed);
    if (per_sm == 0) {
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, Sh::kThreads, bytes),
              "narrow occupancy");
        if (per_sm == 0) { throw std::runtime_error("gguf narrow product: a CTA does not fit an SM"); }
        slot.store(per_sm, std::memory_order_relaxed);
    }
    const int units  = (args.rows + 16 * TW - 1) / (16 * TW);
    const int blocks = std::min(per_sm * device_facts().sm_count, units);
    kern<<<blocks, Sh::kThreads, bytes, stream>>>(args);
    check(cudaGetLastError(), "narrow product launch");
}

template <ggml_type type, int NT, int TW>
void launch_tw(const NarrowArgs& args, cudaStream_t stream) {
    constexpr int c = kConfigs<type>.c[NT - 1][TW - 2];
    static_assert(c / 1000 == TW, "a configuration has its TW");
    launch_k<type, NT, TW, c / 100 % 10, c / 10 % 10, c % 10>(args, stream);
}

template <ggml_type type, int NT>
void launch_nt(const NarrowArgs& args, cudaStream_t stream) {
    switch (narrow_row_warps(Decoder<type>::kBlockBytes, args.rows, args.columns,
                             device_facts().sm_count)) {
    case 2: return launch_tw<type, NT, 2>(args, stream);
    case 3: return launch_tw<type, NT, 3>(args, stream);
    default: return launch_tw<type, NT, 4>(args, stream);
    }
}

} // namespace narrow

template <ggml_type type>
void narrow_launch(const NarrowArgs& args, cudaStream_t stream) {
    switch ((args.columns + 7) / 8) {
    case 1: return narrow::launch_nt<type, 1>(args, stream);
    case 2: return narrow::launch_nt<type, 2>(args, stream);
    case 3: return narrow::launch_nt<type, 3>(args, stream);
    case 4: return narrow::launch_nt<type, 4>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf narrow product: 1 to 32 columns");
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_NARROW_INSTANCE(TYPE)                                                          \
    template void ninfer::ops::gguf::detail::narrow_launch<TYPE>(                                  \
        const ninfer::ops::gguf::detail::NarrowArgs&, cudaStream_t)
