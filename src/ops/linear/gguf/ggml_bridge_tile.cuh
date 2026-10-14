#pragma once

// The tile kernel: the prefill product of the one-scale (IQ3_XXS, IQ3_S, IQ2_XXS, IQ4_XS) and
// two-scale (IQ2_XS, IQ2_S) i-quants on sm_120. A CTA owns 128 weight rows x 128 tokens with 384
// threads in two roles. Four producer warps keep cp.async rings of the raw blocks (two blocks per
// row) and of 64-value activation stages (four), and decode each stage with the vector kernels'
// Decoder<type> (a thread a weight row) into a three-slot int8 ring swizzled for ldmatrix. Eight
// consumer warps (2 x 4, a warp 64 rows x 32 tokens) run ldmatrix and one mma.m16n8k32 per 32 values
// (two m16n8k16 for per-16 scales) and scale every 32-value integer dot by its weight and activation
// factors in FP32: q8_1's arithmetic, as the matrix kernel does it. Named barriers hand the stages
// over: full[s] (producers arrive, consumers wait) and empty[s] (consumers arrive, producers wait).
// Results go straight through the product's epilogue.
//
// The activation is planar: int8 values [columns][K] and FP32 d [K / 32][columns rounded up to 128],
// the matrix kernel's D4 numbers (ggml_bridge.cu quantize_tile_kernel).

#include "ggml_bridge_vec.cuh"

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
inline constexpr int kScales = type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ? 2 : 1;

// Raw-ring copy unit and the guaranteed alignment of a block's first byte: IQ4_XS decodes with
// 8-byte loads (its 136-byte blocks are 8-byte aligned), every other type with 2-byte ones.
template <ggml_type type>
inline constexpr int kCopy = type == GGML_TYPE_IQ4_XS ? 8 : 4;
template <ggml_type type>
inline constexpr int kAlign = type == GGML_TYPE_IQ4_XS ? 8 : 2;

// One-scale types visit token tiles fastest, so the token tiles of a row block share its weights
// in L2 (the 27B's large matrices exceed it); the two-scale ones keep row tiles fastest, which
// measured faster for them (their 2.3-2.5 bits per weight matrices mostly fit L2).
template <ggml_type type>
inline constexpr bool kTokensFastest = kScales<type> == 1;

template <ggml_type type>
struct Layout {
    using D                     = vec::Decoder<type>;
    static constexpr int kTable = D::kTableWords > 0 ? D::kTableWords : 4;
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

// Epilogues compiled into the kernel: vec_store's runtime selection costs registers (and spills in
// the two-scale main loop), so the model's forms get their own instances and the rest kEpiGeneric.
enum Epi : int {
    kEpiF32      = 0, // FP32 Store
    kEpiBf16     = 1, // BF16 Store
    kEpiBf16Add  = 2, // BF16 Accumulate
    kEpiBf16Silu = 3, // BF16 GateProduct with SiLU
    kEpiGeneric  = 4, // vec_store
};

// Stages before the end at which the producers pull the epilogue's reads into L2.
constexpr int kPrefetchStages = 16;

// What the epilogue reads at (row, column): the gate of a gated product, the output it accumulates
// into, nothing for a store.
template <int EPI>
__device__ __forceinline__ float tile_operand(const VecStore& out, int row, int column) {
    const std::int64_t at = std::int64_t(column) * out.column_stride + row;
    if constexpr (EPI == kEpiBf16Add) {
        return __bfloat162float(out.bf16[at]);
    } else if constexpr (EPI == kEpiBf16Silu) {
        return out.gate[std::int64_t(column) * out.gate_column_stride + row];
    } else if constexpr (EPI == kEpiGeneric) {
        if (out.epilogue == Epilogue::GateProduct) {
            return out.gate[std::int64_t(column) * out.gate_column_stride + row];
        }
        if (out.epilogue == Epilogue::Accumulate) {
            return out.f32 != nullptr ? out.f32[at] : __bfloat162float(out.bf16[at]);
        }
        return 0.0f;
    } else {
        return 0.0f;
    }
}

// vec_store's arithmetic given the operand tile_operand read.
template <int EPI>
__device__ __forceinline__ void tile_finish(const VecStore& out, int row, int column, float value,
                                            float operand) {
    const std::int64_t at = std::int64_t(column) * out.column_stride + row;
    if constexpr (EPI == kEpiF32) {
        out.f32[at] = value;
    } else if constexpr (EPI == kEpiBf16) {
        out.bf16[at] = __float2bfloat16(value);
    } else if constexpr (EPI == kEpiBf16Add) {
        out.bf16[at] = __float2bfloat16(value + operand);
    } else if constexpr (EPI == kEpiBf16Silu) {
        out.bf16[at] = __float2bfloat16(value * vec_silu(operand));
    } else {
        if (out.epilogue == Epilogue::GateProduct) { value *= vec_gate(operand, out.activation); }
        if (out.epilogue == Epilogue::Accumulate) { value = operand + value; }
        if (out.f32 != nullptr) {
            out.f32[at] = value;
        } else {
            out.bf16[at] = __float2bfloat16(value);
        }
    }
}

template <ggml_type type, int KS, int EPI>
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
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kThreads) { table[i] = src[i]; }
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
            const std::uint8_t* src = wrow + std::int64_t(b) * D::kBlockBytes;
            const auto* base        = reinterpret_cast<const std::uint8_t*>(
                reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t(kCopy<type> - 1));
            const int n       = int((src + D::kBlockBytes - base + kCopy<type> - 1) / kCopy<type>);
            std::uint8_t* dst = smem + L::kRawOff + (b & 1) * L::kRawSlot + pt * L::kRawRow;
            for (int j = 0; j < n; ++j) {
                if constexpr (kCopy<type> == 8) {
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
                vec::Slice sl;
                D::decode(blk, (i % kStagesPerBlock) * kSubs + us, table, sl);
                std::uint8_t* a = as + us * kAPlane;
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
        float2 dv[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            dv[n] = *reinterpret_cast<const float2*>(dsm + su * kBN + wn * 32 + n * 8 + 2 * t);
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int rs = su * kBM + wm * 64 + m * 16 + 2 * g; // rows g and g + 8 (scale_index)
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
    };
    Frag f0, f1;
    bar_sync(kFull, kThreads);
    if constexpr (kS == 2) {
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
                operand[n][j] = r < p.rows && c < p.columns ? tile_operand<EPI>(p.out, r, c) : 0.0f;
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = row0 + wm * 64 + m * 16 + g + 8 * (j >> 1);
                const int c = tok0 + wn * 32 + n * 8 + 2 * t + (j & 1);
                if (r < p.rows && c < p.columns) {
                    tile_finish<EPI>(p.out, r, c, acc[m][n][j], operand[n][j]);
                }
            }
        }
    }
#endif
}

template <ggml_type type, int KS, int EPI>
void launch_k(const TileArgs& args, cudaStream_t stream) {
    auto* kern = kernel<type, KS, EPI>;
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

// One-scale types run a loop bound by K (measured 1-3.5% faster than static K there); two-scale
// ones take static-K forms at the 27B's widths (2-3% faster for them).
template <ggml_type type, int EPI>
void launch_e(const TileArgs& args, cudaStream_t stream) {
    if constexpr (kScales<type> == 2 && EPI != kEpiGeneric) {
        switch (args.k) {
        case 5120: return launch_k<type, 5120, EPI>(args, stream);
        case 6144: return launch_k<type, 6144, EPI>(args, stream);
        case 17408: return launch_k<type, 17408, EPI>(args, stream);
        default: break;
        }
    }
    launch_k<type, 0, EPI>(args, stream);
}

} // namespace tile

template <ggml_type type>
void tile_launch(const TileArgs& args, cudaStream_t stream) {
    const VecStore& o = args.out;
    if (o.f32 != nullptr && o.epilogue == Epilogue::Store) {
        return tile::launch_e<type, tile::kEpiF32>(args, stream);
    }
    if (o.f32 == nullptr && o.epilogue == Epilogue::Store) {
        return tile::launch_e<type, tile::kEpiBf16>(args, stream);
    }
    if (o.f32 == nullptr && o.epilogue == Epilogue::Accumulate) {
        return tile::launch_e<type, tile::kEpiBf16Add>(args, stream);
    }
    if (o.f32 == nullptr && o.epilogue == Epilogue::GateProduct && o.activation == 0) {
        return tile::launch_e<type, tile::kEpiBf16Silu>(args, stream);
    }
    tile::launch_e<type, tile::kEpiGeneric>(args, stream);
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_TILE_INSTANCE(TYPE)                                                            \
    template void ninfer::ops::gguf::detail::tile_launch<TYPE>(                                    \
        const ninfer::ops::gguf::detail::TileArgs&, cudaStream_t)
