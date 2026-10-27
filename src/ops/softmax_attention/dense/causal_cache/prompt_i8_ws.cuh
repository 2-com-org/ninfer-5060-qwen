#pragma once

// Warp-specialized prompt kernel for the h24 rk4v4 cache (sm_120a), the device profile's "staged"
// launch of attn_prompt/h24/rk4v4.
//
//   * Two consumer warpgroups (warps 0-7) each own 16 query rows of one head for the whole key
//     sweep and run the fast kernel's arithmetic (prompt_i8_fast.cuh): the private G64 INT8 query
//     encoding, m16n8k32.s8 QK with FP32 per-group scaling, the online softmax, FP16 PV partials
//     over each 64-key tile promoted once into the FP32 accumulator, and the exact power-of-two
//     tile shift that keeps those partials representable.
//   * One producer warpgroup (warps 8-11) loads the packed Lloyd4 keys and int4 values, expands
//     keys into the INT8 tile two key tiles ahead (two stages), and decodes the next tile's values
//     once into the FP16 staging tile while the consumers run that tile's QK and softmax.
//   * setmaxnreg moves registers from the producers to the consumers. Named barriers publish and
//     release each key stage and the value tile; there is no CTA-wide barrier in the sweep. Each
//     consumer warpgroup waits on its own full barriers, so the two drift apart within the slack
//     the stages leave instead of starting every phase together.
//
// Fp8Pv (the profile's "staged_fp8") replaces the FP16 PV with block-scaled e4m3 MMAs, which on
// sm_120 accumulate in FP32 at the full e4m3 rate (the unscaled FP32 form runs at half of it).
// Each row's probabilities are taken against the running maximum lowered by the whole octaves that
// bring it within one of the row's maximum in the key tile, times 2^8, and rounded to e4m3: the
// tile's largest lies in (128, 256]. The octaves and 2^-8 return through the MMA's UE8M0 A scale
// of that row. Each decoded value is rounded to e4m3 under a power of two per (key tile,
// 32-dimension group) that maps the group's bound into [128, 256), whose inverse is the B scale.
// Each key is staged as a row at its probability's k position, which the consumers read through
// the transposing b8 ldmatrix. A tile's two 32-key MMAs per n8 tile chain from a zero accumulator,
// and one FFMA per element rescales the running sum and adds that FP32 partial: the MMA's own
// FP32 addition truncates by a few units in the last place, which over a long sweep would bias
// the running sum. Elements below about 2^-14 of their tile's largest probability or value become
// subnormal. The row sums keep the unrounded FP32 probabilities times the same powers of two.
// Fp8Pv also scores each key with one int32 chain over all 256 dimensions, converted and scaled
// once: the image re-expresses each key's G64 groups under the key's largest group scale (each
// group's Lloyd4 levels times its scale over that maximum, rounded to INT8, half a step at most)
// and keeps one FP32 scale per key, and each query row takes one INT8 scale.
// causal_attention_prompt_ws_image_kernel decodes every (KV head, key tile) once into a workspace
// image of its key stage and e4m3 value tile, which every row block of the KV head's query heads
// would otherwise decode again, and the producers only copy images: two key and two value stages,
// the e4m3 tile being half the FP16 one.
// A sweep longer than kCausalPromptWsSegmentTiles key tiles runs as consecutive launches over
// segments of at most that many tiles, so the images take bounded workspace at any context: each
// launch stores every consumer lane's FP32 softmax state (running maxima, lane-partial row sums and
// accumulator) to workspace and the next launch loads it. Every tile still performs the same
// operations in the same order on the same values, so a segmented sweep is bit-identical to one
// launch over all tiles.
//
// A launch given the attention output gate (BF16, the output's layout) publishes each element
// rounded to BF16 and then multiplied by the sigmoid of its gate, rounded again: the bytes the
// standalone sigmoid_mul writes over the ungated output, inactive rows included (zero times the
// gate's sigmoid, so a NaN gate propagates as it would there).
//
// Without Fp8Pv each output element performs the same operations in the same order as the
// eight-warp fast kernel with an FP16 value tile. The exponentials flush subnormal results to
// zero: probabilities below 2^-126 vanish in the FP16 PV operands either way and against the
// tile's unit maximum in the row sum, so only a rescale factor below 2^-126 can differ, by less
// than 2^-126 relative.

#include "ops/common/math.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"

#include <cuda_fp8.h>

namespace ninfer::ops {

inline constexpr int kCausalPromptWsConsumers    = 256;
inline constexpr int kCausalPromptWsProducers    = 128;
inline constexpr int kCausalPromptWsThreads      = kCausalPromptWsConsumers + kCausalPromptWsProducers;
inline constexpr int kCausalPromptWsRows         = kCausalPromptWsConsumers / 32 * 16;
inline constexpr int kCausalPromptWsProducerRegs = 56;
inline constexpr int kCausalPromptWsConsumerRegs = 224;
// Q codes, two key stages (INT8 codes then FP32 G64 scales as [group][key]), then the FP16
// value tile.
inline constexpr int kCausalPromptWsQBytes = kCausalPromptWsRows * kCausalPromptHeadDim;
inline constexpr int kCausalPromptWsKScaleBytes =
    kCausalPromptI8FastBc * kCausalPromptI8FastGroups * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptWsStageBytes =
    kCausalPromptI8FastTileBytes + kCausalPromptWsKScaleBytes;
inline constexpr int kCausalPromptWsValueOffset =
    kCausalPromptWsQBytes + 2 * kCausalPromptWsStageBytes;
inline constexpr int kCausalPromptWsSmemBytes =
    kCausalPromptWsValueOffset + kCausalPromptI8FastBc * kCausalPromptHeadDim * 2;

static_assert(kCausalPromptWsSmemBytes == 100352);
// Fp8Pv's producers only copy images (26 registers on sm_120a), so it moves registers from the
// producers to the consumers, whose QK, softmax and PV otherwise spill. Each SMSP's 16K registers
// hold two consumer warps and one producer warp (2 x 232 + 32 of the launch's 3 x 168); a block of
// fewer producer warps cannot raise the per-thread budget, since nine warps put three on one SMSP.
inline constexpr int kCausalPromptWsFp8ProducerRegs = 32;
inline constexpr int kCausalPromptWsFp8ConsumerRegs = 232;
static_assert(kCausalPromptWsConsumers * kCausalPromptWsConsumerRegs +
                  kCausalPromptWsProducers * kCausalPromptWsProducerRegs <=
              kCausalPromptWsThreads * 168);
static_assert(kCausalPromptWsConsumers * kCausalPromptWsFp8ConsumerRegs +
                  kCausalPromptWsProducers * kCausalPromptWsFp8ProducerRegs <=
              kCausalPromptWsThreads * 168);

// FP8 PV: the value range bound and the probability scale (see the header comment), and the size of
// one of the two transposed e4m3 value stages that share the FP16 value tile's space.
inline constexpr int kCausalPromptWsFp8ValueStageBytes =
    kCausalPromptI8FastBc * kCausalPromptHeadDim;
static_assert(2 * kCausalPromptWsFp8ValueStageBytes ==
              kCausalPromptI8FastBc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half)));
inline constexpr float kCausalPromptWsFp8ValueRange    = 256.0f;
inline constexpr int kCausalPromptWsFp8ProbabilityShift = 8;
// Both operands stay below the largest finite e4m3, 448.
static_assert((1 << kCausalPromptWsFp8ProbabilityShift) <= 448 &&
              kCausalPromptWsFp8ValueRange <= 448.0f);
// The largest whole-octave offset whose A scale, 2^-(offset + shift), is a normal UE8M0 and FP32.
inline constexpr int kCausalPromptWsFp8MaxOctaves = 126 - kCausalPromptWsFp8ProbabilityShift;

// One (KV head, key tile) image of staged_fp8: the key stage (INT8 codes, FP32 scales), the e4m3
// value tile, then its eight group unscales. Images are [KV head][tile] for the launch's tiles.
inline constexpr int kCausalPromptWsImageValueOffset = kCausalPromptWsStageBytes;
inline constexpr int kCausalPromptWsImageUnscaleOffset =
    kCausalPromptWsImageValueOffset + kCausalPromptWsFp8ValueStageBytes;
inline constexpr int kCausalPromptWsImageBytes =
    kCausalPromptWsImageUnscaleOffset +
    kCausalPromptHeadDim / kKVCacheInt4ValueGroup * static_cast<int>(sizeof(float));
static_assert(kCausalPromptWsImageBytes % 16 == 0 && kCausalPromptWsStageBytes % 16 == 0);

// staged_fp8's longest single-launch sweep (32768 keys: 66 MiB of images for four KV heads, which
// with a 2048-column launch's carried state stays below the prompt's MLP workspace), and the
// softmax state one consumer lane carries between segments: its eight groups' four n8 tiles of four
// accumulators, two running maxima and two lane-partial row sums. A launch's state is
// [CTA][float][consumer thread], so each store or load is one coalesced row per CTA.
inline constexpr int kCausalPromptWsSegmentTiles = 512;
inline constexpr int kCausalPromptWsStateFloats =
    kCausalPromptHeadDim / kKVCacheInt4ValueGroup * 4 * 4 + 4;
inline constexpr int kCausalPromptWsLoadState  = 1;
inline constexpr int kCausalPromptWsStoreState = 2;

// staged_fp8's output tile staged in shared memory once the sweep is over: BF16 rows padded by one
// 16-byte chunk, so the accumulator layout's two-byte stores (eight rows of a warp at once) fall on
// distinct banks and each row still leaves as aligned 16-byte chunks.
inline constexpr int kCausalPromptWsOutRowElements = kCausalPromptHeadDim + 8;
static_assert(kCausalPromptWsRows * kCausalPromptWsOutRowElements *
                  static_cast<int>(sizeof(__nv_bfloat16)) <=
              kCausalPromptWsSmemBytes);

// Producer thread p's key chunks are p + 128 i: key chunk >> 4, dimensions 16 (chunk & 15) on.
inline constexpr int kCausalPromptWsKeyChunks =
    kCausalPromptI8FastBc * (kCausalPromptHeadDim / 16) / kCausalPromptWsProducers;
static_assert(kCausalPromptWsKeyChunks * kCausalPromptWsProducers ==
              kCausalPromptI8FastBc * (kCausalPromptHeadDim / 16));

// Byte offset of 16-byte chunk c of key row r in the e4m3 value tile: 256-byte rows whose chunks
// are XOR-swizzled by the row, so the producers' chunk stores and the consumers' transposing
// ldmatrix are both free of bank conflicts.
__device__ __forceinline__ int causal_prompt_ws_fp8_value_offset(int row, int chunk) {
    return row * kCausalPromptHeadDim + ((chunk ^ (row & 15)) << 4);
}

// Two 16 x 16 e4m3 blocks (rows k .. k + 15 and k + 16 .. k + 31 of 16 columns), transposed:
// r0 / r2 are the m16n8k32 B fragment of the first eight columns and r1 / r3 of the second eight.
__device__ __forceinline__ void causal_prompt_ws_ldmatrix_b8_x2_t(unsigned& r0, unsigned& r1,
                                                                  unsigned& r2, unsigned& r3,
                                                                  unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Tile key of logical key slot kk (0..3) in quad q: the probability A fragment holds, per lane,
// keys {2t, 2t + 1} of two adjacent n8 score tiles as four consecutive k positions.
__device__ __forceinline__ int causal_prompt_ws_fp8_quad_key(int quad, int kk) {
    return 32 * (quad >> 3) + 16 * ((quad >> 2) & 1) + 2 * (quad & 3) + (kk & 1) + 8 * (kk >> 1);
}

__device__ __forceinline__ unsigned causal_prompt_ws_e4m3x2(float lo, float hi) {
    return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ unsigned causal_prompt_ws_e4m3x2(__half2 value) {
    return __nv_cvt_halfraw2_to_fp8x2(__half2_raw(value), __NV_SATFINITE, __NV_E4M3);
}

// Thread p's share of key tile `page`: its packed Lloyd4 chunks and key p's G64 scales. Keys from
// `valid` on keep the zero placeholder and zero scales; their scores are masked.
template <typename Geometry>
__device__ __forceinline__ void causal_prompt_ws_load_keys(
    const std::int8_t* __restrict__ cache_k, const __half* __restrict__ cache_k_scale, int page,
    int kv_head, int valid, int p,
    KvPackedKeyChunk<KvKeyCoding::Lloyd4> (&packed)[kCausalPromptWsKeyChunks], uint2& scales) {
#pragma unroll
    for (int i = 0; i < kCausalPromptWsKeyChunks; ++i) {
        const int chunk = p + i * kCausalPromptWsProducers;
        const int key   = chunk >> 4;
        packed[i]       = KvPackedKeyChunk<KvKeyCoding::Lloyd4>{};
        if (key < valid) {
            packed[i] = load_vec<KvPackedKeyChunk<KvKeyCoding::Lloyd4>>(
                reinterpret_cast<const std::uint8_t*>(cache_k) +
                kv_cache_packed_key_chunk_index<Geometry, KvKeyCoding::Lloyd4>(
                    page, kv_head, (chunk & 15) * 16, key));
        }
    }
    scales = make_uint2(0, 0);
    if (p < valid) {
        scales = load_vec<uint2>(cache_k_scale +
                                 kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, p));
    }
}

// Expands thread p's share into a key stage: INT8 codes at [key][dimension] in 16-byte chunks
// swizzled by the key, then the scales widened once to FP32 as [group][key], so a consumer reads a
// key pair's scales with one load.
__device__ __forceinline__ void causal_prompt_ws_commit_keys(
    unsigned char* __restrict__ stage,
    const KvPackedKeyChunk<KvKeyCoding::Lloyd4> (&packed)[kCausalPromptWsKeyChunks], uint2 scales,
    int p) {
    constexpr int D = kCausalPromptHeadDim;
#pragma unroll
    for (int i = 0; i < kCausalPromptWsKeyChunks; ++i) {
        const int chunk = p + i * kCausalPromptWsProducers;
        const int key   = chunk >> 4;
        const int c     = chunk & 15;
        store_vec(stage + key * D + ((c ^ (key & 7)) << 4),
                  kv_cache_packed_key_expand16<KvKeyCoding::Lloyd4>(packed[i]));
    }
    if (p < kCausalPromptI8FastBc) {
        float* const k_scale_s = reinterpret_cast<float*>(stage + kCausalPromptI8FastTileBytes);
        const __half* const h  = reinterpret_cast<const __half*>(&scales);
#pragma unroll
        for (int grp = 0; grp < kCausalPromptI8FastGroups; ++grp) {
            k_scale_s[grp * kCausalPromptI8FastBc + p] = __half2float(h[grp]);
        }
    }
}

// The Lloyd4 expansion tables {positive low, positive high, negative low, negative high} of the
// INT8 codes times `ratio`, in [0, 1], rounded to the nearest INT8.
__device__ __forceinline__ uint4 causal_prompt_ws_lloyd4_tables(float ratio) {
    unsigned lo = 0u;
    unsigned hi = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float low  = ratio * static_cast<float>(kv_cache_lloyd4_code(j));
        const float high = ratio * static_cast<float>(kv_cache_lloyd4_code(j + 4));
        lo |= static_cast<unsigned>(__float2int_rn(low)) << (8 * j);
        hi |= static_cast<unsigned>(__float2int_rn(high)) << (8 * j);
    }
    return make_uint4(lo, hi, __vneg4(lo), __vneg4(hi));
}

// kv_cache_lloyd4_expand4 through runtime tables.
__device__ __forceinline__ unsigned causal_prompt_ws_lloyd4_expand4(unsigned nibbles, uint4 t) {
    const unsigned magnitude = nibbles & 0x7777u;
    const unsigned pos       = __byte_perm(t.x, t.y, magnitude);
    const unsigned neg       = __byte_perm(t.z, t.w, magnitude);
    const unsigned mask      = __byte_perm(0u, 0xFF00u, ((nibbles >> 3) & 0x1111u) | 0x4444u);
    return (neg & mask) | (pos & ~mask);
}

// The image's key stage: thread p's chunks expanded through their (key, group) tables, swizzled as
// causal_prompt_ws_commit_keys stores them, then one FP32 scale per key.
__device__ __forceinline__ void causal_prompt_ws_commit_keys_per_key(
    unsigned char* __restrict__ stage,
    const KvPackedKeyChunk<KvKeyCoding::Lloyd4> (&packed)[kCausalPromptWsKeyChunks],
    const uint4 (&tables)[kCausalPromptI8FastBc][kCausalPromptI8FastGroups],
    const float (&key_scales)[kCausalPromptI8FastBc], int p) {
    constexpr int D = kCausalPromptHeadDim;
#pragma unroll
    for (int i = 0; i < kCausalPromptWsKeyChunks; ++i) {
        const int chunk = p + i * kCausalPromptWsProducers;
        const int key   = chunk >> 4;
        const int c     = chunk & 15;
        const uint4 t   = tables[key][c >> 2];
        const uint2 q   = packed[i];
        const int4 codes =
            make_int4(static_cast<int>(causal_prompt_ws_lloyd4_expand4(q.x, t)),
                      static_cast<int>(causal_prompt_ws_lloyd4_expand4(q.x >> 16, t)),
                      static_cast<int>(causal_prompt_ws_lloyd4_expand4(q.y, t)),
                      static_cast<int>(causal_prompt_ws_lloyd4_expand4(q.y >> 16, t)));
        store_vec(stage + key * D + ((c ^ (key & 7)) << 4), codes);
    }
    if (p < kCausalPromptI8FastBc) {
        reinterpret_cast<float*>(stage + kCausalPromptI8FastTileBytes)[p] = key_scales[p];
    }
}

// One CTA per (key tile, KV head) of a staged_fp8 launch writes that tile's image. Keys and values
// from the launch's last visible key on decode as the producers' placeholders would.
//
// Values: unit u = 0, 1 of a thread is (key quad, 16-dimension slot), lanes 0..7 taking consecutive
// slots, so one load covers 64 contiguous code bytes of each of four keys. Codes widen exactly to
// FP16, take their FP16 scale times the range power of two (one FP16 rounding of code x scale x
// 2^k) and round once more to e4m3. Key quad q's key kk takes tile row 4q + kk, the k position of
// its probability; each conversion's pair, dimensions m and m + 4, fills two adjacent bytes, so
// chunk byte p holds dimension 8 (p >> 3) + ((p >> 1) & 3) + 4 (p & 1) of the unit's 16.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kCausalPromptWsProducers) void causal_attention_prompt_ws_image_kernel(
    const std::int8_t* __restrict__ cache_k, const std::int8_t* __restrict__ cache_v,
    const __half* __restrict__ cache_k_scale, const __half* __restrict__ cache_v_scale,
    Metadata metadata, const std::int32_t* __restrict__ positions, std::int32_t width,
    unsigned char* __restrict__ images, std::int32_t image_tiles, std::int32_t kb_begin) {
    constexpr int Bc            = kCausalPromptI8FastBc;
    constexpr int VGroups       = kCausalPromptHeadDim / kKVCacheInt4ValueGroup;
    constexpr int Warps         = kCausalPromptWsProducers / 32;
    constexpr unsigned FullMask = 0xffffffffu;
    __shared__ float group_maxima[Warps][VGroups];
    __shared__ uint4 key_tables[Bc][kCausalPromptI8FastGroups];
    __shared__ float key_scales[Bc];

    // Image slot blockIdx.x of the segment that starts at key tile kb_begin.
    const int image_slot = static_cast<int>(blockIdx.x);
    const int kb         = kb_begin + image_slot;
    const int kv_head    = static_cast<int>(blockIdx.y);
    const int tokens     = metadata.valid_tokens(width);
    const int visible    = tokens > 0 ? positions[0] + tokens : 0;
    if (kb * Bc >= visible) { return; }
    const int page  = metadata.block_table()[kb];
    const int valid = min(Bc, visible - kb * Bc);
    const int p     = static_cast<int>(threadIdx.x);
    const int pwarp = p >> 5;
    const int lane  = p & 31;
    unsigned char* const image =
        images + (static_cast<std::size_t>(kv_head) * image_tiles + image_slot) *
                     kCausalPromptWsImageBytes;

    KvPackedKeyChunk<KvKeyCoding::Lloyd4> k_packed[kCausalPromptWsKeyChunks];
    uint2 k_scales;
    causal_prompt_ws_load_keys<Geometry>(cache_k, cache_k_scale, page, kv_head, valid, p, k_packed,
                                         k_scales);
    const int slot  = (lane & 7) + 8 * (pwarp & 1);
    const int group = slot >> 1;
    const auto quad = [&](int u) { return (lane >> 3) + 4 * (pwarp >> 1) + 8 * u; };
    uint2 v_packed[8];
    __half2 v_scales[4]; // unit u's key pairs (0, 1) and (2, 3) at 2u and 2u + 1
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const int key        = causal_prompt_ws_fp8_quad_key(quad(u), kk);
            v_packed[4 * u + kk] = make_uint2(0, 0);
            __half scale         = __float2half_rn(0.0f);
            if (key < valid) {
                v_packed[4 * u + kk] = load_vec<uint2>(
                    reinterpret_cast<const std::uint8_t*>(cache_v) +
                    kv_cache_int4_value_code_index<Geometry>(page, kv_head, slot * 8, key));
                scale = cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(page, kv_head,
                                                                                group, key)];
            }
            __half2& pair = v_scales[2 * u + (kk >> 1)];
            pair = kk & 1 ? __halves2half2(__low2half(pair), scale) : __halves2half2(scale, scale);
        }
    }
    // One scale per key, its largest G64 scale; keys from `valid` on have zero scales and tables.
    if (p < Bc) {
        const __half* const h = reinterpret_cast<const __half*>(&k_scales);
        float largest         = 0.0f;
#pragma unroll
        for (int grp = 0; grp < kCausalPromptI8FastGroups; ++grp) {
            largest = fmaxf(largest, __half2float(h[grp]));
        }
        const float inv = largest > 0.0f ? 1.0f / largest : 0.0f;
        key_scales[p]   = largest;
#pragma unroll
        for (int grp = 0; grp < kCausalPromptI8FastGroups; ++grp) {
            key_tables[p][grp] =
                causal_prompt_ws_lloyd4_tables(fminf(1.0f, __half2float(h[grp]) * inv));
        }
    }
    __syncthreads();
    causal_prompt_ws_commit_keys_per_key(image, k_packed, key_tables, key_scales, p);

    // Each group's largest scale over the tile bounds its decoded values by 8 * max; the range
    // power of two maps that bound into [range / 2, range).
    __half2 pair_maximum = __habs2(v_scales[0]);
#pragma unroll
    for (int i = 1; i < 4; ++i) { pair_maximum = __hmax2(pair_maximum, __habs2(v_scales[i])); }
    float maximum = fmaxf(__low2float(pair_maximum), __high2float(pair_maximum));
    maximum       = fmaxf(maximum, __shfl_xor_sync(FullMask, maximum, 1));
    maximum       = fmaxf(maximum, __shfl_xor_sync(FullMask, maximum, 8));
    maximum       = fmaxf(maximum, __shfl_xor_sync(FullMask, maximum, 16));
    if ((lane & 0x19) == 0) { group_maxima[pwarp][group] = maximum; }
    __syncthreads();
    const auto range_shift = [&](int g) {
        const int w       = g >> 2;
        const float bound = 8.0f * fmaxf(group_maxima[w][g], group_maxima[w + 2][g]);
        int exponent      = 0;
        if (bound > 0.0f) { frexpf(bound / kCausalPromptWsFp8ValueRange, &exponent); }
        return -exponent;
    };
    if (p < VGroups) {
        reinterpret_cast<float*>(image + kCausalPromptWsImageUnscaleOffset)[p] =
            ldexpf(1.0f, -range_shift(p));
    }
    const float value_scale = ldexpf(1.0f, range_shift(group));
    const __half2 offset    = __float2half2_rn(1032.0f);
    unsigned char* const tile = image + kCausalPromptWsImageValueOffset;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const float2 pair = __half22float2(v_scales[2 * u + (kk >> 1)]);
            const __half2 sc  = __float2half2_rn((kk & 1 ? pair.y : pair.x) * value_scale);
            const unsigned code[2] = {v_packed[4 * u + kk].x ^ 0x88888888u,
                                      v_packed[4 * u + kk].y ^ 0x88888888u};
            unsigned e[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const unsigned x = ((code[h] >> (4 * r)) & 0x000f000fu) | 0x64006400u;
                    e[h][r]          = causal_prompt_ws_e4m3x2(
                        __hmul2(__hsub2(load_vec<__half2>(&x), offset), sc));
                }
            }
            store_vec(tile + causal_prompt_ws_fp8_value_offset(4 * quad(u) + kk, slot),
                      make_uint4(e[0][0] | (e[0][1] << 16), e[0][2] | (e[0][3] << 16),
                                 e[1][0] | (e[1][1] << 16), e[1][2] | (e[1][3] << 16)));
        }
    }
}

__device__ __forceinline__ void causal_prompt_ws_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void causal_prompt_ws_bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

template <typename Geometry, typename Metadata, bool Fp8Pv = false>
__global__ __launch_bounds__(kCausalPromptWsThreads, 1) void causal_attention_prompt_ws_kernel(
    const __nv_bfloat16* __restrict__ q, const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, const unsigned char* __restrict__ images, std::int32_t image_tiles,
    std::int32_t kb_begin, std::int32_t kb_end, float* __restrict__ state,
    std::int32_t state_flags, const __nv_bfloat16* __restrict__ gate) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    constexpr int VGroups       = D / kKVCacheInt4ValueGroup;
    constexpr int Consumers     = kCausalPromptWsConsumers;
    constexpr int Producers     = kCausalPromptWsProducers;
    constexpr int Threads       = kCausalPromptWsThreads;
    constexpr int Br            = kCausalPromptWsRows;
    constexpr int Bc            = kCausalPromptI8FastBc;
    constexpr int Groups        = kCausalPromptI8FastGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int ChunksPer     = kCausalPromptWsKeyChunks;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    // Named barriers: key stage s full for warpgroup w (1 + 2s + w, the producers and that
    // warpgroup), key stage s empty (5 + s), value tile full for warpgroup w (7 + w), value tile
    // empty (9), producer-only reduction (10). Empty barriers take every thread. Fp8Pv's second
    // value stage takes full 11 + w and empty 13, and its epilogue's consumers 14.
    constexpr int kKeyFull = 1, kKeyEmpty = 5, kValueFull = 7, kValueEmpty = 9, kProducers = 10;
    constexpr int kValueFull1 = 11, kValueEmpty1 = 13, kEpilogue = 14;
    const auto value_full  = [](int stage) { return stage == 0 ? kValueFull : kValueFull1; };
    const auto value_empty = [](int stage) { return stage == 0 ? kValueEmpty : kValueEmpty1; };
    constexpr int kHalf = Producers + 128;
    static_assert(GroupKc == 2 && VGroups == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __half* const values = reinterpret_cast<__half*>(smem_raw + kCausalPromptWsValueOffset);
    // Producer maxima alternate by tile: with two key tiles no barrier that every producer takes
    // separates one tile's reads from the next tile's writes.
    __shared__ float value_scale_maxima[2][Producers / 32];
    __shared__ int value_shift_s;
    __shared__ float value_unscale_s[Fp8Pv ? 2 : 1][Fp8Pv ? VGroups : 1];
    unsigned char* const values8 = smem_raw + kCausalPromptWsValueOffset;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q0         = (row_blocks - 1 - linear / Geometry::QHeads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros (gated, with a gate).
    const auto gated_pair = [](unsigned value, unsigned gate_pair) {
        const __nv_bfloat162 g = load_vec<__nv_bfloat162>(&gate_pair);
        const __nv_bfloat162 x = load_vec<__nv_bfloat162>(&value);
        const __nv_bfloat162 r =
            __floats2bfloat162_rn(__low2float(x) * sigmoid(__low2float(g)),
                                  __high2float(x) * sigmoid(__high2float(g)));
        return load_vec<unsigned>(&r);
    };
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid         = token < tokens;
        const std::int64_t index = causal_prompt_q_index<Geometry>(q_head, d0, token);
        uint2 packed             = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                              pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        if (gate != nullptr) {
            const uint2 g = load_vec<uint2>(&gate[index]);
            packed        = make_uint2(gated_pair(packed.x, g.x), gated_pair(packed.y, g.y));
        }
        store_vec(&out[index], packed);
    };

    if (q0 >= tokens) {
        // Only the sweep's last segment publishes.
        if ((state_flags & kCausalPromptWsStoreState) != 0) { return; }
        for (int element = tid; element < Br * (D / 4); element += Threads) {
            const int row = element / (D / 4);
            store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // This launch sweeps key tiles [kb_begin, kb_begin + sweep) of the CTA's key_blocks; pipeline
    // stages and images are indexed by the position t within that segment. Only staged_fp8 is
    // launched in segments: otherwise kb_begin is 0 and kb_end covers every tile, so t is kb.
    const int sweep = max(0, min(key_blocks, kb_end) - kb_begin);

    const auto stage_base = [&](int t) {
        return smem_raw + kCausalPromptWsQBytes + (t & 1) * kCausalPromptWsStageBytes;
    };

    if (warp >= Consumers / 32) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(
                         Fp8Pv ? kCausalPromptWsFp8ProducerRegs : kCausalPromptWsProducerRegs)
                     : "memory");
        const int ptid  = tid - Consumers;
        const int pwarp = ptid >> 5;
        const auto publish_keys = [&](int stage) {
            __threadfence_block();
            causal_prompt_ws_bar_arrive(kKeyFull + 2 * stage, kHalf);
            causal_prompt_ws_bar_arrive(kKeyFull + 2 * stage + 1, kHalf);
        };
        const auto publish_values = [&](int stage) {
            __threadfence_block();
            causal_prompt_ws_bar_arrive(value_full(stage), kHalf);
            causal_prompt_ws_bar_arrive(value_full(stage) + 1, kHalf);
        };

        if constexpr (Fp8Pv) {
            // A launch's last row block sweeps every image the stage kernel wrote.
            if (sweep > image_tiles) { __trap(); }
            const unsigned char* const head_images =
                images + static_cast<std::size_t>(kv_head) * image_tiles * kCausalPromptWsImageBytes;
            const auto image = [&](int t) {
                return head_images + static_cast<std::size_t>(t) * kCausalPromptWsImageBytes;
            };
            const auto copy_keys = [&](int t) {
                unsigned char* const stage = stage_base(t);
                for (int c = ptid; c < kCausalPromptWsStageBytes / 16; c += Producers) {
                    cp_async<16, Cache::cg>(stage + 16 * c, image(t) + 16 * c);
                }
            };
            const auto copy_values = [&](int t) {
                unsigned char* const tile = values8 + (t & 1) * kCausalPromptWsFp8ValueStageBytes;
                const unsigned char* const source = image(t) + kCausalPromptWsImageValueOffset;
#pragma unroll
                for (int i = 0; i < kCausalPromptWsFp8ValueStageBytes / 16 / Producers; ++i) {
                    const int c = ptid + i * Producers;
                    cp_async<16, Cache::cg>(tile + 16 * c, source + 16 * c);
                }
                if (ptid < VGroups) {
                    cp_async<4>(&value_unscale_s[t & 1][ptid],
                                image(t) + kCausalPromptWsImageUnscaleOffset + 4 * ptid);
                }
            };
            const int first = min(2, sweep);
            for (int t = 0; t < first; ++t) { copy_keys(t); }
            cp_commit();
            for (int t = 0; t < first; ++t) { copy_values(t); }
            cp_commit();
            cp_wait<1>();
            for (int t = 0; t < first; ++t) { publish_keys(t); }
            cp_wait<0>();
            for (int t = 0; t < first; ++t) { publish_values(t); }
            // Iteration t refills key stage t & 1 with tile t + 2 once the consumers finish its
            // QK, then value stage t & 1 once they finish its PV. Each wait pairs with exactly one
            // consumer release, under the same conditions.
#pragma unroll 1
            for (int t = 0; t + 2 < sweep; ++t) {
                causal_prompt_ws_bar_sync(kKeyEmpty + (t & 1), Threads);
                copy_keys(t + 2);
                cp_commit();
                cp_wait<0>();
                publish_keys(t & 1);
                causal_prompt_ws_bar_sync(value_empty(t & 1), Threads);
                copy_values(t + 2);
                cp_commit();
                cp_wait<0>();
                publish_values(t & 1);
            }
            return;
        } else {
            using KeyChunk = KvPackedKeyChunk<KvKeyCoding::Lloyd4>;
            KeyChunk k_packed[ChunksPer];
            uint2 k_scales = make_uint2(0, 0);
            uint2 v_packed[ChunksPer];
            __half v_scales[ChunksPer];
            int value_shift = 0;

            const auto load_keys = [&](int kb) {
                causal_prompt_ws_load_keys<Geometry>(cache_k, cache_k_scale, block_table[kb],
                                                     kv_head, min(Bc, max_query_abs + 1 - kb * Bc),
                                                     ptid, k_packed, k_scales);
            };
            const auto commit_keys = [&](int kb) {
                causal_prompt_ws_commit_keys(stage_base(kb), k_packed, k_scales, ptid);
            };
            const auto load_values = [&](int kb) {
                const int page  = block_table[kb];
                const int valid = min(Bc, max_query_abs + 1 - kb * Bc);
    #pragma unroll
                for (int i = 0; i < ChunksPer; ++i) {
                    const int chunk = ptid + i * Producers;
                    const int key   = chunk >> 4;
                    v_packed[i]     = make_uint2(0, 0);
                    v_scales[i]     = __float2half_rn(0.0f);
                    if (key < valid) {
                        v_packed[i] = load_vec<uint2>(
                            reinterpret_cast<const std::uint8_t*>(cache_v) +
                            kv_cache_int4_value_code_index<Geometry>(page, kv_head, (chunk & 15) * 8, key));
                        v_scales[i] = cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(
                            page, kv_head, (chunk & 15) / 2, key)];
                    }
                }
            };
            // The tile's largest value scale selects the exact power of two that keeps its 64-key FP16
            // partials representable; the consumers multiply the partial back at promotion.
            const auto find_value_shift = [&](int kb) {
                float maximum = 0.0f;
    #pragma unroll
                for (int i = 0; i < ChunksPer; ++i) {
                    maximum = fmaxf(maximum, fabsf(__half2float(v_scales[i])));
                }
                maximum = warp_max(maximum, FullMask);
                if (lane == 0) { value_scale_maxima[kb & 1][pwarp] = maximum; }
                causal_prompt_ws_bar_sync(kProducers, Producers);
                maximum = 0.0f;
    #pragma unroll
                for (int w = 0; w < Producers / 32; ++w) {
                    maximum = fmaxf(maximum, value_scale_maxima[kb & 1][w]);
                }
                value_shift = 0;
                while (ldexpf(maximum, -value_shift) > kCausalPromptI8FastPackedF16PartialScaleLimit) {
                    ++value_shift;
                }
            };
            // Codes widen exactly to FP16 and receive their (shifted) FP16 group scale once. The
            // columns are permuted so n8 tile r of a 32-dimension group holds dimensions 4n + r.
            const auto decode_values = [&] {
                if (ptid == 0) { value_shift_s = value_shift; }
                const __half2 mul    = __float2half2_rn(ldexpf(1.0f, -value_shift));
                const __half2 offset = __float2half2_rn(1032.0f);
    #pragma unroll
                for (int i = 0; i < ChunksPer; ++i) {
                    const int chunk      = ptid + i * Producers;
                    const int key        = chunk >> 4;
                    const int group      = (chunk & 15) >> 1;
                    const int half_group = chunk & 1;
                    const unsigned low   = v_packed[i].x ^ 0x88888888u;
                    const unsigned high  = v_packed[i].y ^ 0x88888888u;
                    __half2 s            = __half2half2(v_scales[i]);
                    if (value_shift != 0) { s = __hmul2(s, mul); }
    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const unsigned a = ((low >> (4 * r)) & 0x000f000fu) | 0x64006400u;
                        const unsigned b = ((high >> (4 * r)) & 0x000f000fu) | 0x64006400u;
                        const __half2 va = __hmul2(__hsub2(load_vec<__half2>(&a), offset), s);
                        const __half2 vb = __hmul2(__hsub2(load_vec<__half2>(&b), offset), s);
                        const int column = group * 32 + r * 8 + half_group * 4;
                        store_vec(&values[key * D + causal_prompt_swz(key, column)],
                                  make_uint2(load_vec<unsigned>(&va), load_vec<unsigned>(&vb)));
                    }
                }
            };

            load_keys(0);
            commit_keys(0);
            publish_keys(0);
            if (key_blocks > 1) {
                load_keys(1);
                commit_keys(1);
                publish_keys(1);
            }
            load_values(0);
            find_value_shift(0);
            decode_values();
            publish_values(0);
            // Iteration kb refills key stage kb & 1 with tile kb + 2 once the consumers finish its
            // QK, then the value tile with tile kb + 1 once they finish this tile's PV. Each wait
            // pairs with exactly one consumer release, under the same conditions.
#pragma unroll 1
            for (int kb = 0; kb < key_blocks; ++kb) {
                const bool next_keys   = kb + 2 < key_blocks;
                const bool next_values = kb + 1 < key_blocks;
                if (next_keys) { load_keys(kb + 2); }
                if (next_values) { load_values(kb + 1); }
                if (next_keys) {
                    causal_prompt_ws_bar_sync(kKeyEmpty + (kb & 1), Threads);
                    commit_keys(kb + 2);
                    publish_keys(kb & 1);
                }
                if (next_values) {
                    find_value_shift(kb + 1);
                    causal_prompt_ws_bar_sync(kValueEmpty, Threads);
                    decode_values();
                    publish_values(0);
                }
            }
            return;
        }
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(
                     Fp8Pv ? kCausalPromptWsFp8ConsumerRegs : kCausalPromptWsConsumerRegs)
                 : "memory");

    // Each warp rotates and encodes its own 16 rows, keeping the two rows each lane owns in the
    // MMA C layout as register scales: one per G64 group, or with Fp8Pv one per row.
    float q_scale_r[2][Fp8Pv ? 1 : Groups];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = warp * 16 + r;
        const int token  = q0 + row;
        const bool valid = row < rows;
        float q_values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_values[k] =
                valid ? __bfloat162float(
                            q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, token)])
                      : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
        float row_max = 0.0f;
        if constexpr (Fp8Pv) {
#pragma unroll
            for (int k = 0; k < 8; ++k) { row_max = fmaxf(row_max, fabsf(q_values[k])); }
            row_max = warp_max(row_max, FullMask);
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            const float mx  = Fp8Pv ? row_max : warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs  = mx > 0.0f ? mx / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_prompt_store_byte_swizzled(q_i8, row, d0 + 32,
                                              kv_cache_int8_quant_code(x1, inv));
            if (gid == (r & 7)) { q_scale_r[r >> 3][Fp8Pv ? 0 : grp] = qs; }
        }
    }
    __syncwarp();

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // Tile r of group g holds dimensions 32g + 4n + r.
    float acc[VGroups][4][4];
#pragma unroll
    for (int b = 0; b < VGroups; ++b) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    // Float r of this lane's carried state (kCausalPromptWsStateFloats per lane, see the header).
    const auto state_slot = [&](int r) -> float& {
        return state[(static_cast<std::size_t>(linear) * kCausalPromptWsStateFloats + r) *
                         Consumers +
                     tid];
    };
    if ((state_flags & kCausalPromptWsLoadState) != 0) {
#pragma unroll
        for (int b = 0; b < VGroups; ++b) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[b][p][i] = state_slot((b * 4 + p) * 4 + i); }
            }
        }
        running_m0 = state_slot(VGroups * 16 + 0);
        running_m1 = state_slot(VGroups * 16 + 1);
        running_l0 = state_slot(VGroups * 16 + 2);
        running_l1 = state_slot(VGroups * 16 + 3);
    }

    unsigned pa[PVKs][4];
    // Fp8Pv: e4m3 probabilities per 32-key chunk, and the UE8M0 A scale this lane supplies: lane
    // 4g + 1 gives row g + 8's, every other lane row g's.
    unsigned pa8[Bc / 32][4];
    unsigned tile_scale_a = 0u;
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    bool tile_live    = false;

    // release_keys runs once the key stage is no longer read, before the softmax. Tile t of the
    // segment is key tile kb_begin + t.
    const auto qk_softmax = [&](int t, auto&& release_keys) {
        tile_live    = false;
        const int k0 = (kb_begin + t) * Bc;
        if (!warp_active || k0 > warp_max_qabs) {
            release_keys();
            return;
        }
        const unsigned char* base  = stage_base(t);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const float* ks_s = reinterpret_cast<const float*>(base + kCausalPromptI8FastTileBytes);

        float score[QKNt][4];
        int chain[QKNt][4]; // Fp8Pv: each score's int32 sum over all four groups
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            chain[nt][0] = chain[nt][1] = chain[nt][2] = chain[nt][3] = 0;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                if constexpr (Fp8Pv) {
                    c0 = chain[nt][0];
                    c1 = chain[nt][1];
                    c2 = chain[nt][2];
                    c3 = chain[nt][3];
                }
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                if constexpr (Fp8Pv) {
                    chain[nt][0] = c0;
                    chain[nt][1] = c1;
                    chain[nt][2] = c2;
                    chain[nt][3] = c3;
                } else {
                    const float2 ks = load_vec<float2>(ks_s + grp * Bc + nt * 8 + 2 * lid);
                    const float ks0 = ks.x;
                    const float ks1 = ks.y;
                    const float qs0 = q_scale_r[0][grp];
                    const float qs1 = q_scale_r[1][grp];
                    score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }
        }
        if constexpr (Fp8Pv) {
            // At most 256 * 127 * 127 < 2^24: each sum converts exactly.
            const float qs0 = q_scale_r[0][0];
            const float qs1 = q_scale_r[1][0];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float2 ks = load_vec<float2>(ks_s + nt * 8 + 2 * lid);
                score[nt][0]    = (qs0 * ks.x) * static_cast<float>(chain[nt][0]);
                score[nt][1]    = (qs0 * ks.y) * static_cast<float>(chain[nt][1]);
                score[nt][2]    = (qs1 * ks.x) * static_cast<float>(chain[nt][2]);
                score[nt][3]    = (qs1 * ks.y) * static_cast<float>(chain[nt][3]);
            }
        }

        release_keys();
        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx_ftz(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx_ftz(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        if constexpr (Fp8Pv) {
            // Against the running maximum lowered by o whole octaves, times 2^8: the largest
            // probability lies in (128, 256]. A row without visible keys keeps o = 0.
            const auto octaves = [&](float bm, float nm_scaled) {
                if (bm == -CUDART_INF_F) { return 0; }
                return min(kCausalPromptWsFp8MaxOctaves,
                           static_cast<int>(floorf(nm_scaled - bm * scale_l2)));
            };
            const int o0     = octaves(bm0, nm0_scaled);
            const int o1     = octaves(bm1, nm1_scaled);
            const float off0 =
                static_cast<float>(kCausalPromptWsFp8ProbabilityShift + o0) - nm0_scaled;
            const float off1 =
                static_cast<float>(kCausalPromptWsFp8ProbabilityShift + o1) - nm1_scaled;
            // 2^-(o + shift) as UE8M0 and FP32 share their exponent field.
            const unsigned e0 =
                static_cast<unsigned>(127 - kCausalPromptWsFp8ProbabilityShift - o0);
            const unsigned e1 =
                static_cast<unsigned>(127 - kCausalPromptWsFp8ProbabilityShift - o1);
            tile_scale_a = (lane & 1) != 0 ? e1 : e0;
            float p[QKNt][4];
            float bl0 = 0.0f;
            float bl1 = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                p[nt][0] = exp2_approx_ftz(__fmaf_rn(score[nt][0], scale_l2, off0));
                p[nt][1] = exp2_approx_ftz(__fmaf_rn(score[nt][1], scale_l2, off0));
                p[nt][2] = exp2_approx_ftz(__fmaf_rn(score[nt][2], scale_l2, off1));
                p[nt][3] = exp2_approx_ftz(__fmaf_rn(score[nt][3], scale_l2, off1));
                bl0 += p[nt][0] + p[nt][1];
                bl1 += p[nt][2] + p[nt][3];
            }
            // Chunk c holds score tiles 4c..4c+3: registers 0/2 row gid, 1/3 row gid + 8, and per
            // register two adjacent tiles' key pair {2t, 2t + 1}.
#pragma unroll
            for (int c = 0; c < Bc / 32; ++c) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int nt = 4 * c + 2 * half;
                    pa8[c][2 * half + 0] =
                        causal_prompt_ws_e4m3x2(p[nt][0], p[nt][1]) |
                        (causal_prompt_ws_e4m3x2(p[nt + 1][0], p[nt + 1][1]) << 16);
                    pa8[c][2 * half + 1] =
                        causal_prompt_ws_e4m3x2(p[nt][2], p[nt][3]) |
                        (causal_prompt_ws_e4m3x2(p[nt + 1][2], p[nt + 1][3]) << 16);
                }
            }
            running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0 * __uint_as_float(e0 << 23));
            running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1 * __uint_as_float(e1 << 23));
            tile_live  = true;
            return;
        }
        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15.
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx_ftz(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx_ftz(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx_ftz(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx_ftz(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            if constexpr (!Fp8Pv) {
                pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
                pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
            }
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);
        tile_live  = true;
    };

    // Fp8Pv: n8 tile j of group g holds dimensions 32g + 8j + ((n >> 1) & 3) + 4 (n & 1); one b8
    // ldmatrix serves two tiles of one 32-key chunk.
    const auto pv_fp8 = [&](int stage) {
        if (!tile_live) { return; }
        const unsigned char* const tile = values8 + stage * kCausalPromptWsFp8ValueStageBytes;
#pragma unroll
        for (int grp = 0; grp < VGroups; ++grp) {
            // The group's power-of-two unscale as the UE8M0 B scale of all eight columns.
            const unsigned scale_b = __float_as_uint(value_unscale_s[stage][grp]) >> 23;
            float h[4][4];
#pragma unroll
            for (int c = 0; c < Bc / 32; ++c) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    unsigned b[4];
                    causal_prompt_ws_ldmatrix_b8_x2_t(
                        b[0], b[1], b[2], b[3],
                        smem_addr(tile + causal_prompt_ws_fp8_value_offset(32 * c + lane,
                                                                            2 * grp + pair)));
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        float* const x = h[2 * pair + r];
                        if (c == 0) {
                            mma_fp8_e4m3_scaled_zero(x[0], x[1], x[2], x[3], pa8[c][0], pa8[c][1],
                                                     pa8[c][2], pa8[c][3], b[r], b[r + 2],
                                                     tile_scale_a, scale_b);
                        } else {
                            mma_fp8_e4m3_scaled(x[0], x[1], x[2], x[3], pa8[c][0], pa8[c][1],
                                                pa8[c][2], pa8[c][3], b[r], b[r + 2],
                                                tile_scale_a, scale_b);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[grp][r][0] = __fmaf_rn(acc[grp][r][0], tile_alpha0, h[r][0]);
                acc[grp][r][1] = __fmaf_rn(acc[grp][r][1], tile_alpha0, h[r][1]);
                acc[grp][r][2] = __fmaf_rn(acc[grp][r][2], tile_alpha1, h[r][2]);
                acc[grp][r][3] = __fmaf_rn(acc[grp][r][3], tile_alpha1, h[r][3]);
            }
        }
    };

    const auto pv = [&](int value_shift) {
        if (!tile_live) { return; }
        const float unscale = ldexpf(1.0f, value_shift);
#pragma unroll
        for (int grp = 0; grp < VGroups; ++grp) {
            unsigned h[4][2] = {};
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int row    = j * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int column = grp * 32 + pair * 16 + ((lane >> 4) & 1) * 8;
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                  smem_addr(&values[row * D + causal_prompt_swz(row, column)]));
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        causal_prompt_i8_fast_mma_f16_acc(h[2 * pair + r][0], h[2 * pair + r][1],
                                                          pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                                          b[2 * r], b[2 * r + 1]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float2 a0 = __half22float2(load_vec<__half2>(&h[r][0]));
                float2 a1 = __half22float2(load_vec<__half2>(&h[r][1]));
                if (value_shift != 0) {
                    a0.x *= unscale;
                    a0.y *= unscale;
                    a1.x *= unscale;
                    a1.y *= unscale;
                }
                acc[grp][r][0] = __fmaf_rn(acc[grp][r][0], tile_alpha0, a0.x);
                acc[grp][r][1] = __fmaf_rn(acc[grp][r][1], tile_alpha0, a0.y);
                acc[grp][r][2] = __fmaf_rn(acc[grp][r][2], tile_alpha1, a1.x);
                acc[grp][r][3] = __fmaf_rn(acc[grp][r][3], tile_alpha1, a1.y);
            }
        }
    };

    // Every warp takes every barrier, including warps whose rows end before a tile. Iteration t
    // runs the previous tile's PV, then this tile's QK and softmax. A segment none of whose tiles
    // this CTA's rows see takes no barrier and keeps the loaded state.
    const int wg = warp >> 2;
#pragma unroll 1
    for (int t = 0; t < sweep; ++t) {
        if (t > 0) {
            if constexpr (Fp8Pv) {
                const int stage = (t - 1) & 1;
                causal_prompt_ws_bar_sync(value_full(stage) + wg, kHalf);
                pv_fp8(stage);
                if (t + 1 < sweep) {
                    causal_prompt_ws_bar_arrive(value_empty(stage), Threads);
                }
            } else {
                causal_prompt_ws_bar_sync(kValueFull + wg, kHalf);
                pv(value_shift_s);
                causal_prompt_ws_bar_arrive(kValueEmpty, Threads);
            }
        }
        causal_prompt_ws_bar_sync(kKeyFull + 2 * (t & 1) + wg, kHalf);
        qk_softmax(t, [&] {
            if (t + 2 < sweep) {
                causal_prompt_ws_bar_arrive(kKeyEmpty + (t & 1), Threads);
            }
        });
    }
    if (sweep > 0) {
        if constexpr (Fp8Pv) {
            const int stage = (sweep - 1) & 1;
            causal_prompt_ws_bar_sync(value_full(stage) + wg, kHalf);
            pv_fp8(stage);
        } else {
            causal_prompt_ws_bar_sync(kValueFull + wg, kHalf);
            pv(value_shift_s);
        }
    }
    if ((state_flags & kCausalPromptWsStoreState) != 0) {
#pragma unroll
        for (int b = 0; b < VGroups; ++b) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { state_slot((b * 4 + p) * 4 + i) = acc[b][p][i]; }
            }
        }
        state_slot(VGroups * 16 + 0) = running_m0;
        state_slot(VGroups * 16 + 1) = running_m1;
        state_slot(VGroups * 16 + 2) = running_l0;
        state_slot(VGroups * 16 + 3) = running_l1;
        return;
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    if constexpr (Fp8Pv) {
        // Every consumer warp is past its last PV and the producers have returned, so the CTA's
        // shared memory is free: the rows are rounded to BF16 into it (lane t holds dimensions
        // 32g + 8j + t and 32g + 8j + t + 4 of tile j of its two rows), then leave as 16-byte
        // chunks with the gate read the same way.
        constexpr int RowElements = kCausalPromptWsOutRowElements;
        __nv_bfloat16* const staged = reinterpret_cast<__nv_bfloat16*>(smem_raw);
        causal_prompt_ws_bar_sync(kEpilogue, Consumers);
#pragma unroll
        for (int grp = 0; grp < VGroups; ++grp) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = grp * 32 + 8 * j + lid;
                staged[row0 * RowElements + d]     = __float2bfloat16_rn(acc[grp][j][0] * inv_l0);
                staged[row0 * RowElements + d + 4] = __float2bfloat16_rn(acc[grp][j][1] * inv_l0);
                staged[row1 * RowElements + d]     = __float2bfloat16_rn(acc[grp][j][2] * inv_l1);
                staged[row1 * RowElements + d + 4] = __float2bfloat16_rn(acc[grp][j][3] * inv_l1);
            }
        }
        causal_prompt_ws_bar_sync(kEpilogue, Consumers);
        constexpr int Chunks = D / 8;
#pragma unroll 4
        for (int c = tid; c < Br * Chunks; c += Consumers) {
            const int row   = c / Chunks;
            const int d0    = (c - row * Chunks) * 8;
            const int token = q0 + row;
            if (token >= width) { continue; }
            uint4 v = token < tokens ? load_vec<uint4>(&staged[row * RowElements + d0])
                                     : make_uint4(0u, 0u, 0u, 0u);
            const std::int64_t index = causal_prompt_q_index<Geometry>(q_head, d0, token);
            if (gate != nullptr) {
                const uint4 g = load_vec<uint4>(&gate[index]);
                v = make_uint4(gated_pair(v.x, g.x), gated_pair(v.y, g.y), gated_pair(v.z, g.z),
                               gated_pair(v.w, g.w));
            }
            store_vec(&out[index], v);
        }
    } else {
        // Lane (g, t) owns the eight contiguous dimensions 32g + 8t .. 32g + 8t + 7 of its rows.
#pragma unroll
        for (int grp = 0; grp < VGroups; ++grp) {
            const int d0 = grp * 32 + 8 * lid;
            store_row(row0, d0, acc[grp][0][0] * inv_l0, acc[grp][1][0] * inv_l0,
                      acc[grp][2][0] * inv_l0, acc[grp][3][0] * inv_l0);
            store_row(row0, d0 + 4, acc[grp][0][1] * inv_l0, acc[grp][1][1] * inv_l0,
                      acc[grp][2][1] * inv_l0, acc[grp][3][1] * inv_l0);
            store_row(row1, d0, acc[grp][0][2] * inv_l1, acc[grp][1][2] * inv_l1,
                      acc[grp][2][2] * inv_l1, acc[grp][3][2] * inv_l1);
            store_row(row1, d0 + 4, acc[grp][0][3] * inv_l1, acc[grp][1][3] * inv_l1,
                      acc[grp][2][3] * inv_l1, acc[grp][3][3] * inv_l1);
        }
    }
}

} // namespace ninfer::ops
