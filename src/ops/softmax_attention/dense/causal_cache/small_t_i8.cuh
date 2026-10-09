#pragma once

// ninfer::ops - split-KV causal small-T attention, INT8 KV-cache partial kernel.
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q and newly appended K receive the same fixed
//     register-only D256 rotation before G64 quantization. K stays int8 in the cache and is read
//     straight into smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV uses FP16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     an FP16 tile, then FP16 PV MMA runs with FP32 accumulation. V is still read from DRAM
//     as int8, so the bandwidth win is kept. SmallTPv::Fp8 rounds both operands to e4m3 for
//     the sm_120a block-scaled MMA instead (see the kernel's comment).
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/f16 MMA helpers, the reducer) lives in causal_attention_small_t.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

// How the kernel forms P.V: FP16 operands summed in FP32 (Fp32) or, per key tile, in FP16 (Fp16),
// or e4m3 operands through the block-scaled MMA (Fp8: packed values and 32-key blocks only).
enum class SmallTPv : std::uint8_t { Fp32, Fp16, Fp8 };

// Two 16 x 16 e4m3 blocks (keys k .. k + 15 and k + 16 .. k + 31 of 16 dimensions), transposed:
// r0 / r2 are the m16n8k32 B fragment of the first eight dimensions and r1 / r3 of the second.
__device__ __forceinline__ void causal_small_t_ldmatrix_b8_x2_t(unsigned& r0, unsigned& r1,
                                                                unsigned& r2, unsigned& r3,
                                                                unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ unsigned causal_small_t_e4m3x2(float lo, float hi) {
    return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ unsigned causal_small_t_e4m3x2(unsigned half2_bits) {
    return __nv_cvt_halfraw2_to_fp8x2(load_vec<__half2_raw>(&half2_bits), __NV_SATFINITE,
                                      __NV_E4M3);
}

// Decode-specialized producer/consumer kernel for up to 48 query rows. One producer warp per
// m16 row tile computes QK + online softmax, while all CTA warps partition the
// tile's 256-wide PV output. This keeps each thread's PV accumulator at 16, 32,
// or 64 floats instead of 128 and uses otherwise-idle warps for useful output
// work.
//
// Q has a dedicated shared tile so producers can reload one 64-dimension group
// at a time. K/V codes and scales are staged asynchronously; non-producer warps
// dequantize V while producers execute QK. After both consume the code tile, the
// next K/V tile is prefetched into the same arena while the current PV runs.
// PackedValues selects the rk8v4 half-width value plane, for both the fused append this kernel
// performs and the value staging it consumes.
//
// Keys selects a packed key plane: rk4v4's 4-bit Lloyd-Max indices or rk4v4-e8's E8-snapped int4
// codes (half the INT8 key bytes), or rk2v4-e8's E8 root codes (a quarter). The shared K tile and
// the whole QK path are unchanged. Packed keys for the next tile are loaded into registers at the
// top of each key-loop iteration (load_k_tile), stay in flight across QK, the V dequant and PV,
// are expanded to their INT8 codes (kv_cache_packed_key_expand16), and are stored into the tile
// just before the tile barrier; which warps expand them, and when, depends on the row-tile count
// (see LoaderKeys). The K tile is idle during PV, so this adds no shared memory and no barrier.
//
// EarlyFetch moves the rest of the next tile the same way: its value codes, both scale rows and,
// for INT8 keys, the key codes are loaded into registers at the top of the iteration and stored
// into the arena after PV. Otherwise they are issued as cp.async after the tile barrier, where
// their DRAM latency overlaps PV alone.
//
// Pv == Fp8 runs each 32-key tile's PV as one block-scaled e4m3 m16n8k32 MMA per n8 tile, which
// on sm_120 accumulates in FP32 at four times the rate of the FP16 MMA with an FP32 accumulator.
// The producers take each row's probabilities against the running maximum lowered by the whole
// octaves that bring it within one of the row's maximum in the tile, times 2^8, and round them to
// e4m3: the tile's largest lies in (128, 256]. The octaves and 2^-8 return through that row's
// UE8M0 A scale. Every decoded value of a tile is rounded to e4m3 under its 32-dimension group's
// power of two, which maps eight times the group's largest scale in the tile into [128, 256), and
// its inverse is the B scale of the group's n8 tiles.
// The MMA starts from zero and one FFMA per element rescales the running sum and adds that FP32
// tile sum, as the FP32 MMA addition truncates when the running sum dwarfs the products. The row
// sums keep the unrounded FP32 probabilities. Products below about 2^-14 of their tile's largest
// probability or value become subnormal.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput,
          bool PackedValues = false, KvKeyCoding Keys = KvKeyCoding::Int8,
          SmallTPv Pv = SmallTPv::Fp32, int QkSplit = 1, bool EarlyFetch = false>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void causal_attention_small_t_i8_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t wave_splits,
        float scale, float* partial_acc, float* partial_m, float* partial_l) {
    constexpr bool PackedKeys          = Keys != KvKeyCoding::Int8;
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kCausalHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kKVCacheInt8Groups;
    constexpr int GroupKc              = kKVCacheInt8Group / 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    // A split stages up to 64 page IDs, enough for a 262,144-key window in this split
    // geometry; a split that spans more pages reads the block table directly.
    constexpr int PageIds         = 64;
    // QkSplit producer warps share each row tile, each scoring its slice of the key tile; their row
    // maxima meet in shared memory. With every warp a producer, all of them dequantize V first.
    constexpr int ProducerWarps   = RowTiles * QkSplit;
    constexpr int ProducerThreads = ProducerWarps * 32;
    constexpr int VLoaderThreads  = Threads - ProducerThreads;
    constexpr int QKNtWarp        = QKNt / QkSplit;
    constexpr bool SharedDequant  = VLoaderThreads == 0;
    constexpr float Log2E         = 1.4426950408889634074f;
    constexpr unsigned FullMask   = 0xffffffffu;
    constexpr bool Fp8Pv          = Pv == SmallTPv::Fp8;
    // A UE8M0 A scale 2^-(octaves + 8) stays normal.
    constexpr int Fp8MaxOctaves   = 118;

    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == Groups * GroupKc);
    static_assert(QkSplit == 1 || (QkSplit == 2 && DynamicArena && QKNt % 2 == 0));
    static_assert(ProducerWarps <= Wc);
    static_assert(!Fp8Pv || (PackedValues && Bc == 32));

    // Keep Q in a compact dedicated tile so the producer can reload one
    // 64-dimension group at a time instead of carrying all eight fragments in
    // registers across the whole kernel. The main arena holds K i8, V i8, and
    // V FP16 during the key loop.
    __shared__ __align__(16) std::int8_t q_s[Br * D];
    __shared__ __align__(16) std::int8_t static_r_s[DynamicArena ? 16 : 4 * Bc * D];
    extern __shared__ __align__(16) std::int8_t dynamic_r_s[];
    std::int8_t* r_s     = DynamicArena ? dynamic_r_s : static_r_s;
    std::int8_t* q_i8    = q_s;
    float* q_scale_tmp   = reinterpret_cast<float*>(r_s);
    std::int8_t* k_i8    = r_s;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);
    std::int8_t* v_i8    = r_s + Bc * D;
    __half* v_f16        = reinterpret_cast<__half*>(r_s + 2 * Bc * D);
    // Fp8Pv: the e4m3 value tile, 256-byte key rows of 16-byte chunks XOR-swizzled by the key.
    std::uint8_t* v_e4m3 = reinterpret_cast<std::uint8_t*>(r_s + 2 * Bc * D);
    __shared__ __align__(16) __half p_s[Br * Bc];
    // Fp8Pv: the e4m3 probabilities fill the first half of p_s (32-byte rows whose two 16-byte
    // halves swap every four rows), each row's UE8M0 A scale the second.
    std::uint8_t* p_e4m3      = reinterpret_cast<std::uint8_t*>(p_s);
    std::uint8_t* p_scale_e8  = reinterpret_cast<std::uint8_t*>(p_s) + Br * Bc;
    const auto p_e4m3_offset  = [](int row, int col) {
        return row * Bc + ((((col >> 4) ^ (row >> 2)) & 1) << 4) + (col & 15);
    };
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc * Groups];
    __shared__ __align__(16) __half v_scale_s[Bc * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float split_max_s[QkSplit > 1 ? QkSplit * Br : 1];
    // A packed value row occupies only the leading D/2 bytes of its D-byte slot, so that key's
    // eight G32 scales live in the upper half of its own slot. Placing them per key rather than in
    // one block matters because the staged rows keep the full D stride. Static shared memory is
    // untouched, which this kernel needs: it already sits at the 48 KiB static ceiling. Fp8Pv
    // staggers the 16-byte scale blocks of eight consecutive keys, so one warp reads 32 keys' scales
    // without bank conflicts.
    const auto value_scale_row = [&](int key_l) -> __half* {
        if constexpr (PackedValues) {
            return reinterpret_cast<__half*>(v_i8 + key_l * D + D / 2 +
                                             (Fp8Pv ? 16 * (key_l & 7) : 0));
        } else {
            return &v_scale_s[key_l * Groups];
        }
    };

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = 0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile, wave_splits);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count    = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    const bool staged_pages = page_count <= PageIds;
    if (staged_pages) {
        for (int page = tid; page < page_count; page += Threads) {
            physical_pages_s[page] = block_table[first_page + page];
        }
    }
    const auto page_at = [&](int key) {
        return staged_pages ? physical_pages_s[(key >> kPagedKVPageShift) - first_page]
                            : block_table[key >> kPagedKVPageShift];
    };

    if constexpr (CacheInput::writes_cache) {
        // Decompose H256 as H4 over four independently transformed H64 groups. The existing
        // (token, group) warp schedule computes all H64 fragments in parallel; the FP32 main arena
        // is the exchange point for the final H4 stage. This retains the complete transform's
        // butterfly/rounding order while shortening the fused append critical path. V stays in
        // the original coordinates and retains the existing G64 codec.
        float* k_h64_s = reinterpret_cast<float*>(r_s);
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();

        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);

            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }

            const float kv0    = k_out[0];
            const float kv1    = k_out[1];
            const float vv0    = __bfloat162float(input.v[src0]);
            const float vv1    = __bfloat162float(input.v[src1]);
            kv_cache_i8_family_store_key_group<Geometry, Keys>(
                cache_k_i8, cache_k_scale, physical_page, kv_head, grp, page_offset, lane, kv0,
                kv1);
            // The vv0 lanes span dimensions [64g, 64g+32) and the vv1 lanes the next 32, so the
            // packed coding's two G32 groups fall out of the lane assignment directly.
            const float vamax_lo = warp_max(fabsf(vv0), FullMask);
            const float vamax_hi = warp_max(fabsf(vv1), FullMask);
            const auto v_quant   = PackedValues
                                       ? kv_cache_int4_quant_params(vamax_lo)
                                       : kv_cache_int8_quant_params(fmaxf(vamax_lo, vamax_hi));
            const auto v_quant_hi =
                PackedValues ? kv_cache_int4_quant_params(vamax_hi) : v_quant;
            if constexpr (PackedValues) {
                // A packed byte holds the adjacent dimension pair, which spans lanes l and l^1.
                const std::int8_t c0 = kv_cache_int4_quant_code(vv0, v_quant.inverse_scale);
                const std::int8_t c1 = kv_cache_int4_quant_code(vv1, v_quant_hi.inverse_scale);
                const int partner0   = __shfl_xor_sync(FullMask, static_cast<int>(c0), 1);
                const int partner1   = __shfl_xor_sync(FullMask, static_cast<int>(c1), 1);
                if ((lane & 1) == 0) {
                    auto* packed_v = reinterpret_cast<std::uint8_t*>(cache_v_i8);
                    const std::int64_t packed_base = kv_cache_int4_value_code_index<Geometry>(
                        physical_page, kv_head, grp * (kKVCacheInt8Group / 2), page_offset);
                    packed_v[packed_base + (lane >> 1)] =
                        kv_cache_int4_pack(c0, static_cast<std::int8_t>(partner0));
                    packed_v[packed_base + (lane >> 1) + 16] =
                        kv_cache_int4_pack(c1, static_cast<std::int8_t>(partner1));
                }
            } else {
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            }
            if (lane == 0) {
                if constexpr (PackedValues) {
                    cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 2 * grp, page_offset)] = v_quant.scale;
                    cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 2 * grp + 1, page_offset)] = v_quant_hi.scale;
                } else {
                    cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset)] = v_quant.scale;
                }
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_small_t_store_byte_swizzled(q_i8, row, d0, DB16,
                                               kv_cache_int8_quant_code(x0, inv));
            causal_small_t_store_byte_swizzled(q_i8, row, d1, DB16,
                                               kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Replacing these runtime-lane __shfl_sync broadcasts (and the k/v-scale ones below) with
    // direct shared-memory reads was tried on the 3090: it removed every SHFL.IDX but grew the
    // stack frame of the production T=8 Bc=32 rk4v4 instantiations (16/48 -> 48/80 bytes at the
    // 168-register cap) and measured -0.1 to -0.9% end to end, so the shuffles stay.
    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < ProducerWarps) {
        const int producer_row0 = (warp % RowTiles) * 16 + gid;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            q_scale_r0[g] =
                producer_row0 < RowCount ? q_scale_tmp[producer_row0 * Groups + g] : 0.0f;
            q_scale_r1[g] =
                producer_row0 + 8 < RowCount ? q_scale_tmp[(producer_row0 + 8) * Groups + g] : 0.0f;
        }
    }
    __syncthreads();

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    // Packed keys are loaded into registers at the top of each iteration for the next tile, so the
    // loads are in flight across QK, the V dequant and PV. Who expands them depends on the shape.
    //
    // With two row tiles the producers' QK is the long phase and at least half the CTA are
    // V-loader warps, so the loaders own the keys: they expand them to INT8 codes right after their
    // V dequant, inside the producers' shadow, and after PV only store them. That took MTP3 decode
    // on the 27B from 1.5% behind rk8v4 at 32K to 0.5%. With one row tile QK is short and the
    // loaders become the critical path, and with three the loaders are too few (three of six warps);
    // both measured 6-12% slower in the attention op that way, so every warp instead expands and
    // stores its share after PV.
    constexpr int KChunks     = Bc * (D / 16);
    constexpr bool LoaderKeys = PackedKeys && RowTiles == 2 && 2 * VLoaderThreads >= Threads;
    constexpr int KeyThreads  = LoaderKeys ? VLoaderThreads : Threads;
    constexpr int KChunksPerKeyThread = (KChunks + KeyThreads - 1) / KeyThreads;
    using KeyChunk = KvPackedKeyChunk<PackedKeys ? Keys : KvKeyCoding::Lloyd4>;
    KeyChunk k_packed[PackedKeys ? KChunksPerKeyThread : 1];
    int4 k_codes[LoaderKeys ? KChunksPerKeyThread : 1];
    const bool key_thread = LoaderKeys ? tid >= ProducerThreads : true;
    const int key_tid     = LoaderKeys ? tid - ProducerThreads : tid;
    const int loader_tid  = tid - ProducerThreads;
    auto load_k_tile = [&](int tile_k0, int physical_page) {
        if constexpr (PackedKeys) {
            if (!key_thread) { return; }
#pragma unroll
            for (int i = 0; i < KChunksPerKeyThread; ++i) {
                const int chunk = key_tid + i * KeyThreads;
                k_packed[i]     = KeyChunk{};
                if (chunk < KChunks) {
                    const int key_l = chunk / (D / 16);
                    const int d     = (chunk - key_l * (D / 16)) * 16;
                    const int key   = tile_k0 + key_l;
                    if (key >= split_start && key < split_end) {
                        const std::int64_t off = kv_cache_packed_key_chunk_index<Geometry, Keys>(
                            physical_page, kv_head, d, key & kPagedKVPageMask);
                        // A plain load: the fused append above may have written these bytes in
                        // this launch, so the read-only path is not an option.
                        k_packed[i] = *reinterpret_cast<const KeyChunk*>(
                            reinterpret_cast<const std::uint8_t*>(cache_k_i8) + off);
                    }
                }
            }
        }
    };
    // Loader-owned keys only: expand into registers ahead of the store.
    auto expand_k_tile = [&]() {
        if constexpr (LoaderKeys) {
#pragma unroll
            for (int i = 0; i < KChunksPerKeyThread; ++i) {
                k_codes[i] = kv_cache_packed_key_expand16<Keys>(k_packed[i]);
            }
        }
    };
    // Chunks outside the split keep whatever code their placeholder expands to: their scales are
    // staged as zero and their scores masked, exactly as the zero-filled INT8 chunks are.
    auto commit_k_tile = [&]() {
        if constexpr (PackedKeys) {
            if (!key_thread) { return; }
#pragma unroll
            for (int i = 0; i < KChunksPerKeyThread; ++i) {
                const int chunk = key_tid + i * KeyThreads;
                if (chunk < KChunks) {
                    const int key_l = chunk / (D / 16);
                    const int dc    = chunk - key_l * (D / 16);
                    int4 codes;
                    if constexpr (LoaderKeys) {
                        codes = k_codes[i];
                    } else {
                        codes = kv_cache_packed_key_expand16<Keys>(k_packed[i]);
                    }
                    store_vec(&k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2], codes);
                }
            }
        }
    };

    constexpr int CodeChunks          = Bc * (D / 16);
    constexpr int CodeChunksPerThread = (CodeChunks + Threads - 1) / Threads;
    using ValueChunk                  = std::conditional_t<PackedValues, int2, int4>;
    using ValueScaleRow               = std::conditional_t<PackedValues, int4, int2>;
    static_assert(!EarlyFetch || Bc <= Threads);
    ValueChunk v_next[EarlyFetch ? CodeChunksPerThread : 1];
    int4 k_next[EarlyFetch && !PackedKeys ? CodeChunksPerThread : 1];
    int2 k_scale_next{};
    ValueScaleRow v_scale_next{};
    auto fetch_next_tile = [&](int tile_k0, int page) {
        if constexpr (EarlyFetch) {
            if (tid < Bc) {
                const int key = tile_k0 + tid;
                k_scale_next  = int2{};
                v_scale_next  = ValueScaleRow{};
                if (key >= split_start && key < split_end) {
                    const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                        page, kv_head, 0, key & kPagedKVPageMask);
                    k_scale_next = load_vec<int2>(&cache_k_scale[off]);
                    if constexpr (PackedValues) {
                        v_scale_next = load_vec<int4>(&cache_v_scale[kv_cache_int4_value_scale_index<
                            Geometry>(page, kv_head, 0, key & kPagedKVPageMask)]);
                    } else {
                        v_scale_next = load_vec<int2>(&cache_v_scale[off]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < CodeChunksPerThread; ++i) {
                const int chunk = tid + i * Threads;
                v_next[i]       = ValueChunk{};
                if constexpr (!PackedKeys) { k_next[i] = int4{}; }
                if (chunk < CodeChunks) {
                    const int key_l = chunk / (D / 16);
                    const int d     = (chunk - key_l * (D / 16)) * 16;
                    const int key   = tile_k0 + key_l;
                    if (key >= split_start && key < split_end) {
                        if constexpr (PackedValues) {
                            v_next[i] = load_vec<int2>(
                                reinterpret_cast<const std::uint8_t*>(cache_v_i8) +
                                kv_cache_int4_value_code_index<Geometry>(page, kv_head, d >> 1,
                                                                         key & kPagedKVPageMask));
                        }
                        const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                            page, kv_head, d, key & kPagedKVPageMask);
                        if constexpr (!PackedValues) { v_next[i] = load_vec<int4>(&cache_v_i8[off]); }
                        if constexpr (!PackedKeys) { k_next[i] = load_vec<int4>(&cache_k_i8[off]); }
                    }
                }
            }
        }
    };
    auto store_next_tile = [&]() {
        if constexpr (EarlyFetch) {
            if (tid < Bc) {
                store_vec(&k_scale_s[tid * Groups], k_scale_next);
                store_vec(value_scale_row(tid), v_scale_next);
            }
#pragma unroll
            for (int i = 0; i < CodeChunksPerThread; ++i) {
                const int chunk = tid + i * Threads;
                if (chunk < CodeChunks) {
                    const int key_l = chunk / (D / 16);
                    const int dc    = chunk - key_l * (D / 16);
                    store_vec(&v_i8[key_l * D + (PackedValues ? dc * 8 : dc * 16)], v_next[i]);
                    if constexpr (!PackedKeys) {
                        store_vec(&k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2],
                                  k_next[i]);
                    }
                }
            }
        }
    };

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                if constexpr (PackedValues) {
                    const std::int64_t voff = kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 0, key & kPagedKVPageMask);
                    ninfer::ops::cp_async<16>(value_scale_row(key_l), &cache_v_scale[voff]);
                } else {
                    ninfer::ops::cp_async<8>(value_scale_row(key_l), &cache_v_scale[off]);
                }
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                if constexpr (PackedValues) {
                    store_vec(value_scale_row(key_l), make_int4(0, 0, 0, 0));
                } else {
                    store_vec(value_scale_row(key_l), make_int2(0, 0));
                }
            }
        }
        if constexpr (PackedKeys) {
            // Keys arrive through load_k_tile; this stages only the packed value plane.
            static_assert(PackedValues, "packed keys pair with packed values");
#pragma unroll 1
            for (int chunk = tid; chunk < KChunks; chunk += Threads) {
                const int key_l = chunk / (D / 16);
                const int d     = (chunk - key_l * (D / 16)) * 16;
                const int key   = tile_k0 + key_l;
                if (key >= split_start && key < split_end) {
                    const std::int64_t off = kv_cache_int4_value_code_index<Geometry>(
                        physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
                    ninfer::ops::cp_async<8>(
                        &v_i8[key_l * D + (d >> 1)],
                        reinterpret_cast<const std::uint8_t*>(cache_v_i8) + off);
                } else {
                    store_vec(&v_i8[key_l * D + (d >> 1)], make_int2(0, 0));
                }
            }
        } else {
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                const int key   = tile_k0 + key_l;
                if (key >= split_start && key < split_end) {
                    const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                        physical_page, kv_head, d, key & kPagedKVPageMask);
                    std::int8_t* dst = &k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2];
                    ninfer::ops::cp_async<16>(dst, &cache_k_i8[off]);
                    if constexpr (PackedValues) {
                        // Sixteen dimensions occupy eight packed bytes, staged into the leading
                        // half of the same value slot so the arena footprint matches the INT8
                        // coding.
                        const std::int64_t voff = kv_cache_int4_value_code_index<Geometry>(
                            physical_page, kv_head, d >> 1, key & kPagedKVPageMask);
                        ninfer::ops::cp_async<8>(
                            &v_i8[key_l * D + (d >> 1)],
                            reinterpret_cast<const std::uint8_t*>(cache_v_i8) + voff);
                    } else {
                        ninfer::ops::cp_async<16>(&v_i8[key_l * D + d], &cache_v_i8[off]);
                    }
                } else {
                    std::int8_t* dst = &k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2];
                    store_vec(dst, make_int4(0, 0, 0, 0));
                    if constexpr (PackedValues) {
                        store_vec(&v_i8[key_l * D + (d >> 1)], make_int2(0, 0));
                    } else {
                        store_vec(&v_i8[key_l * D + d], make_int4(0, 0, 0, 0));
                    }
                }
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = page_at(first_tile);
    load_k_tile(first_tile, physical_page);
    issue_kv_tile(first_tile, physical_page);
    if (key_thread) { expand_k_tile(); }
    commit_k_tile();
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    // Fp8Pv: every warp reads the tile's value scales (lane = key) for each 32-dimension group's
    // power of two 2^e that takes eight times the group's largest scale into [128, 256). Byte g of
    // v_powers holds group g's e + 15, the FP16 exponent field of 2^e; the UE8M0 B scale 2^-e is
    // 142 minus it. Keys outside the split carry zero scales.
    unsigned v_powers[2] = {0u, 0u};
    const auto value_power = [&]() {
        if constexpr (Fp8Pv) {
            const uint4 s             = load_vec<uint4>(value_scale_row(lane));
            const unsigned pairs[4]   = {s.x, s.y, s.z, s.w};
#pragma unroll
            for (int word = 0; word < 2; ++word) {
                unsigned packed = 0u;
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const unsigned pair = pairs[2 * word + (byte >> 1)];
                    // Nonnegative FP16 values order as their bit patterns.
                    const unsigned largest =
                        __reduce_max_sync(FullMask, (byte & 1) != 0 ? pair >> 16 : pair & 0xffffu);
                    const int e = largest == 0u ? 0 : min(15, 19 - static_cast<int>(largest >> 10));
                    packed |= static_cast<unsigned>(e + 15) << (8 * byte);
                }
                v_powers[word] = packed;
            }
        }
    };
    const auto group_power = [&](int group) -> unsigned {
        return __byte_perm(v_powers[0], v_powers[1], static_cast<unsigned>(group)) & 0xffu;
    };

    const auto dequant_v = [&](int k0, int first, int stride) {
#pragma unroll 1
        for (int chunk = first; chunk < Bc * (D / 8); chunk += stride) {
            const int key_l = chunk / (D / 8);
            const int dc    = chunk - key_l * (D / 8);
            const int d     = dc * 8;
            const int key   = k0 + key_l;
            if constexpr (Fp8Pv) {
                // Eight dimensions to eight e4m3 bytes, half of one swizzled 16-byte chunk.
                std::uint8_t* dst =
                    v_e4m3 + key_l * D + ((((d >> 4) ^ key_l) & 15) << 4) + (d & 8);
                uint2 codes = make_uint2(0u, 0u);
                if (key >= split_start && key < split_end) {
                    const __half up =
                        __ushort_as_half(static_cast<unsigned short>(group_power(d >> 5) << 10));
                    const __half vs = __hmul(value_scale_row(key_l)[d >> 5], up);
                    const int4 h    = kv_cache_int4_dequant_f16x8_from(
                        reinterpret_cast<const std::uint8_t*>(v_i8) + key_l * D + (d >> 1), vs);
                    codes.x = causal_small_t_e4m3x2(static_cast<unsigned>(h.x)) |
                              (causal_small_t_e4m3x2(static_cast<unsigned>(h.y)) << 16);
                    codes.y = causal_small_t_e4m3x2(static_cast<unsigned>(h.z)) |
                              (causal_small_t_e4m3x2(static_cast<unsigned>(h.w)) << 16);
                }
                store_vec(dst, codes);
                continue;
            }
            __half* dst     = &v_f16[key_l * D + causal_small_t_tc_swz(key_l, d)];
            if (key >= split_start && key < split_end) {
                // The lanes of one value group read the same scale: a shared-memory broadcast,
                // where a shuffle from a computed lane costs an out-of-line call.
                const int grp   = PackedValues ? (d >> 5) : (d >> 6);
                const __half vs = value_scale_row(key_l)[grp];
                // f16 loaders, because dst is __half* and this tile feeds mma_f16. The bf16
                // loaders are the same width, so using them here compiles and silently
                // reinterprets every value.
                if constexpr (PackedValues) {
                    store_vec(dst, kv_cache_int4_dequant_f16x8_from(
                                       reinterpret_cast<const std::uint8_t*>(v_i8) + key_l * D +
                                           (d >> 1),
                                       vs));
                } else {
                    store_vec(dst, kv_cache_int8_dequant_f16x8_from(&v_i8[key_l * D + d],
                                                                    __half2float(vs)));
                }
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0        = first_tile + kb * Bc;
        const bool has_next = kb + 1 < key_blocks;
        if constexpr (PackedKeys || EarlyFetch) {
            // Prefetch the next tile now; see load_k_tile and EarlyFetch.
            if (has_next) {
                const int next_k0 = k0 + Bc;
                const int next_page =
                    (next_k0 & kPagedKVPageMask) == 0 ? page_at(next_k0) : physical_page;
                load_k_tile(next_k0, next_page);
                fetch_next_tile(next_k0, next_page);
            }
        }

        value_power();
        if constexpr (SharedDequant) { dequant_v(k0, tid, Threads); }

        // Producer warps (QkSplit per row tile) produce P and alpha while any remaining warps
        // stream/dequant V.
        if (warp < ProducerWarps) {
            const int producer_row_base = (warp % RowTiles) * 16;
            const int key_part          = warp / RowTiles;
            const int nt_base           = key_part * QKNtWarp;
            __half* p_sw                = &p_s[producer_row_base * Bc];
            float score[QKNtWarp][4];
#pragma unroll
            for (int nt = 0; nt < QKNtWarp; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(
                            &q_b16[(producer_row_base + a_rowoff) * DB16 +
                                   causal_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNtWarp; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = (nt_base + nt) * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(
                            bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + causal_small_t_tc_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = (nt_base + nt) * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    const float ka  = __half2float(k_scale_s[keya * Groups + g]);
                    const float kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNtWarp; ++nt) {
                const int col0 = (nt_base + nt) * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                        ? score[nt][3] * scale
                        : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if constexpr (QkSplit > 1) {
                // Both key parts of a row tile need the row maximum over the whole key tile; only
                // the producer warps meet at this barrier.
                if (lid == 0) {
                    split_max_s[key_part * Br + row0] = bm0;
                    split_max_s[key_part * Br + row1] = bm1;
                }
                asm volatile("bar.sync 1, %0;\n" ::"r"(ProducerThreads) : "memory");
#pragma unroll
                for (int part = 0; part < QkSplit; ++part) {
                    bm0 = fmaxf(bm0, split_max_s[part * Br + row0]);
                    bm1 = fmaxf(bm1, split_max_s[part * Br + row1]);
                }
            }

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

            float bl0 = 0.0f, bl1 = 0.0f;
            // Fp8Pv: whole octaves that bring each row within one of its tile maximum, plus 2^8.
            int shift0 = 0, shift1 = 0;
            if constexpr (Fp8Pv) {
                const auto octaves = [&](float bm, float nm) {
                    if (bm == -CUDART_INF_F) { return 0; }
                    return min(Fp8MaxOctaves, static_cast<int>(floorf((nm - bm) * Log2E)));
                };
                shift0 = 8 + octaves(bm0, nm0);
                shift1 = 8 + octaves(bm1, nm1);
            }
#pragma unroll
            for (int nt = 0; nt < QKNtWarp; ++nt) {
                const int col0  = (nt_base + nt) * 8 + 2 * lid;
                if constexpr (Fp8Pv) {
                    const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                          ? exp2_approx(fmaf(score[nt][0] - nm0, Log2E, shift0))
                                          : 0.0f;
                    const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                          ? exp2_approx(fmaf(score[nt][1] - nm0, Log2E, shift0))
                                          : 0.0f;
                    const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                          ? exp2_approx(fmaf(score[nt][2] - nm1, Log2E, shift1))
                                          : 0.0f;
                    const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                          ? exp2_approx(fmaf(score[nt][3] - nm1, Log2E, shift1))
                                          : 0.0f;
                    bl0 += p00 + p01;
                    bl1 += p10 + p11;
                    store_vec(&p_e4m3[p_e4m3_offset(producer_row_base + gid, col0)],
                              static_cast<unsigned short>(causal_small_t_e4m3x2(p00, p01)));
                    store_vec(&p_e4m3[p_e4m3_offset(producer_row_base + gid + 8, col0)],
                              static_cast<unsigned short>(causal_small_t_e4m3x2(p10, p11)));
                    continue;
                }
                const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                      : 0.0f;
                const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                      : 0.0f;
                const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                      : 0.0f;
                const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                // The swizzle keeps a column's position within its eight-column chunk, so the even
                // column and its successor are one aligned FP16 pair.
                store_vec(&p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)],
                          __floats2half2_rn(p00, p01));
                store_vec(&p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)],
                          __floats2half2_rn(p10, p11));
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            // Each key part keeps its own share of the row sums; the maxima are shared, so the
            // shares add up after the loop.
            if constexpr (Fp8Pv) {
                // 2^-shift as UE8M0 and as FP32 share their exponent field.
                const unsigned e0 = static_cast<unsigned>(127 - shift0);
                const unsigned e1 = static_cast<unsigned>(127 - shift1);
                l0 = fmaf(l0, alpha0, bl0 * __uint_as_float(e0 << 23));
                l1 = fmaf(l1, alpha1, bl1 * __uint_as_float(e1 << 23));
                if (lid == 0 && key_part == 0) {
                    p_scale_e8[row0] = static_cast<std::uint8_t>(e0);
                    p_scale_e8[row1] = static_cast<std::uint8_t>(e1);
                }
            } else {
                l0 = l0 * alpha0 + bl0;
                l1 = l1 * alpha1 + bl1;
            }
            m0 = nm0;
            m1 = nm1;
            if (lid == 0 && key_part == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if constexpr (!SharedDequant) {
            dequant_v(k0, loader_tid, VLoaderThreads);
            if (kb + 1 < key_blocks) { expand_k_tile(); }
        }
        __syncthreads();

        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = page_at(next_k0);
            }
            if constexpr (!EarlyFetch) { issue_kv_tile(next_k0, physical_page); }
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __half* p_consumer          = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
        if constexpr (Fp8Pv) {
            // Lane 4g supplies row g's A scale and lane 4g + 1 row g + 8's.
            const unsigned scale_a = p_scale_e8[consumer_row_base + gid + ((lane & 1) << 3)];
            unsigned pa[4];
            ldmatrix_x4(pa[0], pa[1], pa[2], pa[3],
                        smem_addr(&p_e4m3[p_e4m3_offset(consumer_row_base + a_rowoff,
                                                        a_coloff * 2)]));
#pragma unroll
            for (int pair = 0; pair < PVNtPerWarp / 2; ++pair) {
                const int chunk = consumer_slice * (PVNtPerWarp / 2) + pair;
                // Both n8 tiles of a 16-dimension chunk lie in its 32-dimension group.
                const unsigned v_scale_b = 142u - group_power(chunk >> 1);
                unsigned b[4];
                causal_small_t_ldmatrix_b8_x2_t(
                    b[0], b[1], b[2], b[3],
                    smem_addr(v_e4m3 + lane * D + (((chunk ^ lane) & 15) << 4)));
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float h[4];
                    mma_fp8_e4m3_scaled_zero(h[0], h[1], h[2], h[3], pa[0], pa[1], pa[2], pa[3],
                                             b[r], b[r + 2], scale_a, v_scale_b);
                    float* const a = acc[2 * pair + r];
                    a[0]           = __fmaf_rn(a[0], alpha0, h[0]);
                    a[1]           = __fmaf_rn(a[1], alpha0, h[1]);
                    a[2]           = __fmaf_rn(a[2], alpha1, h[2]);
                    a[3]           = __fmaf_rn(a[3], alpha1, h[3]);
                }
            }
        }
#pragma unroll
        for (int n = 0; n < (Fp8Pv ? 0 : PVNtPerWarp); ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < (Fp8Pv ? 0 : PVNtPerWarp); ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
            // PvF16: this tile's products are summed in FP16 (P is at most one after the running
            // maximum and a tile spans at most 64 keys) and folded into the FP32 accumulator once.
            unsigned tile_acc[2] = {0u, 0u};
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(
                        &p_consumer[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                if constexpr (Pv == SmallTPv::Fp16) {
                    mma_f16_f16acc(tile_acc[0], tile_acc[1], pf[0], pf[1], pf[2], pf[3], vf[0],
                                   vf[1]);
                } else {
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                            pf[3], vf[0], vf[1]);
                }
            }
            if constexpr (Pv == SmallTPv::Fp16) {
                const float2 top    = __half22float2(*reinterpret_cast<const __half2*>(&tile_acc[0]));
                const float2 bottom = __half22float2(*reinterpret_cast<const __half2*>(&tile_acc[1]));
                acc[n][0] += top.x;
                acc[n][1] += top.y;
                acc[n][2] += bottom.x;
                acc[n][3] += bottom.y;
            }
        }
        if (has_next) {
            commit_k_tile();
            if constexpr (EarlyFetch) {
                store_next_tile();
            } else {
                ninfer::ops::cp_wait<0>();
            }
        }
        __syncthreads();
    }

    if constexpr (QkSplit > 1) {
        // Fold the key parts' row sums into part zero's warps.
        __syncthreads();
        if (warp < ProducerWarps && lid == 0) {
            const int row0 = (warp % RowTiles) * 16 + gid;
            split_max_s[(warp / RowTiles) * Br + row0]     = l0;
            split_max_s[(warp / RowTiles) * Br + row0 + 8] = l1;
        }
        __syncthreads();
        if (warp < RowTiles && lid == 0) {
            const int row0 = warp * 16 + gid;
#pragma unroll
            for (int part = 1; part < QkSplit; ++part) {
                l0 += split_max_s[part * Br + row0];
                l1 += split_max_s[part * Br + row0 + 8];
            }
        }
    }
    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        // The e4m3 value tile keeps the dimensions in order, so both PV forms leave lane t
        // dimensions 2t and 2t + 1 of n8 tile n.
        const int d0     = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const auto store = [&](int row, float x, float y) {
            if (row >= RowCount) { return; }
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(x, y);
        };
        store(row0, acc[n][0], acc[n][1]);
        store(row1, acc[n][2], acc[n][3]);
    }
}

} // namespace ninfer::ops
