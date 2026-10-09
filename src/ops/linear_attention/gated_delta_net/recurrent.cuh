#pragma once

#include "ops/common/bf16_vector.cuh"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

// FP16 state storage: four halves per lane, widened to FP32 for the recurrence and rounded back
// once per stored step.
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const __half* base,
                                             std::uint32_t dqk_base) {
    const uint2 bits = load_vec<uint2>(base + dqk_base);
    const float2 lo  = __half22float2(*reinterpret_cast<const __half2*>(&bits.x));
    const float2 hi  = __half22float2(*reinterpret_cast<const __half2*>(&bits.y));
    reg[0]           = lo.x;
    reg[1]           = lo.y;
    reg[2]           = hi.x;
    reg[3]           = hi.y;
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], __half* base,
                                              std::uint32_t dqk_base) {
    const __half2 lo = __floats2half2_rn(reg[0], reg[1]);
    const __half2 hi = __floats2half2_rn(reg[2], reg[3]);
    store_vec(base + dqk_base, make_uint2(*reinterpret_cast<const unsigned*>(&lo),
                                          *reinterpret_cast<const unsigned*>(&hi)));
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ float load_value_lane(const __nv_bfloat16* base, int lane,
                                                std::uint32_t dv_base) {
    return lane < kDvPerWarp ? __bfloat162float(base[dv_base + lane]) : 0.0f;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

template <class StateT = float>
struct DirectAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* state_read;
    StateT* state_write;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(0, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)coord;
        return token;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        return state_read + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ StateT* state_write_base(const RecurrentCoordinates& coord) const {
        return state_write + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <class StateT = float>
struct BatchUpdateAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* states_read;
    StateT* states_write;
    const std::int32_t* source_state_slots;
    const std::int32_t* destination_state_slots;
    __nv_bfloat16* out;
    head_map heads;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)token;
        return coord.batch;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states_read +
               static_cast<std::int64_t>(source_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ StateT* state_write_base(const RecurrentCoordinates& coord) const {
        return states_write +
               static_cast<std::int64_t>(destination_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

// Record and fold run the recurrence with kReplayRowsPerLane state rows of 16 columns per lane: lane
// 8 r + i of a warp holds its rows r kReplayRowsPerLane + q at columns 4 (i + 8 j) + c for j, c < 4.
// Those are the columns of lane i + 8 j in the layout of apply_gdn_transition, so a row dot product
// can reproduce that recurrence's arithmetic exactly (replay_row_dots), while four rows share every
// staged key and query load and every reduction step.
inline constexpr int kReplayRowsPerLane = 4;
inline constexpr int kReplayWarpRows    = kReplayRowsPerLane * kWarpSize / 8;
inline constexpr int kReplayTileRows    = kNumWarps * kReplayWarpRows;
inline constexpr int kReplayTiles       = kStateDim / kReplayTileRows;
static_assert(kStateDim % kReplayTileRows == 0);

struct ReplayCoordinates {
    int lane;
    int warp;
    int segment;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    // The first of the lane's kReplayRowsPerLane consecutive state rows.
    std::uint32_t row;
};

__device__ __forceinline__ ReplayCoordinates replay_coordinates(std::int32_t batch,
                                                                std::int32_t layer,
                                                                std::int32_t tile,
                                                                std::uint32_t qk_head) {
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    return {lane,
            warp,
            lane & 7,
            batch,
            layer,
            tile,
            blockIdx.x,
            qk_head,
            static_cast<std::uint32_t>(tile * kReplayTileRows + warp * kReplayWarpRows +
                                       (lane >> 3) * kReplayRowsPerLane)};
}

template <bool Masked, class StateT = float>
struct RecordAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const StateT* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ ReplayCoordinates coordinates() const {
        return replay_coordinates(
            static_cast<std::int32_t>(blockIdx.y), 0, static_cast<std::int32_t>(blockIdx.z),
            static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(blockIdx.x))));
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const ReplayCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const ReplayCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const ReplayCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    // The offsets of a column's query/key head and value head in their [column][head][dim]
    // tensors and records, and of its gate pair.
    __device__ __forceinline__ std::int64_t qk_offset(const ReplayCoordinates& coord,
                                                      std::int32_t token) const {
        return (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ std::int64_t value_offset(const ReplayCoordinates& coord,
                                                         std::int32_t token) const {
        return (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ std::int64_t gate_offset(const ReplayCoordinates& coord,
                                                        std::int32_t token) const {
        return column(coord, token) * heads.H_v + coord.value_head;
    }
};

// The per-layer shape of a fold. The number of layers is not part of it: each layer folds
// independently, so the launch grid takes the count from the records. That is what lets a model split
// into pipeline stages fold each stage's layers with the same kernel.
template <int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
    // Each state tile CTA publishes one 128-channel block of the conv history.
    static_assert(ValueHeads * kReplayTiles * 128 >= ConvChannels);
};

using FoldGeometry16x48 = FoldGeometry<16, 48, 10240>;
using FoldGeometry16x32 = FoldGeometry<16, 32, 8192>;

template <class Geometry, class StateT = float>
struct FoldAccess {
    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    StateT* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t layer_stride_columns;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ ReplayCoordinates coordinates() const {
        constexpr std::uint32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        return replay_coordinates(static_cast<std::int32_t>(blockIdx.y), layer_tile / kReplayTiles,
                                  layer_tile % kReplayTiles, blockIdx.x / kGroup);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const ReplayCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_column(const ReplayCoordinates& coord,
                                                          std::int32_t token) const {
        return static_cast<std::int64_t>(coord.layer) * layer_stride_columns +
               static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ std::int64_t state_offset(const ReplayCoordinates& coord,
                                                         std::int32_t slot) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const StateT*
    state_read_base(const ReplayCoordinates& coord) const {
        return recurrent_layer0 + state_offset(coord, rows.row[coord.batch].source_state_slot);
    }

    __device__ __forceinline__ StateT* state_write_base(const ReplayCoordinates& coord) const {
        return recurrent_layer0 + state_offset(coord, rows.row[coord.batch].destination_state_slot);
    }

    __device__ __forceinline__ std::int64_t key_offset(const ReplayCoordinates& coord,
                                                       std::int32_t token) const {
        return (record_column(coord, token) * Geometry::kQkHeads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ std::int64_t value_offset(const ReplayCoordinates& coord,
                                                         std::int32_t token) const {
        return (record_column(coord, token) * Geometry::kValueHeads + coord.value_head) *
               kStateDim;
    }

    __device__ __forceinline__ std::int64_t gate_offset(const ReplayCoordinates& coord,
                                                        std::int32_t token) const {
        return record_column(coord, token) * Geometry::kValueHeads + coord.value_head;
    }

    __device__ __forceinline__ void publish_final_conv_history(const ReplayCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * kReplayTiles + coord.tile;
        if (tile_block >= Geometry::kConvChannels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        const __nv_bfloat16* source_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        __nv_bfloat16* destination_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_column(coord, 0) * Geometry::kConvChannels + channel;

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = source_history[Geometry::kConvChannels];
            h1 = source_history[2LL * Geometry::kConvChannels];
            h2 = record[0];
        } else if (commit == 2) {
            h0 = source_history[2LL * Geometry::kConvChannels];
            h1 = record[0];
            h2 = record[Geometry::kConvChannels];
        } else {
            h0 = record[static_cast<std::int64_t>(commit - 3) * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(commit - 2) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(commit - 1) * Geometry::kConvChannels];
        }
        destination_history[0]                             = h0;
        destination_history[Geometry::kConvChannels]       = h1;
        destination_history[2LL * Geometry::kConvChannels] = h2;
    }
};

template <class StateT>
__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const StateT* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

template <class StateT>
__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 StateT* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

template <bool NormalizeInputs, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const float value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);

        apply_gdn_transition(state, key.value, value, gate.g, gate.beta);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
}

template <bool NormalizeInputs, class StateT>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const StateT* __restrict__ state_read,
                                 StateT* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const DirectAccess<StateT> access{q,     k,          v,   g,     beta, state_read,
                                      state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs>(state, access, coord, width);
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool NormalizeInputs, class StateT>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_batch_update_kernel(BatchUpdateAccess<StateT> access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs>(state, access, coord, 1);
    store_state_tile(state, access.state_write_base(coord), coord);
}

// A replay window: its normalized keys (and, for a record, queries), this CTA's value rows and the
// gates, staged before the recurrence. One warp normalizes each vector in the lane layout of
// normalize_qk_lane, so a CTA does it once per vector rather than every warp at every token.
inline constexpr int kReplayWindow = 8;

template <bool Queries>
struct ReplayStage {
    __align__(16) float key[kReplayWindow][kStateDim];
    __align__(16) float query[Queries ? kReplayWindow : 1][kStateDim];
    __align__(16) __nv_bfloat16 value[kReplayWindow][kReplayTileRows];
    float g[kReplayWindow];
    float beta[kReplayWindow];
};

using ReplayColumns = float[kQkPerLane][kQkPerLane];
using ReplayRows    = float[kReplayRowsPerLane][kQkPerLane][kQkPerLane];

// Each row dot product sums every column chunk as its lane in the apply_gdn_transition layout did,
// adds the chunks across bits 4 and 3 of that lane in registers and across bits 2..0 with three
// shuffles, in the order of warp_sum's butterfly: each result is bit-identical to that recurrence's.
__device__ __forceinline__ void replay_row_dots(const ReplayRows& state, const ReplayColumns& x,
                                                float (&out)[kReplayRowsPerLane]) {
#pragma unroll
    for (int q = 0; q < kReplayRowsPerLane; ++q) {
        float chunk[kQkPerLane];
#pragma unroll
        for (int j = 0; j < kQkPerLane; ++j) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += state[q][j][c] * x[j][c]; }
            chunk[j] = partial;
        }
        out[q] = (chunk[0] + chunk[2]) + (chunk[1] + chunk[3]);
    }
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
        for (int q = 0; q < kReplayRowsPerLane; ++q) {
            out[q] += __shfl_xor_sync(kFullWarpMask, out[q], offset, kWarpSize);
        }
    }
}

__device__ __forceinline__ void load_replay_columns(ReplayColumns& x, const float* row,
                                                    int segment) {
#pragma unroll
    for (int j = 0; j < kQkPerLane; ++j) {
        store_vec(x[j], load_vec<float4>(row + (segment + j * 8) * kQkPerLane));
    }
}

template <class StateT>
__device__ __forceinline__ void load_replay_rows(ReplayRows& state, const StateT* head,
                                                 const ReplayCoordinates& coord) {
    const StateT* base = head + static_cast<std::int64_t>(coord.row) * kStateDim;
#pragma unroll
    for (int q = 0; q < kReplayRowsPerLane; ++q) {
#pragma unroll
        for (int j = 0; j < kQkPerLane; ++j) {
            load_qk_lane(state[q][j], base + q * kStateDim,
                         static_cast<std::uint32_t>((coord.segment + j * 8) * kQkPerLane));
        }
    }
}

template <class StateT>
__device__ __forceinline__ void store_replay_rows(const ReplayRows& state, StateT* head,
                                                  const ReplayCoordinates& coord) {
    StateT* base = head + static_cast<std::int64_t>(coord.row) * kStateDim;
#pragma unroll
    for (int q = 0; q < kReplayRowsPerLane; ++q) {
#pragma unroll
        for (int j = 0; j < kQkPerLane; ++j) {
            store_qk_lane(state[q][j], base + q * kStateDim,
                          static_cast<std::uint32_t>((coord.segment + j * 8) * kQkPerLane));
        }
    }
}

// apply_gdn_transition, operation for operation, on the lane's rows.
__device__ __forceinline__ void replay_transition(ReplayRows& state, const ReplayColumns& key,
                                                  const float (&value)[kReplayRowsPerLane],
                                                  float g, float beta) {
    const float alpha = expf(g);
    float partial[kReplayRowsPerLane];
    replay_row_dots(state, key, partial);
#pragma unroll
    for (int q = 0; q < kReplayRowsPerLane; ++q) {
        const float delta = beta * (value[q] - alpha * partial[q]);
#pragma unroll
        for (int j = 0; j < kQkPerLane; ++j) {
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                state[q][j][c] = alpha * state[q][j][c] + delta * key[j][c];
            }
        }
    }
}

template <bool Masked, class StateT>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 3)
    recurrent_record_kernel(RecordAccess<Masked, StateT> access) {
    __shared__ ReplayStage<true> stage;
    const ReplayCoordinates coord = access.coordinates();
    ReplayRows state;
    // Verification leaves the persistent state unchanged: it was last written by the preceding
    // round's fold, which completes before this round's graph starts, and its slot comes from the
    // round's ingress. The tile therefore loads before waiting on the projection that feeds q/k/v.
    load_replay_rows(state, access.state_read_base(coord), coord);
    pdl::enter_streaming();
    const std::int32_t valid   = access.active_columns(coord);
    const int tid              = coord.warp * kWarpSize + coord.lane;
    const int stage_row        = static_cast<int>(coord.row) - coord.tile * kReplayTileRows;
    const auto dqk_base        = static_cast<std::uint32_t>(coord.lane * kQkPerLane);
    constexpr int kValueChunks = kReplayTileRows * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    constexpr int kWarpVectors = 2 * kReplayWindow / kNumWarps;
    // One CTA per query/key head group records the keys; the first tile of each value head records
    // its gates.
    const bool records_key = coord.tile == 0 &&
                             static_cast<int>(coord.value_head) % access.heads.group_size() == 0;
    // A wider single-row copy round runs its columns window by window over the state kept in
    // registers.
    for (std::int32_t base = 0; base < valid; base += kReplayWindow) {
        const std::int32_t count = min(kReplayWindow, valid - base);
        if (base != 0) { __syncthreads(); }
        for (int i = tid; i < count * kValueChunks; i += kWarpSize * kNumWarps) {
            const int token = i / kValueChunks;
            const int chunk = (i - token * kValueChunks) * 8;
            cp_async<16>(&stage.value[token][chunk],
                         access.v + access.value_offset(coord, base + token) +
                             coord.tile * kReplayTileRows + chunk);
        }
        cp_commit();
        if (tid < count) {
            const std::int64_t offset = access.gate_offset(coord, base + tid);
            const RawGatePair gate    = load_source_gate(access.g, access.beta, offset);
            stage.g[tid]              = gate.g;
            stage.beta[tid]           = gate.beta;
            if (coord.tile == 0) { access.gate_record[offset] = gate.bits; }
        }
        // Vector n of warp w is key w + 4 n while that is below count, else query w + 4 n - count.
        RawQkLane raw[kWarpVectors];
#pragma unroll
        for (int n = 0; n < kWarpVectors; ++n) {
            const int vector = coord.warp + n * kNumWarps;
            if (vector < 2 * count) {
                const bool key = vector < count;
                raw[n]         = load_raw_qk_lane(
                    (key ? access.k : access.q) +
                        access.qk_offset(coord, base + vector - (key ? 0 : count)),
                    dqk_base);
            }
        }
#pragma unroll
        for (int n = 0; n < kWarpVectors; ++n) {
            const int vector = coord.warp + n * kNumWarps;
            if (vector < 2 * count) {
                const bool key = vector < count;
                if (key && records_key) {
                    store_vec(access.key_record + access.qk_offset(coord, base + vector) + dqk_base,
                              raw[n].bits);
                }
                normalize_qk_lane<true>(raw[n].value, coord.lane);
                float* destination = key ? stage.key[vector] : stage.query[vector - count];
                store_vec(destination + dqk_base, load_vec<float4>(raw[n].value));
            }
        }
        cp_wait<0>();
        __syncthreads();

        for (std::int32_t token = 0; token < count; ++token) {
            const std::int64_t values = access.value_offset(coord, base + token) + coord.row;
            float value[kReplayRowsPerLane];
#pragma unroll
            for (int q = 0; q < kReplayRowsPerLane; ++q) {
                const __nv_bfloat16 bits = stage.value[token][stage_row + q];
                value[q]                 = __bfloat162float(bits);
                if (coord.segment == 0) { access.value_record[values + q] = bits; }
            }
            ReplayColumns key;
            load_replay_columns(key, stage.key[token], coord.segment);
            replay_transition(state, key, value, stage.g[token], stage.beta[token]);

            ReplayColumns query;
            load_replay_columns(query, stage.query[token], coord.segment);
            float attn_val[kReplayRowsPerLane];
            replay_row_dots(state, query, attn_val);
            if (coord.segment == 0) {
#pragma unroll
                for (int q = 0; q < kReplayRowsPerLane; ++q) {
                    access.out[values + q] = __float2bfloat16(attn_val[q] * access.scale);
                }
            }
        }
    }
    if (coord.segment == 0) {
        for (std::int32_t token = valid; token < access.width; ++token) {
            const std::int64_t values = access.value_offset(coord, token) + coord.row;
#pragma unroll
            for (int q = 0; q < kReplayRowsPerLane; ++q) {
                access.out[values + q] = __float2bfloat16(0.0f);
            }
        }
    }
}

template <class Geometry, class StateT>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 3)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Geometry, StateT> access) {
    __shared__ ReplayStage<false> stage;
    const ReplayCoordinates coord = access.coordinates();
    const std::int32_t valid      = access.active_columns(coord);
    if (valid == 0) { return; }
    ReplayRows state;
    load_replay_rows(state, access.state_read_base(coord), coord);
    const int tid              = coord.warp * kWarpSize + coord.lane;
    const int stage_row        = static_cast<int>(coord.row) - coord.tile * kReplayTileRows;
    const auto dqk_base        = static_cast<std::uint32_t>(coord.lane * kQkPerLane);
    constexpr int kValueChunks = kReplayTileRows * static_cast<int>(sizeof(__nv_bfloat16)) / 16;
    constexpr int kWarpKeys    = kReplayWindow / kNumWarps;
    for (std::int32_t base = 0; base < valid; base += kReplayWindow) {
        const std::int32_t count = min(kReplayWindow, valid - base);
        if (base != 0) { __syncthreads(); }
        for (int i = tid; i < count * kValueChunks; i += kWarpSize * kNumWarps) {
            const int token = i / kValueChunks;
            const int chunk = (i - token * kValueChunks) * 8;
            cp_async<16>(&stage.value[token][chunk],
                         access.value_record + access.value_offset(coord, base + token) +
                             coord.tile * kReplayTileRows + chunk);
        }
        cp_commit();
        if (tid < count) {
            const RawGatePair gate =
                load_record_gate(access.gate_record, access.gate_offset(coord, base + tid));
            stage.g[tid]    = gate.g;
            stage.beta[tid] = gate.beta;
        }
        RawQkLane raw[kWarpKeys];
#pragma unroll
        for (int n = 0; n < kWarpKeys; ++n) {
            const int key = coord.warp + n * kNumWarps;
            if (key < count) {
                raw[n] = load_raw_qk_lane(
                    access.key_record + access.key_offset(coord, base + key), dqk_base);
            }
        }
#pragma unroll
        for (int n = 0; n < kWarpKeys; ++n) {
            const int key = coord.warp + n * kNumWarps;
            if (key < count) {
                normalize_qk_lane<true>(raw[n].value, coord.lane);
                store_vec(stage.key[key] + dqk_base, load_vec<float4>(raw[n].value));
            }
        }
        cp_wait<0>();
        __syncthreads();

        for (std::int32_t token = 0; token < count; ++token) {
            float value[kReplayRowsPerLane];
#pragma unroll
            for (int q = 0; q < kReplayRowsPerLane; ++q) {
                value[q] = __bfloat162float(stage.value[token][stage_row + q]);
            }
            ReplayColumns key;
            load_replay_columns(key, stage.key[token], coord.segment);
            replay_transition(state, key, value, stage.g[token], stage.beta[token]);
        }
    }
    store_replay_rows(state, access.state_write_base(coord), coord);
    access.publish_final_conv_history(coord, valid);
}

} // namespace ninfer::ops::detail::gated_delta_net
