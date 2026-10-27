#pragma once

// Shared by the bridge's translation units only: the vendored ggml headers and the device facts the
// matrix kernel's host side needs.

#include "ggml_bridge.h"
#include "lt_fp4.cuh"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::gguf::detail {

struct DeviceFacts {
    int cc         = 0; // ggml's compute capability: 100 * major + 10 * minor
    int sm_count   = 0;
    std::size_t shared_per_block_optin = 0;
};

[[nodiscard]] const DeviceFacts& device_facts();

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gguf ") + what + ": " + cudaGetErrorString(status));
    }
}

[[nodiscard]] inline ggml_type to_ggml(GgmlType type) { return static_cast<ggml_type>(type); }

// Column tiles instantiated for the matrix kernel. A product is launched with the smallest number of
// column tiles these allow, which is where ggml's own selection lands for them as well.
inline constexpr int kMatrixColumnTiles[] = {16, 32, 64, 128};

// The matrix kernel's activation carries one block_q8_1_mmq per 128 values per column; ggml over-reads
// up to one widest column tile past the end, so the plane keeps that much slack.
inline constexpr std::size_t kMatrixActivationSlack = 128 * sizeof(block_q8_1_mmq);

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns);

template <ggml_type type>
bool moe_matrix_fits_impl(int rows, int k, bool tail);

template <ggml_type type>
void moe_matrix_product_impl(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                             const MoeRouting& routing, int max_active, int pairs, int max_columns,
                             bool tail, const void* activation, float* out, cudaStream_t stream);

// Where and how the vector kernel (ggml_bridge_vec.cuh) writes its rows.
struct VecStore {
    __nv_bfloat16* bf16;
    float* f32;
    std::int64_t column_stride;
    Epilogue epilogue;
    const float* gate;
    std::int64_t gate_column_stride;
    int activation; // GateActivation of the gate product and of a fused [gate; up] product
};

struct VecArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows; // output rows; a fused [gate; up] parent holds 2 * rows
    int k;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    VecStore out;
};

// The tile kernel (ggml_bridge_tile.cuh): a product over the planar tile activation.
struct TileArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    int columns;
    int columns_pad;       // columns rounded up to the 128-column tile
    const std::int8_t* qs; // [columns][k]: int8, or e4m3 bytes in the MXFP8 form
    const float* d;        // [k / 32][columns_pad]: FP32, or 2^E in the MXFP8 form
    VecStore out;
    bool mxfp8; // the MXFP8 form (a 120a build): e4m3 weights and activations, UE8M0 scales
};

template <ggml_type type>
void tile_launch(const TileArgs& args, cudaStream_t stream);

// The narrow kernel (ggml_bridge_narrow.cuh): a product over the narrow activation.
struct NarrowArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    int columns;
    const std::int8_t* qs; // [columns][k] in the kernel's order
    const float* d;        // [k / 64][columns rounded up to 8][2]
    VecStore out;
};

template <ggml_type type>
void narrow_launch(const NarrowArgs& args, cudaStream_t stream);

// The warps of rows TW the narrow kernel runs a product with, of 2, 3 and 4: the one of least
// modeled time. An SM runs its share of the units in turn (units = rows / 16 TW rounded up, dealt
// round-robin: the most an SM gets is units / SMs rounded up), a unit's cost per K value its 16 TW
// rows' weight bytes plus a fifth of its T activation bytes (read from L2, which every CTA shares).
// The fifth is fitted on the RTX 5060 Ti: over the 27B's shapes at 6-32 columns the rule's time is
// within 2.2% of each cell's fastest configuration, count-weighted (65% at worst, IQ3_XXS 7168 x
// 5120 at 24 columns); activation weights from none to four fifths come within 0.2% of it.
inline int narrow_row_warps(int block_bytes, int rows, int columns, int sm_count) {
    int best               = 4;
    std::int64_t best_cost = 0;
    for (int tw = 4; tw >= 2; --tw) {
        const std::int64_t units  = (rows + 16 * tw - 1) / (16 * tw);
        const std::int64_t rounds = (units + sm_count - 1) / sm_count;
        // 1280 x (16 TW block_bytes / 256 + T / 5)
        const std::int64_t cost = rounds * (80 * tw * block_bytes + 256 * columns);
        if (tw == 4 || cost < best_cost) {
            best      = tw;
            best_cost = cost;
        }
    }
    return best;
}

// The two-pass MXFP8 product's image pass (ggml_bridge_tile.cuh image_kernel).
struct ImageArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    std::uint8_t* image;  // e4m3 [rows][k]
    std::uint8_t* scales; // UE8M0 exponents at lt_scale_offset(row, group, k / 32)
};

template <ggml_type type>
void image_launch(const ImageArgs& args, cudaStream_t stream);

// The NVFP4 two-pass product's image pass (ggml_bridge_tile.cuh image_fp4_kernel): the rows'
// decoded values times the weight's global -> e2m1 codes [rows][k / 2] (a byte's first value in
// its low nibble) and E4M3 scales at lt_scale_offset(row, group, k / 16) (fp4_encode16), and
// alpha = {1 / (global g_x), 0, 1} for the GEMM, g_x the activation's global.
struct Fp4ImageArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    std::uint8_t* image;
    std::uint8_t* scales;
    float global;           // lt_fp4_global of the weight's max |w|
    const unsigned* x_amax; // the activation's max |x| (FP32 bits)
    float* alpha;
    // The rows' scale-choice table: fp4_encode16's candidate index of each 16-value group, two bits,
    // a 32-value slice's two in a nibble (its first group in the low bits), slice s of the rows (row
    // major) in byte s / 2, the even slice in the low nibble. 128-byte aligned, read in whole
    // 128-byte blocks of 256 slices. With it the image encodes each group's named candidate
    // (fp4_encode16_pick), the search's result when the table holds the search's winners under
    // this global; null: the image searches.
    const std::uint8_t* choices = nullptr;
};

template <ggml_type type>
void image_fp4_launch(const Fp4ImageArgs& args, cudaStream_t stream);

// The table pass: the search's winners of rows [0, rows) under `global` into `table` (the layout
// of Fp4ImageArgs::choices, whole 128-byte blocks written); no image, scales or alpha.
template <ggml_type type>
void fp4_choices_launch(const Fp4ImageArgs& args, std::uint8_t* table, cudaStream_t stream);

// max |w| over the decoded values of rows [0, rows) into *amax as FP32 bits (atomicMax; the caller
// zeroes it).
struct AmaxArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows;
    int k;
    unsigned* amax;
};

template <ggml_type type>
void amax_launch(const AmaxArgs& args, cudaStream_t stream);


// The vector kernel's MoE form (ggml_bridge_vec.cuh moe_kernel).
struct MoeVecArgs {
    const std::uint8_t* const* first;  // expert table (the gate table when fused)
    const std::uint8_t* const* second; // fused: the up table
    std::int64_t row_bytes;
    int rows;
    int k;
    const std::int32_t* bounds;
    const std::int32_t* sorted;
    const std::int32_t* active;
    const std::int32_t* active_count;
    int column_group;
    int per_token;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    __nv_bfloat16* out_bf16;
    float* out_f32;
    unsigned long long* weighted;
    const float* weights;
    const float* gate;
    int activation; // GateActivation of `gate` and of the fused form
};

template <ggml_type type>
void moe_launch(const MoeVecArgs& args, int max_active, int chunk, bool fused, cudaStream_t stream);
template <ggml_type type>
void moe_decode_up_launch(const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_down_launch(const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_prep_launch(const MoeDecodePrepArgs& args, cudaStream_t stream);

// Whether slice_dequantize serves the type (every type but IQ1_M).
template <ggml_type type>
inline constexpr bool kSliceDequantize = type != GGML_TYPE_IQ1_M;

// Rows of a GGUF matrix to dst_t through the vector kernels' slice decoders: one thread a 32-value
// slice, written as 16-byte stores (ggml_bridge_vec.cuh).
template <ggml_type type, typename dst_t>
void slice_dequantize(const void* weight, std::int64_t row_bytes, int k, const std::int32_t* row_ids,
                      int rows, dst_t* out, std::int64_t out_row_stride, cudaStream_t stream);

// Byte offset of the scales inside a planar vector activation of `columns` x `k`.
[[nodiscard]] constexpr std::size_t vec_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 15) / 16 * 16;
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream);

// The rescoring pass of shortlist_argmax (ggml_bridge_vec.cuh): candidate i of column c, row
// rows[c * candidates + i] of the matrix, is dotted with column c of the q8_1 activation by the
// vector kernel's arithmetic, and the column's best (value, lower row) key is raised to it.
struct ShortlistRescoreArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int k;
    int candidates;
    const std::int8_t* qs;            // [columns][k]
    const half2* ds;                  // [columns][k / 32]
    const std::int32_t* rows;         // [columns][candidates]
    unsigned long long* best;         // [columns], zero before the pass
};

template <ggml_type type>
void shortlist_rescore_launch(const ShortlistRescoreArgs& args, int columns, cudaStream_t stream);

// Order-preserving integer keys: a larger key is a larger value, -0 and +0 one key, NaN the least.
// The 16-bit key of a BF16 value is at least 2 for every non-NaN value (-inf maps to 0x7F), 1 for
// NaN; 0 marks padding. The 32-bit key of an FP32 value is 0 for NaN.
__device__ __forceinline__ std::uint32_t bf16_order_key(std::uint16_t bits) {
    if ((bits & 0x7FFFu) > 0x7F80u) { return 1u; }
    if (bits == 0x8000u) { bits = 0; }
    return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

__device__ __forceinline__ std::uint32_t f32_order_key(float value) {
    std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) { return 0u; }
    if (bits == 0x80000000u) { bits = 0; }
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float vec_silu(float x) { return x / (1.0f + expf(-x)); }

// GateActivation: 0 SiLU, 1 tanh-approximated GELU.
__device__ __forceinline__ float vec_gate(float x, int activation) {
    if (activation == 0) { return vec_silu(x); }
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ void vec_store(const VecStore& out, int row, int column, float value) {
    const std::int64_t at = column * out.column_stride + row;
    if (out.epilogue == Epilogue::GateProduct) {
        value *= vec_gate(out.gate[column * out.gate_column_stride + row], out.activation);
    }
    if (out.f32 != nullptr) {
        out.f32[at] = out.epilogue == Epilogue::Accumulate ? out.f32[at] + value : value;
        return;
    }
    if (out.epilogue == Epilogue::Accumulate) { value += __bfloat162float(out.bf16[at]); }
    out.bf16[at] = __float2bfloat16(value);
}

// Epilogues compiled into the vector and tile kernels: vec_store's runtime selection costs
// registers (in the tile kernel spills in the two-scale main loop; in the vector kernel 123 against
// 95, 16 against 20 warps per SM), so the model's forms get their own instances and the rest
// kEpiGeneric.
enum Epi : int {
    kEpiF32      = 0, // FP32 Store
    kEpiBf16     = 1, // BF16 Store
    kEpiBf16Add  = 2, // BF16 Accumulate
    kEpiBf16Silu = 3, // BF16 GateProduct with SiLU
    kEpiGeneric  = 4, // vec_store
};

// What the epilogue reads at (row, column): the gate of a gated product, the output it accumulates
// into, nothing for a store.
template <int EPI>
__device__ __forceinline__ float epi_operand(const VecStore& out, int row, int column) {
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

// vec_store's arithmetic given the operand epi_operand read.
template <int EPI>
__device__ __forceinline__ void epi_finish(const VecStore& out, int row, int column, float value,
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

// The compiled epilogue of a destination.
[[nodiscard]] inline int store_epi(const VecStore& o) {
    if (o.epilogue == Epilogue::Store) { return o.f32 != nullptr ? kEpiF32 : kEpiBf16; }
    if (o.f32 != nullptr) { return kEpiGeneric; }
    if (o.epilogue == Epilogue::Accumulate) { return kEpiBf16Add; }
    if (o.epilogue == Epilogue::GateProduct && o.activation == 0) { return kEpiBf16Silu; }
    return kEpiGeneric;
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                         bool tail);

} // namespace ninfer::ops::gguf::detail
