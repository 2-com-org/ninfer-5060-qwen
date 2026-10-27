#pragma once

// Plain entry points into the vendored ggml CUDA kernels (third_party/ggml-quants). The ggml headers
// define their own CUDA_CHECK and friends, so only the bridge's own translation unit includes them;
// everything here is raw pointers and cudaStream_t.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf {

// The block type's id in a GGUF tensor directory (ggml_type).
enum class GgmlType : int {
    Q4_0    = 2,
    Q5_0    = 6,
    Q8_0    = 8,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    IQ1_M   = 29,
    Q2_0    = 42,
};

struct BlockShape {
    int elements = 0;
    int bytes    = 0;
};

[[nodiscard]] BlockShape block_shape(GgmlType type);

// Streaming multiprocessors of the current device (cached per device).
[[nodiscard]] int multiprocessor_count();

// Whether the integer tensor-core prefill kernel exists for the type. IQ1_M has none (nor does it in
// llama.cpp); its wide products dequantize and take a BF16 GEMM.
[[nodiscard]] bool has_matrix_kernel(GgmlType type);

// Columns the vector kernel takes in one launch; wider products take the matrix kernel.
inline constexpr int kMaxVectorColumns = 8;

// Activation for the vector kernel: q8_1 blocks of 32 values, [columns][k / 32]. The vector and MoE
// vector products take it 32-byte aligned (a block's values are one 256-bit load) and refuse others.
[[nodiscard]] std::size_t vector_activation_bytes(int k, int columns);
// Its layout: int8 values [columns][k] from byte 0, then half2 (d, sum) pairs [columns][k / 32]
// from this byte offset; d = max|x| / 127 over the 32 values, sum their (unquantized) total.
[[nodiscard]] std::size_t vector_activation_scales_offset(int k, int columns);
// `input_columns` (optional, [k]) makes element c of each column x[input_columns[c]].
void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

enum class Epilogue : int {
    Store,       // out = y
    Accumulate,  // out = out + y
    GateProduct, // out = act(gate) * y, gate read from a float plane
};

// The activation a gated product applies to its gate: SiLU (SwiGLU) or tanh-approximated GELU
// (GeGLU, Gemma 4).
enum class GateActivation : int {
    Silu     = 0,
    GeluTanh = 1,
};

struct VectorOutput {
    __nv_bfloat16* bf16 = nullptr; // exactly one of bf16 / f32
    float* f32          = nullptr;
    std::int64_t column_stride = 0; // elements between consecutive columns
    Epilogue epilogue          = Epilogue::Store;
    const float* gate          = nullptr; // GateProduct only
    std::int64_t gate_column_stride = 0;
    GateActivation activation       = GateActivation::Silu; // GateProduct only
};

// out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for rows [0, rows) and columns [0, columns),
// columns <= kMaxVectorColumns. `weight` points at row 0; rows are `row_bytes` apart.
void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream);

// --- Shortlist argmax (coarse proposal heads) --------------------------------------------------
// The argmax of a matrix's rows over the candidates a coarse head ranks highest. For column c of
// the coarse logits L (BF16, element r of column c at coarse[c * coarse_column_stride + r]):
//   C_c    = the `candidates` rows r < rows with the largest L[r, c], a tie to the lower row;
//   y[r, c] = vector_product's BF16 Store of W_r . x_c (the q8_1 activation of x, bit for bit);
//   out[c] = ids[r*] (r* without ids), r* the lowest row of C_c with the largest y[r, c],
// where W is `rows` rows of `type` and x BF16 [columns][k] (column c at x + c * k). Only the rows of
// C_c are read from W, so it may live in mapped host memory. L is finite (a NaN ranks below every
// other value); 1 <= candidates <= kMaxShortlistCandidates <= rows; at most
// kMaxShortlistMerged / candidates tiles of kShortlistTileRows rows. No state.
inline constexpr int kMaxShortlistCandidates = 64;
inline constexpr int kShortlistTileRows      = 2048;
inline constexpr int kMaxShortlistMerged     = 8192;
[[nodiscard]] std::size_t shortlist_argmax_bytes(int rows, int k, int columns, int candidates);
void shortlist_argmax(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                      const __nv_bfloat16* coarse, std::int64_t coarse_column_stride,
                      const __nv_bfloat16* x, int columns, int candidates,
                      const std::int32_t* ids, std::int32_t* out, void* workspace,
                      cudaStream_t stream);

// out[r, c] = act(gate_r . x_c) * (up_r . x_c) where gate row r and up row r are rows r and
// rows + r of one parent of this type.
void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream,
                   GateActivation gate_activation = GateActivation::Silu);

// --- Narrow products ---------------------------------------------------------------------------
// The narrow kernel (ggml_bridge_narrow.cuh): out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for up
// to kMaxNarrowColumns columns on integer tensor cores, in the vector kernel's arithmetic (its q8_1
// activation; each 32-value integer dot exact, scaled by the weight's and the activation's factors
// in FP32), so the two differ in FP32 summation order only. IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S
// and Q2_K on sm_120, K a whole number of 1024-value steps, any rows.
inline constexpr int kMaxNarrowColumns = 32;
[[nodiscard]] bool has_narrow_kernel(GgmlType type, int k);
// Whether a product of `columns` takes it rather than the vector or the matrix kernel: from
// narrow_min_columns(type) to kMaxNarrowColumns.
[[nodiscard]] int narrow_min_columns(GgmlType type);
[[nodiscard]] bool takes_narrow_kernel(GgmlType type, int k, int columns);
// Its activation: int8 values [columns][k] in the kernel's order from byte 0 -- per column, per
// pair p of 32-value slices, the values 8t..8t+7 of slice 2p and then of slice 2p + 1 for
// t = 0..3 -- and from narrow_activation_scales_offset FP32 d [k / 64][columns rounded up to 8][2]
// (slice 2p + i of column c at [p][c][i], zero for the padding columns); d = max|x| / 127 over the
// 32 values and each value x / d rounded, the vector activation's values and d (d the exact value
// of its FP16 rounding). 16-byte aligned.
[[nodiscard]] std::size_t narrow_activation_bytes(int k, int columns);
[[nodiscard]] std::size_t narrow_activation_scales_offset(int k, int columns);
// `input_columns` as for the vector activation.
void quantize_narrow_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);
// `weight` points at row 0 (any alignment); `has_narrow_kernel(type, k)`, 1 <= columns <=
// kMaxNarrowColumns, rows of at most 16 MiB (the kernel's copy offsets are 32-bit within a unit).
void narrow_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream);
// The warps of 16 rows (2, 3 or 4) whose configuration narrow_product runs `rows` x `columns` of
// `type` with on this device: each width has one configuration per count.
[[nodiscard]] int narrow_row_warps(GgmlType type, int rows, int columns);

// Activation for the matrix kernel, laid out for the type's scale/sum schedule. Types that share a
// schedule share an activation, see matrix_activation_layout().
[[nodiscard]] int matrix_activation_layout(GgmlType type);
[[nodiscard]] std::size_t matrix_activation_bytes(int k, int columns);
void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

// Bytes of the stream-k fixup plane the matrix kernel may need for this product on the current
// device (zero when the tiles divide evenly across the machine).
[[nodiscard]] std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns);

// out[c * out_column_stride + r] = sum_k W[r, k] x[k, c] in FP32. `k` is a whole number of the
// kernel's 256-value steps. A matrix whose rows are shorter (row_bytes holds fewer than k values) is
// read past each row's end into the next row, and past the last row into zeros the caller promises
// (`tail`); its activation must be zero from the row width up to k (quantize_matrix_activation_padded).
void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                    bool tail = false);
// The 256-value step a padded matrix product covers for rows of `k` values.
[[nodiscard]] int matrix_padded_k(int k);
// The activation of a padded product: x BF16 [k, columns] quantized with zeros from k up to
// matrix_padded_k(k), into matrix_activation_bytes(matrix_padded_k(k), columns).
void quantize_matrix_activation_padded(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                       const std::int32_t* input_columns, void* out,
                                       cudaStream_t stream);

// --- Tile kernel (sm_120) ---------------------------------------------------------------------
// The prefill product of IQ3_XXS, IQ3_S, IQ2_XXS, IQ4_XS, IQ2_XS and IQ2_S on sm_120 devices in
// 128-row x 128-column tiles, the epilogue fused, in one of two arithmetics: the matrix kernel's
// q8_1, or on a 120a build MXFP8 (e4m3 activations and weights, one power-of-two scale per 32
// values of each, block-scaled tensor cores), which also serves Q2_K and Q4_K. Measured on the
// RTX 5060 Ti only.

// Whether this build has the MXFP8 form (the block-scaled e4m3 MMA of a 120a build).
inline constexpr bool kTileMxfp8Built =
#if defined(NINFER_SM120_FP8)
    true;
#else
    false;
#endif

// Columns from which wide products of a tile type may take the tile kernel, whose 128-column tile
// a narrower product pads. On the RTX 5060 Ti the 27B's prompts prefilled 6-8% faster than on the
// matrix kernel at 33-64 tokens and 24-26% at 72-127, but slower at 32 and below (4% at 32, 13% at
// 16).
inline constexpr int kTileMinColumns = 33;

// Whether the tile kernel serves products of this type and width K on the current device in the
// MXFP8 form (`mxfp8`, never without kTileMxfp8Built) or the q8_1 one (not Q2_K and Q4_K).
[[nodiscard]] bool has_tile_kernel(GgmlType type, int k, bool mxfp8);

// Whether a wide product takes the tile kernel rather than the matrix kernel: a served type from
// kTileMinColumns columns, when the matrix kernel cannot take its rows (not whole 128-row tiles)
// or when its grid of 128 x 128 tiles, one per SM at a time, fills its waves enough for the tile
// kernel's rate to beat the matrix kernel's stream-k schedule (a measured fill per type).
[[nodiscard]] bool takes_tile_kernel(GgmlType type, int rows, int k, int columns, bool mxfp8);

// Its activation: 8-bit values [columns][k] from byte 0, then FP32 d [k / 32][columns rounded up
// to 128] from tile_activation_scales_offset(). q8_1: the matrix kernel's D4 numbers (d = max|x| /
// 127 over 32 values, each value rounded to x / d). MXFP8: d = 2^E with max|x| / d in (224, 448]
// over 32 values, each value x / d rounded to e4m3. `input_columns` as for the vector activation.
[[nodiscard]] std::size_t tile_activation_bytes(int k, int columns);
[[nodiscard]] std::size_t tile_activation_scales_offset(int k, int columns);
void quantize_tile_activation(const __nv_bfloat16* x, int k, int columns,
                              const std::int32_t* input_columns, bool mxfp8, void* out,
                              cudaStream_t stream);

// out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for rows [0, rows) and columns [0, columns), for a
// type and K with has_tile_kernel in that form, any rows and columns; `activation` of that form.
void tile_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                  const void* activation, bool mxfp8, int columns, const VectorOutput& out,
                  cudaStream_t stream);

// --- Two-pass product (sm_120, cuBLASLt) ---------------------------------------------------------
// A wide product in two passes per slab of rows: the slab's blocks decode into a block-scaled image
// in scratch, and cuBLASLt's block-scaled GEMM multiplies it by the activation in the same form
// with FP32 accumulation into BF16. A slab of about kLtImageBytes stays in L2 between its passes.
// cuBLASLt offers no FP32 output for it: products into FP32 planes keep the tile kernel, and a
// gated product rounds its gate and up products to BF16 before act(gate) * up.
//
// Mxfp8: the tile kernel's MXFP8 arithmetic (the same e4m3 values and UE8M0 exponent per 32
// values, VEC32_UE8M0), for the types with that form. On the RTX 5060 Ti the 27B's products ran
// 4-20% faster than the tile kernel at 1024 columns and 16-20% at 2048, but slower at 256-768 for
// some shapes (a narrow GEMM fills the device poorly while the image pass costs the same).
//
// Nvfp4: e2m1 values with an E4M3 scale per 16 (VEC16_UE4M3) under one power-of-two global per
// operand (lt_fp4_global: max |x| g <= 2688), the scale of each 16 chosen by least squared error
// among four candidates; alpha = 1 / (g_w g_x). The weight's global comes from its max |w|, which
// the caller measures once (matrix_amax); the activation's from a max |x| pass per product. The
// image decodes with the vector kernel's decoder: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S, IQ4_XS, Q2_K, Q4_K and Q6_K. cuBLASLt's NVFP4 GEMM runs ~340 TOPS at the 27B's prefill
// shapes against ~170 for MXFP8; on the 27B's 1m quick perplexity corpus (head on MXFP8) the form
// costs +1.7% perplexity against MXFP8, which costs +0.45% against q8_1.
enum class LtForm : std::uint8_t { Mxfp8, Nvfp4 };

// Columns from which the Ops take it in each form. The image pass costs the same at any width, so
// the matrix and tile kernels serve the narrow products. Mxfp8 from 1024 (above). Nvfp4 from 129:
// on the RTX 5060 Ti the 27B's prompts prefilled 6-11% faster than on the matrix and tile kernels
// at 129-384 tokens, 32% at 512, 57% at 768 and 74% at 1000, but slower at 128 and below (29% at a
// whole 128-column tile, 12% at 127 on the matrix kernel).
[[nodiscard]] constexpr int lt_min_columns(LtForm form) {
    return form == LtForm::Nvfp4 ? 129 : 1024;
}
// Image bytes of an MXFP8 slab, whose rows an Nvfp4 slab keeps (in half the bytes).
inline constexpr std::size_t kLtImageBytes = std::size_t{21} << 19; // 10.5 MiB

// Whether it serves products of this type and K in this form on the current device: a 120a
// build, a cuBLASLt with the block-scaled GEMM (12.8 or newer), and for Mxfp8 a type with the
// MXFP8 tile form.
[[nodiscard]] bool has_lt_product(GgmlType type, int k, LtForm form);

// Whether an output can take it: BF16 Store or Accumulate, 16-byte aligned columns.
[[nodiscard]] bool lt_takes_output(const VectorOutput& out);

// Its activation: values [columns][k] from byte 0 (Mxfp8: the tile activation's e4m3 bytes; Nvfp4:
// e2m1 codes, two a byte, the first value in the low nibble), then from
// lt_activation_scales_offset() the scales in cuBLASLt's layout (Mxfp8: UE8M0 exponents E of d =
// 2^(E - 127), E = 127 for the padding up to 128 columns; Nvfp4: E4M3, zero for the padding),
// then for Nvfp4 the activation's max |x| (FP32 bits) and, in the scales' layout, the quantizer's
// per-group powers of two.
[[nodiscard]] std::size_t lt_activation_bytes(int k, int columns, LtForm form);
[[nodiscard]] std::size_t lt_activation_scales_offset(int k, int columns, LtForm form);
void quantize_lt_activation(const __nv_bfloat16* x, int k, int columns,
                            const std::int32_t* input_columns, LtForm form, void* out,
                            cudaStream_t stream);

// The parts of an Nvfp4 activation of x [columns][k] for a producer outside the bridge that
// encodes x as it forms it, in lt_fp4_encode_kernel's form (lt_fp4.cuh lt_fp4_encode_group and
// lt_fp4_block_max): begin_lt_fp4_activation zeroes the max |x| slot; after the encode,
// finish_lt_fp4_activation runs the scale pass over x, which leaves quantize_lt_activation(x, k,
// columns, input_columns, Nvfp4) byte for byte for the gather the encode took.
struct LtFp4Parts {
    std::uint8_t* codes;
    std::uint8_t* scales;
    std::int8_t* shifts;
    unsigned* slot;
};
[[nodiscard]] LtFp4Parts begin_lt_fp4_activation(void* activation, int k, int columns,
                                                 cudaStream_t stream);
void finish_lt_fp4_activation(const __nv_bfloat16* x, const std::int32_t* input_columns, int k,
                              int columns, const LtFp4Parts& parts, cudaStream_t stream);

// Scratch of a pass over `rows` rows, a gated pass's `rows` gate and `rows` up rows with their
// BF16 products; the most rows per pass (whole 128-row tiles) a scratch of `scratch_bytes` holds
// (a product runs passes of at most that many, as many as fill the GEMM's waves best); and the
// scratch a product of `rows` rows plans (passes of up to a slab).
[[nodiscard]] std::size_t lt_scratch_bytes(int k, int rows, int columns, bool gated, LtForm form);
[[nodiscard]] int lt_rows_per_pass(int k, int columns, bool gated, LtForm form,
                                   std::size_t scratch_bytes);
[[nodiscard]] std::size_t lt_planned_scratch_bytes(int k, int rows, int columns, bool gated,
                                                   LtForm form);

// The scale-choice table of an Nvfp4 weight image: which of fp4_encode16's four candidate scales
// (lt_fp4.cuh) each 16-value group of the image takes, two bits a group, row-major (k / 64 bytes a
// row), built by fp4_choices under the power-of-two global of `weight_amax`. A product whose own
// global is that one, given the table 128-byte aligned, encodes each group's recorded candidate
// instead of searching: the same image, in about a third of the image pass's instructions
// (decode-bound then). It ignores any other table and searches. The table may live in mapped host
// memory: a pass reads one 2-bit choice per 16 weights over the bus.
struct Fp4Choices {
    const std::uint8_t* table = nullptr;
    float weight_amax         = 0.0F;
};

// Bytes of the table of `rows` rows (whole 128-byte blocks); the table of rows [first, ...) of a
// table (null where row `first` does not start a byte); the table of the search's winners for a
// matrix that matrix_amax serves, into `table` (128-byte aligned, fp4_choice_bytes, device-writable).
[[nodiscard]] std::size_t fp4_choice_bytes(int rows, int k);
[[nodiscard]] const std::uint8_t* fp4_choice_rows(const std::uint8_t* table, int first, int k);
void fp4_choices(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                 float weight_amax, std::uint8_t* table, cudaStream_t stream);

// out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for an output lt_takes_output accepts, rows [0,
// rows) and columns [0, columns); `activation` from quantize_lt_activation in `form`. Nvfp4:
// `weight_amax` is max |w| over the matrix (matrix_amax), positive; `choices` the weight's table,
// if any.
void lt_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                LtForm form, float weight_amax, const Fp4Choices& choices,
                const void* activation, int columns, const VectorOutput& out, void* scratch,
                std::size_t scratch_bytes, cudaStream_t stream);

// out[r, c] = act(gate_r . x_c) * (up_r . x_c) in BF16 (out 16-byte aligned, out_column_stride a
// multiple of 8), gate and up rows each of their own type. A pass stacks its gate and up rows in
// one image and, over one activation (`up_activation` null), one GEMM; an up activation of its
// own (another input gather) takes a GEMM per half. Nvfp4: `weight_amax` covers gate and up (and the
// global each one's choice table must have), and a non-null `out_activation` (lt_activation_bytes(rows, columns, Nvfp4), out contiguous
// [columns][rows], rows whole 256-value steps) also receives quantize_lt_activation(out, rows,
// columns, no gather, Nvfp4), encoded as the products are formed.
void lt_gated_product(GgmlType gate_type, const void* gate, std::int64_t gate_row_bytes,
                      GgmlType up_type, const void* up, std::int64_t up_row_bytes, int rows, int k,
                      LtForm form, float weight_amax, const Fp4Choices& gate_choices,
                      const Fp4Choices& up_choices, const void* activation,
                      const void* up_activation, int columns, __nv_bfloat16* out,
                      std::int64_t out_column_stride, GateActivation gate_activation,
                      void* out_activation, void* scratch, std::size_t scratch_bytes,
                      cudaStream_t stream);

// max |w| over the decoded values of rows [0, rows) of a type has_lt_product serves in Nvfp4, into
// *amax as FP32 bits (zeroed here first). Load-time preparation of the Nvfp4 form's weight global.
void matrix_amax(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                 unsigned* amax, cudaStream_t stream);

// out <- epilogue(in) for an FP32 [rows, columns] plane whose columns are `in_column_stride` apart.
void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream);

// out[c * k + i] = x[c * k + input_columns[i]] for each of `columns` BF16 columns of width k.
void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream);

// Rows per pass of dequantized_product for a scratch of `scratch_bytes`.
[[nodiscard]] int dequantized_rows_per_pass(int k, std::size_t scratch_bytes);

// The same product as matrix_product for a type without an integer kernel: each pass dequantizes
// up to dequantized_rows_per_pass rows into `scratch` (BF16) and runs a cuBLAS BF16 GEMM with an
// FP32 accumulator. x is BF16 [k, columns].
void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream);

// --- Mixture of experts -------------------------------------------------------------------------
// A product over the experts a router selected: pair p (token p / per_token, its slot p %
// per_token) multiplies the rows of expert ids[p]. Experts are reached through a device table of
// base pointers, so an expert may live in a device bank, a cache slot or mapped host memory.

// The pairs grouped by expert: `sorted` lists the pairs of expert e at [bounds[e], bounds[e + 1]),
// and `active` the `*active_count` experts that have any, ascending. Device arrays.
struct MoeRouting {
    const std::int32_t* bounds       = nullptr; // [experts + 1]
    const std::int32_t* sorted       = nullptr; // [pairs]
    const std::int32_t* active       = nullptr; // [experts]
    const std::int32_t* active_count = nullptr; // [1]
};

[[nodiscard]] std::size_t moe_routing_bytes(int experts, int pairs);
// Groups the pairs of `ids` ([pairs], each in [0, experts)) into `workspace` (moe_routing_bytes).
// Within one expert the pairs keep no particular order; every product below writes each pair's
// result to its own place, so the order changes no value.
[[nodiscard]] MoeRouting moe_sort_routes(const std::int32_t* ids, int pairs, int experts,
                                         void* workspace, cudaStream_t stream);

// One projection of every expert: expert e's `rows` rows of `k` values start at experts[e].
struct MoeTable {
    const void* const* experts = nullptr; // device array [experts]
    std::int64_t row_bytes     = 0;
    int rows                   = 0;
    int k                      = 0;
};

// Where a product over experts puts pair p's row r.
struct MoeOutput {
    __nv_bfloat16* bf16 = nullptr; // [pairs][rows], or
    float* f32          = nullptr; // [pairs][rows], or
    // sum over a token's pairs of weights[p] * value, as 2^-32 fixed point in two's complement:
    // [tokens][rows] where token = p / per_token. Integer addition is associative, so the sum does
    // not depend on the order the experts finish in. The caller zeroes it first.
    unsigned long long* weighted = nullptr;
    const float* weights         = nullptr; // [pairs], with `weighted`
    // Multiplies the value by act(gate[p][r]) first ([pairs][rows]).
    const float* gate = nullptr;
    GateActivation activation = GateActivation::Silu;
};

// Activation column of pair p is p / column_group: the token's (column_group = per_token) for the
// first projection, the pair's own middle (column_group = 1) for the second. `max_active` bounds
// *routing.active_count (min(pairs, experts)); `chunk` is how many of an expert's columns share one
// pass over its rows (1, 2, 4 or 8).
void moe_vector_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int column_group, int per_token, const void* activation,
                        int columns, const MoeOutput& out, int chunk, cudaStream_t stream);
// silu(gate . x) * (up . x) of each pair into out [pairs][rows] (BF16): gate and up are two tables
// of one block type.
void moe_vector_swiglu(GgmlType type, const MoeTable& gate, const MoeTable& up,
                       const MoeRouting& routing, int max_active, int column_group,
                       const void* activation, int columns, __nv_bfloat16* out, int chunk,
                       cudaStream_t stream, GateActivation gate_activation = GateActivation::Silu);

// One decode or verification call's experts (up to 8 tokens) in few launches: moe_decode_up writes
// silu(gate . x) * (up . x) of every listed pair to `middle`, quantizing its tokens' BF16 inputs
// itself; moe_decode_down runs the down rows of every listed pair into the 2^-32 fixed-point sum
// and, when `y` is given, its last block adds the external products and stores y. Each block row
// serves one expert's pairs, found by scanning the ids (no sort). The arithmetic of every row is
// moe_vector_swiglu's and moe_vector_product's, so the sums are theirs bit for bit. `chunk` is how
// many of an expert's pairs share a pass over its rows (1, 2, 4 or 8).
//
// A call can be prepared once instead of in every block: moe_decode_prep lists each slot's pairs
// (records) and quantizes the tokens' inputs, and the up launches quantize the pairs' middles for
// the down kernel (and list the second stage's slots, whose ids are final only then).
struct MoeDecodeRecord {
    // The slot's expert and the id set it came from (0: ids_a, 1: ids_b), with that expert's
    // pairs in ascending order and their tokens; n == 0 when the slot is negative in its set or
    // repeats an earlier slot's expert.
    std::int32_t expert = -1, n = 0, source = 0, reserved = 0;
    std::int32_t pairs[8] = {}, columns[8] = {};
};
struct MoeDecodePrepArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][2560]
    int tokens             = 0;
    const std::int32_t* ids = nullptr; // [slots], the first stage's set (ids_a)
    int slots = 0, per_token = 10;
    MoeDecodeRecord* records = nullptr; // [slots]
    // The inputs as `type`'s up kernel reads them: q8_1 values [tokens][2560], (d, sum) and the
    // integer sum of every 32 values [tokens][80].
    std::int8_t* qs          = nullptr;
    half2* ds                = nullptr;
    int* qsum                = nullptr;
    unsigned long long* zero = nullptr; // zeroed when given
    int zero_count           = 0;
};
void moe_decode_prep(GgmlType type, const MoeDecodePrepArgs& args, cudaStream_t stream);
// Whether `type`'s decode kernels read transposed activations (MoeDecodeUpArgs::middle_transposed).
[[nodiscard]] bool moe_decode_transposed(GgmlType type);

struct MoeDecodeUpArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][k]
    int k = 2560, rows = 640;
    const std::int32_t* ids = nullptr; // [slots]: negative entries skipped; null: all expert 0
    int slots = 0, per_token = 10;
    const std::uint8_t* const* gate = nullptr;
    const std::uint8_t* const* up   = nullptr;
    std::int64_t row_bytes          = 0;
    __nv_bfloat16* middle           = nullptr; // [slots][rows]
    unsigned long long* zero        = nullptr; // zeroed first when given
    int zero_count                  = 0;
    // Prepared: the slots' records (instead of scanning ids) and the quantized inputs (instead of m).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
    // When given, every block quantizes its 32 middle values of every pair for the down kernel
    // (transposed when its decoder reads them so) instead of storing them: [slots][640], and
    // [slots][20] (d, sum) and integer sums. Requires rows to be whole groups of 32.
    std::int8_t* middle_qs         = nullptr;
    half2* middle_ds               = nullptr;
    int* middle_qsum               = nullptr;
    bool middle_transposed         = false;
    // Scanning `ids`, a block row also lists its slot in `write_records` as `record_source`: the
    // first stage (0) every slot, the second (1) the slots it serves.
    MoeDecodeRecord* write_records = nullptr;
    int record_source              = 1;
};
struct MoeDecodeDownArgs {
    int rows = 2560; // the down projection's outputs; its k is 640
    // A slot is served from table_a when ids_a[slot] >= 0, else from table_b when ids_b[slot] >= 0.
    const std::int32_t* ids_a = nullptr; // null: all expert 0
    const std::int32_t* ids_b = nullptr;
    int slots = 0, per_token = 10;
    const std::uint8_t* const* table_a = nullptr;
    const std::uint8_t* const* table_b = nullptr;
    std::int64_t row_bytes             = 0;
    const __nv_bfloat16* middle        = nullptr; // [slots][640]
    const float* weights               = nullptr; // [slots]
    unsigned long long* fixed          = nullptr; // [tokens][rows]
    int tokens                         = 0;
    // The call's last launch: y [tokens][rows], external products (as add_products), and a
    // completion counter that starts at zero and is left at zero.
    float* y                         = nullptr;
    unsigned int* counter            = nullptr;
    const float* products            = nullptr;
    const std::int32_t* product_pairs = nullptr;
    const std::int32_t* product_count = nullptr;
    // Prepared: the slots' records and the quantized middles (MoeDecodeUpArgs::middle_qs).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
};
void moe_decode_up(GgmlType type, const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream);
void moe_decode_down(GgmlType type, const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream);

// The matrix kernel over the experts a router selected, for products wide enough for its tiles:
// pair p's FP32 row lands at out[p * table.rows]. The activation holds the pairs' columns in
// routing order (quantize_moe_matrix_activation into moe_matrix_activation_bytes), and
// `max_columns` bounds any one expert's pairs. The kernel reads k in whole 256-value steps: a
// geometry whose k is not a whole number of them needs `tail`, the caller's promise that zeros
// follow each bank's last expert (Binder::device_tail, a zeroed slot pool), since a row's last
// step reads into the next row. A type without an integer kernel, or a geometry no tile fits
// (rows a whole number of row tiles), is not supported and stays on moe_vector_product.
[[nodiscard]] bool moe_matrix_supported(GgmlType type, int rows, int k, bool tail);
void moe_matrix_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int pairs, int max_columns, bool tail,
                        const void* activation, float* out, cudaStream_t stream);
// Column c of the activation is x column routing.sorted[c] / column_group, in the layout of the
// type's scale/sum schedule (matrix_activation_layout), zero past k up to a whole 256-value step.
[[nodiscard]] std::size_t moe_matrix_activation_bytes(int k, int pairs);
void quantize_moe_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k,
                                    const MoeRouting& routing, int pairs, int column_group,
                                    void* out, cudaStream_t stream);

// out[i, :] = dequantize(W[row_ids ? row_ids[i] : i, :]) as BF16, rows `out_row_stride` apart.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream);
// The same rows in FP32, where every value is exact.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream);
// The same rows in FP16, the operand type of sm_70's tensor cores (ops/linear/sm70).
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __half* out,
                     std::int64_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::gguf
