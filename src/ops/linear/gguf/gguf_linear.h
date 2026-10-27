#pragma once

// Products of GGUF block matrices (QType::GGUF_*, QuantLayout::GgufBlocks), accumulated in FP32:
// ggml's vector kernel through eight columns, its integer tensor-core kernel above, and on sm_120
// the tile kernel for the types and widths it serves (gguf::takes_tile_kernel). Each quantizes its
// activation to ggml's q8_1 exactly as llama.cpp does -- the format's arithmetic -- except that a
// policy allowing A8 activations (allows_a8) lets the tile kernel of a 120a build run its MXFP8
// form instead (gguf::kTileMxfp8Built): e4m3 activations and weights under one power-of-two scale
// per 32 values each, which also serves Q2_K and Q4_K. The two-pass product (gguf::LtForm) takes
// the BF16 outputs in its NVFP4 form from 129 columns where the policy allows A4 activations
// (allows_a4; the matrix then carries Weight::gguf_amax), else in its MXFP8 form from 1024
// (gguf::lt_min_columns).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

[[nodiscard]] gguf::GgmlType gguf_type(QType format);

// Throws unless `w` is a resident GGUF matrix over whole blocks.
void require_gguf(const Weight& w, const char* label);

enum class GgufEpilogue {
    Store,       // out = W x
    Accumulate,  // out += W x
    GateProduct, // out = silu(gate) * (W x), gate an FP32 plane of the same rows
};

// One GGUF matrix and where its rows land: rows [row, row + weight->n) of `out` (BF16 [R,T]) or,
// with `f32`, of an FP32 plane with `f32_rows` rows per column.
struct GgufProduct {
    const Weight* weight  = nullptr;
    Tensor* out           = nullptr;
    std::int32_t row      = 0;
    GgufEpilogue epilogue = GgufEpilogue::Store;
    const float* gate     = nullptr;
    float* f32            = nullptr;
    std::int32_t f32_rows = 0;
    gguf::GateActivation activation = gguf::GateActivation::Silu; // GateProduct only
};

// What a workspace query needs of a product: its format and [rows, k]. A gather of the input
// (Weight::input_columns) is assumed wherever it can cost bytes.
struct GgufShape {
    QType qtype       = QType::GGUF_Q8_0;
    std::int32_t rows = 0;
    std::int32_t k    = 0;
};

// Transient bytes gguf_project needs for these products (one K) under `policy` at every width in
// [min_tokens, max_tokens]; with `x_activation`, those of a call given it.
[[nodiscard]] std::size_t gguf_project_workspace_bytes(std::span<const GgufShape> shapes,
                                                       LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens,
                                                       bool x_activation = false);

// Runs every product against one BF16 activation x [K,T], quantizing x once per activation layout
// (and once more per distinct input gather). `policy` admits the MXFP8 forms (allows_a8) and the
// NVFP4 two-pass form (allows_a4). A non-null `x_activation` is x's NVFP4 activation
// (ops::gguf_activation_bytes) under the input gather `x_activation_columns`, which every product
// in that form over x so gathered takes.
void gguf_project(const Tensor& x, std::span<const GgufProduct> products, LinearPolicy policy,
                  WorkspaceArena& workspace, cudaStream_t stream,
                  const void* x_activation = nullptr,
                  const std::int32_t* x_activation_columns = nullptr);

// Whether a product of `w` over `t` columns under `policy` takes the NVFP4 two-pass form over x
// gathered by w.input_columns, an output it cannot take aside (see ops::gguf_takes_activation).
[[nodiscard]] bool gguf_takes_nvfp4_activation(const Weight& w, LinearPolicy policy,
                                               std::int32_t t);

void gguf_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena& workspace, cudaStream_t stream);
void gguf_linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream,
                     const void* x_activation = nullptr);

// out = act(gate x) * (up x), act SiLU unless `activation` says GELU (tanh). `up` null means
// `gate` is one [gate; up] parent of 2 * out rows.
void gguf_swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
                 LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                 gguf::GateActivation activation = gguf::GateActivation::Silu);
// `up` null: `gate` is the one [gate; up] parent.
[[nodiscard]] std::size_t gguf_swiglu_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                                      LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens,
                                                      bool x_activation = false);

// residual += down (gguf_swiglu(x, gate, up) under gate_policy) under down_policy, SiLU. Where the
// gated pair and down both take the NVFP4 two-pass product, the gate kernel encodes down's
// activation as it forms it (gguf::lt_gated_product), byte for byte quantize_lt_activation of the
// BF16 products, so down skips that pass.
// A non-null `x_activation` as for gguf_project, for the gated pair.
void gguf_swiglu_add(const Tensor& x, const Weight& gate, const Weight* up, const Weight& down,
                     Tensor& residual, LinearPolicy gate_policy, LinearPolicy down_policy,
                     WorkspaceArena& workspace, cudaStream_t stream,
                     const void* x_activation = nullptr);
[[nodiscard]] std::size_t gguf_swiglu_add_workspace_bytes(const GgufShape& gate,
                                                          const GgufShape* up,
                                                          const GgufShape& down,
                                                          LinearPolicy gate_policy,
                                                          LinearPolicy down_policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens,
                                                          bool x_activation = false);

// See ops::gguf_max_abs.
void gguf_max_abs(const Weight& w, unsigned* amax, cudaStream_t stream);

// See ops::gguf_fp4_choice_bytes and ops::gguf_fp4_choices.
[[nodiscard]] std::size_t gguf_fp4_choice_bytes(const Weight& w, LinearPolicy policy);
void gguf_fp4_choices(const Weight& w, float amax, std::uint8_t* table, cudaStream_t stream);

// Gather of token-table rows: out[:, i] = W[ids[i], :] as BF16.
void gguf_embedding(const Weight& table, const Tensor& ids, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
