#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 through T=16; AllowA4 accepts every positive T. Row-scaled FP8 accepts all
 * policies, with A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4_G64_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [12288,2048], x [2048,T], out [6144,T];
 *   - Q8_G32_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16 RowScale weight [34816,5120], x [5120,T], out [17408,T].
 *   Inputs and output are contiguous BF16. Q4/Q8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 multiplier per gate/up parent row. Gate rows `[0,17408)` precede
 *   their matching up rows `[17408,34816)`.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   AllowA8/AllowA4 permit FP8 activation quantization; route thresholds are implementation
 * choices.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. Q8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; A4/A8 routes use
 *   caller-owned activation storage and may use private projection storage. There is no persistent
 *   state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 is admitted only through T=16; larger NVFP4 extents require the policy-bearing AllowA4
 * form.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

/**
 * Two-parent form: out[:,t] = silu(gate_weight * x[:,t]) * (up_weight * x[:,t]) for gate and up
 * matrices of one shape [N,K] stored as separate parents. Registered for GGUF block matrices, whose
 * two halves may use different block types; their products quantize the activation to q8_1 as
 * llama.cpp does, whatever the policy. Size the workspace with
 * linear_swiglu_pair_workspace_capacity_bytes().
 */
void linear_swiglu(const Tensor& x, const Weight& gate_weight, const Weight& up_weight, Tensor& out,
                   LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream);

[[nodiscard]] std::size_t linear_swiglu_pair_workspace_capacity_bytes(
    QType gate_qtype, QType up_qtype, std::int32_t rows, std::int32_t input_rows,
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu_add
 *
 * Math: residual[:,t] += down_weight * h[:,t], h = linear_swiglu(x) under gate_policy (one
 * [gate; up] parent when up_weight is null, else the two-parent form), the product under
 * down_policy: linear_swiglu into a BF16 h, then linear_add of h.
 *
 * Registered for GGUF block matrices (gate, up and down). Numerically that composition: where
 * both products take the NVFP4 two-pass route, the gate kernel encodes h for down as it forms it,
 * byte for byte what linear_add's A4 activation quantization of h gives.
 *
 * Workspace: linear_swiglu_add_workspace_capacity_bytes() (gate_rows = gate_weight.n; up_qtype
 * null for one parent), scoped to the call, including h.
 */
void linear_swiglu_add(const Tensor& x, const Weight& gate_weight, const Weight* up_weight,
                       const Weight& down_weight, Tensor& residual, LinearPolicy gate_policy,
                       LinearPolicy down_policy, WorkspaceArena& ws, cudaStream_t stream,
                       const void* x_activation = nullptr);

// `x_activation`: x's NVFP4 activation for the gated pair (ninfer/ops/gguf_activation.h); the
// query's sizes a call given it.
[[nodiscard]] std::size_t linear_swiglu_add_workspace_capacity_bytes(
    QType gate_qtype, const QType* up_qtype, std::int32_t gate_rows, std::int32_t input_rows,
    QType down_qtype, std::int32_t output_rows, LinearPolicy gate_policy, LinearPolicy down_policy,
    std::int32_t min_tokens, std::int32_t max_tokens, bool x_activation = false);

} // namespace ninfer::ops
