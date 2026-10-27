#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// The most candidates shortlist_argmax rescores per column.
inline constexpr std::int32_t kShortlistMaxCandidates = 64;

/**
 * Op: shortlist_argmax
 *
 * The argmax of an exact head over the rows a coarse head ranks highest: a two-stage proposal head,
 * whose coarse logits come from a cheaper matrix over the same rows.
 *
 * Math / indexing:
 *   For column t:
 *     C_t    = the `candidates` rows r of the largest coarse[r, t], a tie to the lower r;
 *     y[r]   = BF16(sum_k W[r, k] x[k, t]) for r in C_t, by the GGUF vector product's arithmetic:
 *              x quantized to ggml's q8_1 per 32 values, the integer dot of each 32-value slice
 *              scaled in FP32 and summed in the vector kernel's order, so y[r] is bit for bit the
 *              row's value in linear()'s BF16 output for x at T <= 8;
 *     out[t] = token_ids[r*] (r* when token_ids is null), r* the lowest r in C_t with the largest
 *              y[r].
 *
 * Logical shapes:
 *   coarse BF16 [R, T] and x BF16 [K, T], both contiguous; `exact` a GGUF matrix [R, K]
 *   (QuantLayout::GgufBlocks over whole blocks, no input gather); token_ids I32 [R] contiguous or
 *   null; out I32 [T].
 *
 * Supported domain:
 *   1 <= candidates <= min(kShortlistMaxCandidates, R); R <= 2048 * floor(8192 / candidates)
 *   (262144 at 64 candidates, 524288 at 32); 1 <= T <= 65535. The coarse logits are finite: a NaN
 *   ranks below every number, as does a NaN y.
 *
 * Residency:
 *   Only the rows of C_t are read from `exact`, each once per column, so its blocks may live in
 *   mapped, device-readable host memory. coarse, x, token_ids, out and the workspace are device
 *   memory.
 *
 * Numeric:
 *   Selection compares BF16 values exactly. y is the GGUF vector product's value: an approximation
 *   of the FP64 dot of the decoded weights with x under q8_1 activation rounding; the independent
 *   oracle bounds it by the activation's quantization error.
 *
 * Effects:
 *   Writes all of out. Inputs are unchanged; out overlaps nothing. No persistent state.
 *
 * Workspace:
 *   Caller-owned transient storage of shortlist_argmax_workspace_bytes(R, K, T, candidates).
 */
void shortlist_argmax(const Tensor& coarse, const Tensor& x, const Weight& exact,
                      std::int32_t candidates, const Tensor* token_ids, Tensor& out,
                      WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t shortlist_argmax_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                           std::int32_t columns,
                                                           std::int32_t candidates);

} // namespace ninfer::ops
