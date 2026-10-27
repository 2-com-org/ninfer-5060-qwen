#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * GGUF NVFP4 input activation.
 *
 * Under AllowA4 a GGUF product over T >= kGgufActivationMinTokens columns may take the NVFP4
 * two-pass form, which quantizes its BF16 input x [K,T] into a private NVFP4 activation before its
 * GEMM. The producers here write that activation while they form x, and the GGUF consumers
 * (linear_add, linear_swiglu_add, and the GGUF forms of attn_input_proj and gdn_input_proj) take
 * it as `x_activation` instead of quantizing x themselves: every product of theirs in that form
 * over an ungathered x reads it (linear_add's one product: under the matrix's own input gather),
 * any other quantizes x as before. The bytes are those such a product forms from x, so no result
 * changes. Its workspace queries take `x_activation` as well, and then leave out the bytes of the
 * activation the caller provides.
 *
 * gguf_takes_activation reports whether a GGUF product of `weight` over `tokens` columns under
 * `policy` takes that form, over x gathered by weight.input_columns (an output it cannot take
 * aside); a caller provides the activation where every product of a consumer does, ungathered for
 * the consumers of several products.
 */
[[nodiscard]] std::size_t gguf_activation_bytes(std::int32_t k, std::int32_t tokens);
// The fewest columns of a product in that form (and of the producers below).
inline constexpr std::int32_t kGgufActivationMinTokens = 129;
// The hidden width rmsnorm_gguf_activation serves and the head width and the most heads of
// gated_rmsnorm_gguf_activation.
inline constexpr std::int32_t kRmsNormGgufActivationWidth        = 5120;
inline constexpr std::int32_t kGatedRmsNormGgufActivationHead     = 128;
inline constexpr std::int32_t kGatedRmsNormGgufActivationMaxHeads = 96;
[[nodiscard]] bool gguf_takes_activation(const Weight& weight, LinearPolicy policy,
                                         std::int32_t tokens);

/**
 * rmsnorm(x, weight, eps, unit_offset = true, out) for x [5120, T], T >= kGgufActivationMinTokens,
 * and into `activation` (gguf_activation_bytes(5120, T) bytes) the GGUF NVFP4 activation of out.
 * out is byte for byte that of rmsnorm.
 */
void rmsnorm_gguf_activation(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                             void* activation, cudaStream_t stream);

/**
 * gated_rmsnorm(x, weight, z, Silu, eps, out) for x [128, H, T] (H even and at most 96,
 * T >= kGgufActivationMinTokens), and into `activation` (gguf_activation_bytes(128 H, T) bytes)
 * the GGUF NVFP4 activation of out read as [128 H, T] under the consumer's input gather
 * `input_columns` (Weight::input_columns, 128 H device indices; null for none). out is byte for
 * byte that of gated_rmsnorm.
 */
void gated_rmsnorm_gguf_activation(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                                   Tensor& out, void* activation,
                                   const std::int32_t* input_columns, cudaStream_t stream);

} // namespace ninfer::ops
