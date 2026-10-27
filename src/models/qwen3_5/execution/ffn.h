#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"

namespace ninfer::models::qwen3_5::execution {

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
// `x_activation`: the query's sizes a call given hidden's GGUF NVFP4 activation.
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false, bool wide_verification = false,
                                              bool x_activation = false);
// `wide_verification` keeps the neural path's residual activation precision in a wide single-row
// copy verification (see residual_projection_policy). `x_activation` is hidden's GGUF NVFP4
// activation, where ffn_takes_gguf_activation.
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false, bool verify = false, InputBasis basis = InputBasis::Primal,
         bool wide_verification = false, const void* x_activation = nullptr);
// Whether the FFN over `tokens` columns of a `hidden`-wide state takes the post-attention norm's
// GGUF NVFP4 activation (ops::rmsnorm_gguf_activation): an unrotated dense GGUF FFN whose gated
// pair takes it.
[[nodiscard]] bool ffn_takes_gguf_activation(const FfnParameters& parameters, std::int32_t hidden,
                                             std::int32_t tokens);
// The sign vector of a rotated Dense FFN's input projection, or null.
[[nodiscard]] const Tensor* ffn_input_signs(const FfnParameters& parameters);

} // namespace ninfer::models::qwen3_5::execution
