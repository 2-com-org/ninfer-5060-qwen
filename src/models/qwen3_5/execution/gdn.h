#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "core/device.h"

namespace ninfer::models::qwen3_5::execution {

// `x_activation`: the hidden state's GGUF NVFP4 activation, where
// gdn_projection_takes_gguf_activation (the query's: a call given it).
[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters,
                                                         std::int32_t first, std::int32_t last,
                                                         bool x_activation = false);
// Whether the projection over `tokens` columns of a `hidden`-wide state takes the input norm's
// GGUF NVFP4 activation (gdn_norm_control's `h_activation`): an unrotated GGUF projection every
// part of which takes it.
[[nodiscard]] bool gdn_projection_takes_gguf_activation(const GdnParameters& parameters,
                                                        std::int32_t hidden, std::int32_t tokens);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters,
                                                       const GdnConfig& config, std::int32_t batch,
                                                       std::int32_t first_width,
                                                       std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters,
                                                     const GdnConfig& config, std::int32_t batch,
                                                     std::int32_t first_width,
                                                     std::int32_t last_width);
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal, const void* x_activation = nullptr);
// Whether the output projection over `tokens` columns takes the gated norm's GGUF NVFP4
// activation (ops::gated_rmsnorm_gguf_activation over the recurrence's heads).
[[nodiscard]] bool gdn_output_takes_gguf_activation(const GdnParameters& parameters,
                                                    const GdnConfig& config, std::int32_t tokens,
                                                    bool wide_verification = false);
// Returns the basis `hidden` is written in: rotated when the input projection is. A non-null
// `h_activation` (gdn_projection_takes_gguf_activation) also receives hidden's GGUF NVFP4
// activation.
[[nodiscard]] InputBasis gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                                          const GdnParameters& parameters, Tensor& hidden,
                                          Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                          DeviceExecutionView execution,
                                          void* h_activation = nullptr);
void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream, InputBasis basis = InputBasis::Primal);
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis = InputBasis::Primal);

} // namespace ninfer::models::qwen3_5::execution
