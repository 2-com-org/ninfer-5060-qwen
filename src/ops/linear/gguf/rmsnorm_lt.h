#pragma once

// RMSNorm producers that also write their output's NVFP4 two-pass activation (ggml_bridge.h,
// LtForm::Nvfp4), so the GGUF products over that output skip its quantization pass. Each output is
// byte for byte that of the rmsnorm route it mirrors, and the activation byte for byte
// gguf::quantize_lt_activation(out, k, columns, input_columns, Nvfp4). sm_120a builds only.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Hidden width of rmsnorm_lt_fp4.
inline constexpr int kRmsLtHidden = 5120;
// Head width and the most heads of gated_rmsnorm_lt_fp4.
inline constexpr int kRmsLtHead     = 128;
inline constexpr int kRmsLtMaxHeads = 96;

// out = rmsnorm(x, weight, eps, unit_offset) of BF16 x [columns][5120], as rmsnorm's 5120-wide
// route (rmsnorm_cta_bf16x2_kernel<Offset, 256, 10, true, 5120>), and its activation.
void rmsnorm_lt_fp4(const __nv_bfloat16* x, const __nv_bfloat16* weight, float eps, int columns,
                    __nv_bfloat16* out, void* activation, cudaStream_t stream);

// out = gated_rmsnorm(x, weight, z, Silu, eps) of BF16 x [columns][k / 128][128], as
// gated_rmsnorm's 128-wide route above one block per SM (rmsnorm_warp_bf16x2_kernel<Gated, 512,
// false>), and the activation of out [columns][k] under the input gather `input_columns` (null for
// none). k is whole 256-value steps of at most kRmsLtMaxHeads heads.
void gated_rmsnorm_lt_fp4(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                          const __nv_bfloat16* z, float eps, int columns, int k,
                          const std::int32_t* input_columns, __nv_bfloat16* out, void* activation,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail
