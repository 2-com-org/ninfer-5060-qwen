// ninfer::ops - GGUF NVFP4 input activation producers: public api validation and dispatch.
#include "ninfer/ops/gguf_activation.h"

#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/gguf/rmsnorm_lt.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void require_bf16(const Tensor& t, std::int64_t numel, const char* op, const char* name) {
    if (t.dtype != DType::BF16 || !t.is_contiguous() || t.numel() != numel || !aligned16(t.data)) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be a contiguous, 16-byte aligned BF16 tensor of the "
                                    "expected shape");
    }
}

void require_eps(float eps, const char* op) {
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(op) + ": eps must be positive and finite");
    }
}

void require_activation(const void* activation, const char* op) {
    if (activation == nullptr || !aligned16(activation)) {
        throw std::invalid_argument(std::string(op) + ": activation must be 16-byte aligned");
    }
}

static_assert(kGgufActivationMinTokens == gguf::lt_min_columns(gguf::LtForm::Nvfp4));

} // namespace

std::size_t gguf_activation_bytes(std::int32_t k, std::int32_t tokens) {
    if (k <= 0 || k % 256 != 0 || tokens <= 0) {
        throw std::invalid_argument("gguf_activation_bytes: K must be whole 256-value steps");
    }
    return gguf::lt_activation_bytes(k, tokens, gguf::LtForm::Nvfp4);
}

bool gguf_takes_activation(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return detail::gguf_takes_nvfp4_activation(weight, policy, tokens);
}

void rmsnorm_gguf_activation(const Tensor& x, const Tensor& weight, float eps, Tensor& out,
                             void* activation, cudaStream_t stream) {
    constexpr const char* op = "rmsnorm_gguf_activation";
    constexpr std::int32_t d = detail::kRmsLtHidden;
    const std::int32_t t     = x.ne[1];
    if (x.ne[0] != d || x.ne[2] != 1 || x.ne[3] != 1 || t < kGgufActivationMinTokens) {
        throw std::invalid_argument(std::string(op) + ": x must be [5120, T], T >= " +
                                    std::to_string(kGgufActivationMinTokens));
    }
    require_eps(eps, op);
    require_bf16(x, std::int64_t(d) * t, op, "x");
    require_bf16(weight, d, op, "weight");
    require_bf16(out, std::int64_t(d) * t, op, "out");
    require_activation(activation, op);
    detail::rmsnorm_lt_fp4(static_cast<const __nv_bfloat16*>(x.data),
                           static_cast<const __nv_bfloat16*>(weight.data), eps, t,
                           static_cast<__nv_bfloat16*>(out.data), activation, stream);
}

void gated_rmsnorm_gguf_activation(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                                   Tensor& out, void* activation,
                                   const std::int32_t* input_columns, cudaStream_t stream) {
    constexpr const char* op = "gated_rmsnorm_gguf_activation";
    constexpr std::int32_t d = detail::kRmsLtHead;
    const std::int32_t heads = x.ne[1];
    const std::int32_t t     = x.ne[2];
    if (x.ne[0] != d || x.ne[3] != 1 || heads <= 0 || heads % 2 != 0 ||
        heads > detail::kRmsLtMaxHeads || t < kGgufActivationMinTokens) {
        throw std::invalid_argument(std::string(op) +
                                    ": x must be [128, H, T], H even and at most 96, T >= " +
                                    std::to_string(kGgufActivationMinTokens));
    }
    require_eps(eps, op);
    const std::int64_t numel = std::int64_t(d) * heads * t;
    require_bf16(x, numel, op, "x");
    require_bf16(z, numel, op, "z");
    require_bf16(weight, d, op, "weight");
    require_bf16(out, numel, op, "out");
    require_activation(activation, op);
    detail::gated_rmsnorm_lt_fp4(static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<const __nv_bfloat16*>(weight.data),
                                 static_cast<const __nv_bfloat16*>(z.data), eps, t, d * heads,
                                 input_columns, static_cast<__nv_bfloat16*>(out.data), activation,
                                 stream);
}

} // namespace ninfer::ops
