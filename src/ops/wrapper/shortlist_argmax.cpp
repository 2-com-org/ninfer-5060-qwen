// ninfer::ops - shortlist_argmax wrapper: public api validation and GGUF bridge dispatch.
#include "ninfer/ops/shortlist_argmax.h"

#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h" // detail::gguf_type, detail::require_gguf

#include <stdexcept>

namespace ninfer::ops {
namespace {

static_assert(kShortlistMaxCandidates == gguf::kMaxShortlistCandidates);

bool matrix(const Tensor& t, DType dtype, std::int32_t rows) {
    return t.dtype == dtype && t.ne[0] == rows && t.ne[1] >= 0 && t.ne[2] == 1 && t.ne[3] == 1 &&
           t.is_contiguous();
}

} // namespace

std::size_t shortlist_argmax_workspace_bytes(std::int32_t rows, std::int32_t k,
                                             std::int32_t columns, std::int32_t candidates) {
    return gguf::shortlist_argmax_bytes(rows, k, columns, candidates);
}

void shortlist_argmax(const Tensor& coarse, const Tensor& x, const Weight& exact,
                      std::int32_t candidates, const Tensor* token_ids, Tensor& out,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    detail::require_gguf(exact, "shortlist_argmax exact matrix");
    if (exact.input_columns != nullptr) {
        throw std::invalid_argument("shortlist_argmax: the exact matrix takes no input gather");
    }
    const std::int32_t rows = exact.n;
    const std::int32_t t    = coarse.ne[1];
    if (!matrix(coarse, DType::BF16, rows) || !matrix(x, DType::BF16, exact.k) || x.ne[1] != t) {
        throw std::invalid_argument("shortlist_argmax: coarse must be BF16 [R,T] and x [K,T]");
    }
    if (out.dtype != DType::I32 || out.ne[0] != t || out.ne[1] != 1 || out.ne[2] != 1 ||
        out.ne[3] != 1 || !out.is_contiguous()) {
        throw std::invalid_argument("shortlist_argmax: out must be I32 [T]");
    }
    if (token_ids != nullptr &&
        (token_ids->dtype != DType::I32 || token_ids->ne[0] != rows || token_ids->ne[1] != 1 ||
         token_ids->ne[2] != 1 || token_ids->ne[3] != 1 || !token_ids->is_contiguous() ||
         token_ids->data == nullptr)) {
        throw std::invalid_argument("shortlist_argmax: token_ids must be I32 [R]");
    }
    if (t == 0) { return; }
    if (coarse.data == nullptr || x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("shortlist_argmax: tensors must be allocated");
    }
    const auto block = gguf_block_shape(exact.qtype);
    const auto scope = workspace.scope();
    const DeviceSpan scratch =
        workspace.alloc_bytes(gguf::shortlist_argmax_bytes(rows, exact.k, t, candidates), 256);
    gguf::shortlist_argmax(
        detail::gguf_type(exact.qtype), exact.qdata,
        std::int64_t(exact.k / block.elements) * block.bytes, rows, exact.k,
        static_cast<const __nv_bfloat16*>(coarse.data), rows,
        static_cast<const __nv_bfloat16*>(x.data), t, candidates,
        token_ids != nullptr ? static_cast<const std::int32_t*>(token_ids->data) : nullptr,
        static_cast<std::int32_t*>(out.data), scratch.data, stream);
}

} // namespace ninfer::ops
