#pragma once

#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/shortlist_argmax.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

// The proposal head's draft for each column of `hidden` [H, T]: the argmax of its logits over its
// rows (the first `domain` when the head is not indexed), mapped through its token IDs. A coarse
// head ranks the rows first and the head rescores only its top kProposalCandidates.
// `logits` [rows, T] receives the head's logits, or the coarse head's.
inline void propose(const Tensor& hidden, const ProposalParameters& p, Tensor& logits,
                    Tensor& tokens, std::int32_t domain, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    const Tensor* ids = p.token_ids ? &*p.token_ids : nullptr;
    if (p.coarse_head) {
        project(hidden, *p.coarse_head, logits, workspace, stream);
        ops::shortlist_argmax(logits, hidden, p.head.weight, kProposalCandidates, ids, tokens,
                              workspace, stream);
        return;
    }
    project(hidden, p.head, logits, workspace, stream);
    ops::argmax(logits, tokens, ids ? dimension(p.rows) : domain, stream);
    if (ids) {
        ops::proposal_remap_token_ids(tokens, static_cast<const std::int32_t*>(ids->data),
                                      dimension(p.rows), stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
