# Roadmap

## Open work

1. **Evaluations** (paused). IFBench and AIME 2025 are done. AIME 2026, GPQA-Diamond and
   LiveCodeBench v6 remain; [Evaluation](Evaluation.md) records the protocol. Later: HLE's
   text-only questions with a budget of at least 122,880 tokens, the responses judged separately.
2. **Native Linux.** Test the setup scripts and the release archive outside WSL2.

## From NInfer-all, not yet ported

This line merged NInfer-all's `master` on 2026-10-31 to replace it. The 254 commits that `master` had
beyond this branch, and 38 newer upstream ones, were reviewed one by one: the fixes that reach this
model were ported, and
the rest serve other model families, other cards, tensor parallelism or CI. NInfer-all's typed
decisions API (`/v1/systemone`, `/v1/decisions`) was left out. These are worth porting later; each
needs a measurement or a port into this line's tuned kernels first:

| Commits | Change | Before taking it |
|---|---|---|
| `0020fc44` | GGUF prefill: BF16 stores fused after the stream-K reduction (+1.5% Op on an RTX 3090) | measure; this line's prompts mostly run the NVFP4 two-pass route |
| `b4081ed4`, `809f0818` | GGUF decode: cooperative Q4_K/Q6_K vector loads, chosen per card | calibrate on this card |
| `6826ac0e` | Startup: pin host memory on several threads (one thread pins ~1.5 GiB/s; the launcher pins 22 GiB) | measure startup |
| `ddacb967` | Identical replies whether or not `max_tokens` cuts an n-gram copy | port into the tuned small-T attention kernels |
| `3cb79cdf`, `ffaa13f9` | `--prefill-stall-budget-ms`: split a prefill step while other requests generate | measure with several requests |
| `2013d01f`, `8e196922` | FP32 vocabulary-head logits, so near-equal logits stop tying in BF16 | changes replies and draft acceptance; measure both |

## Parked optimisations

Ranked for decode and multi-request throughput first, then long-prompt prefill. The gains are
estimates from earlier measurements; none of these has been started.

| # | Idea | Expected effect |
|---|---|---|
| 1 | A linear route for 33-96 columns. Verifying more than 32 tokens hits a cliff of about 30 ms per round, so 9-16 requests with MTP cost as much as 16. | Up to ~40% of round time in that band; nothing with one request |
| 2 | Batch-cost gate or ragged packing for copying lanes: one lane copying text widens the whole batch past 32 columns | Concurrency throughput |
| 3 | Mixed tiny-prefill and decode pass (≤32 columns) | Concurrency time to first token and throughput |
| 4 | Prompt-attention redesign: attention is ~50% of 160K prefill at ~65% of tensor peak | ~+10% prefill at 160K, +3-4% at 30K (low confidence) |
| 5 | Skip MTP drafting during validated copy continuation | ~5-7% of a copy round; nothing on prose |
| 6 | A non-spilling 6×2 tier for `rk4v4` widths 6-8 past 8K | ~0.1 ms per launch; more when Windows pages GPU memory |
| 7 | FP8 register-value path for widths 6-8; WDDM margin profiles; harness fixes | Small |
| 8 | Remove the turn-checkpoint split | ~50 ms time to first token per short returning turn |
| 9 | KV format by context length; a model-ranked shortlist (≤0.5% decode); a long-window perplexity guard | Small or quality-only |
