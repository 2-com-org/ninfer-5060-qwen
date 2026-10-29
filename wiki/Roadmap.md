# Roadmap

## Open work

1. **Evaluations.** Run the EvalScope suite (IFBench, AIME25/26, GPQA-Diamond) on this engine,
   and fill in [Evaluation](Evaluation.md).
2. **NInfer-all baseline.** Build the current NInfer-all `master` for `sm_120a`. Measure it on the
   same card with the same artifact and settings, and fill in the placeholders in
   [Performance](Performance.md) and the README.
3. **Release.** Package only what running and serving need: `ninfer-serve` and its runtime, the
   launch scripts, the WebUI and the run documentation. Test harnesses, benchmarks and evaluation
   tooling stay out.
4. **Prebuilt model.** Publish the converted `.ninfer` on Hugging Face. Both sources are
   Apache-2.0, so this is allowed with attribution. It would make conversion optional.
5. **Merge into `master`.** This branch conflicts with the repository's `master`, which mirrors
   NInfer-all and is 366 commits past the fork point: 141 files conflict, most of them under
   `src/models` and `src/ops`. Once resolved, verify with a rebuild, the tests, identical paper
   replies and the 30K/160K speeds.
6. **Native Linux.** Test the setup scripts outside WSL2.

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
