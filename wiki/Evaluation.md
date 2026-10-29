# Evaluation

**Status: running.** No capability scores exist yet for this engine with the IQ3_XXS artifact,
or for NInfer-all on this card. The tables below are placeholders.

## Protocol

The repository's evaluation coordinator ([`eval/`](../eval/README.md)) runs EvalScope 1.10.0
against the server's OpenAI API. The 16 GB card's configuration
([`eval/configs/qwen3_8_27b_gsq_rco_5060ti.yaml`](../eval/configs/qwen3_8_27b_gsq_rco_5060ti.yaml))
runs each suite on its own server with the shipped settings at a 262,144-token context
(`scripts/qwen38/serve.sh --port 18080 --local-only --concurrency N`). A request reserves its prompt
plus its whole output budget in the KV pool, which holds 263-265K tokens at two or three requests,
so the budgets decide how many run at once. Nothing is truncated to make room: a request the pool
cannot hold waits. Concurrency changes speed, not the sampling or the budgets.

| Suite | Benchmark | Samples | Output budget | Requests at once |
|---|---|---:|---:|---:|
| `ifbench` | IFBench | 300 | 65,536 | 3 |
| `aime` | AIME 2025 | 30 | 122,880 | 2 |
| | AIME 2026 | 30 | 122,880 | 2 |
| `gpqa` | GPQA-Diamond | 198 | 245,760 | 1 |
| `livecodebench` | LiveCodeBench v6 (`release_v6`, problems dated February to April 2025) | 131 | 122,880 | 2 |

This engine's IFBench ran one request at a time, before the suites were split.

- **Sampling.** Qwen's thinking-mode settings: thinking on, temperature 1.0, top-p 0.95, top-k 20,
  min-p 0, no presence penalty, seed 42. An answer cut off by its budget counts as wrong. The
  budgets are those of NInfer-all's reasoning suite, whose scores the results table cites; Qwen's
  card states no per-benchmark budget and recommends up to 262,144 reasoning tokens.
- **Scoring.** Every benchmark is scored by rule or by running code; none needs an LLM judge.
  Every response is kept (`samples.retention: all`).
- **LiveCodeBench.** The window is the one Qwen's earlier cards name for "v6" (2025.02-2025.05); the
  Qwen3.8 card does not state one. Generated code runs only in EvalScope's Docker sandbox. Without
  a Docker daemon the responses are kept and the samples are reported failed and unscored until
  they are scored in the sandbox.
- **Not run.** HLE needs an LLM judge (Qwen used GPT-4o), and its full set would take weeks at one
  request at a time. The agentic and multimodal benchmarks need harnesses this repository does not
  have, and the published artifact has no vision component.

NInfer-all's `master` cannot start at 262,144 tokens with this artifact on this card (see
[Performance](Performance.md)). It runs the same configuration at the largest context with a
comparable memory margin, 221,184 tokens, with this line's flags for each concurrency. Where it
cannot start with several requests it serves one, and the others wait. Both engines clamp a
request's output budget to the context its prompt leaves, so GPQA-Diamond's 245,760-token budget
becomes about 218,000 there.

## Results

| Benchmark | NInfer-Qwen (IQ3_XXS, RTX 5060 Ti) | NInfer-all (IQ3_XXS, RTX 5060 Ti) | Qwen3.8-27B card |
|---|---:|---:|---:|
| IFBench | *pending* | *pending* | 79.5 |
| AIME25 | *pending* | *pending* | — |
| AIME26 | *pending* | *pending* | — |
| GPQA-Diamond | *pending* | *pending* | 89.2 |
| LiveCodeBench v6 | *pending* | *pending* | 90.3 |

The card's scores are Qwen's own runs of the full-precision model with its own harness.

### Published references

These scores come from NInfer-all's [GGUF guide](../docs/gguf.md), measured on an RTX 5090. They
use the **IQ3_S** GSQ-RCO quantization, which is larger than the IQ3_XXS this engine serves. They
show what the model family reaches, not what this artifact will score.

| Benchmark | GSQ-RCO IQ3_S on NInfer-all | Official Qwen3.8-27B NInfer artifact |
|---|---:|---:|
| IFBench | 80.33 | 77.67 |
| AIME25 | 100 | 96.67 |
| AIME26 | 100 | 96.67 |
| GPQA-Diamond | 88.38 | 87.37 |
