# Evaluation

**Status: in progress.** IFBench and AIME 2025 are done; AIME 2026, GPQA-Diamond and LiveCodeBench
are still to run.
NInfer-all is compared on speed only
([Performance](Performance.md)); its capability scores are not measured on this card.

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
  budgets are those of NInfer-all's reasoning suite, whose published scores are listed below; Qwen's
  card states no per-benchmark budget and recommends up to 262,144 reasoning tokens.
- **Scoring.** Every benchmark is scored by rule or by running code; none needs an LLM judge.
  Every response is kept (`samples.retention: all`).
- **LiveCodeBench.** The window is the one Qwen's earlier cards name for "v6" (2025.02-2025.05); the
  Qwen3.8 card does not state one. Generated code runs only in EvalScope's Docker sandbox. Without
  a Docker daemon the responses are kept and the samples are reported failed and unscored until
  they are scored in the sandbox.
- **Not run yet.** HLE needs an LLM judge (Qwen used GPT-4o) and is planned for later. The agentic
  and multimodal benchmarks need harnesses this repository does not have, and the published
  artifact has no vision component.

## Results

| Benchmark | NInfer-Qwen (IQ3_XXS, RTX 5060 Ti) | Qwen3.8-27B card |
|---|---:|---:|
| IFBench | 79.33 | 79.5 |
| AIME25 | 93.33 | — |
| AIME26 | *pending* | — |
| GPQA-Diamond | *pending* | 89.2 |
| LiveCodeBench v6 | *pending* | 90.3 |

The card's scores are Qwen's own runs of the full-precision model with its own harness.

- **IFBench.** The score is prompt-level strict accuracy, as in the references below. All 300
  answers finished within the budget; the replies had a median of 4,231 output tokens, 11 used more
  than 32,768 and the longest used 64,074. Instruction-level strict accuracy was 81.83, and the
  loose scores were 83.67 (prompt) and 85.83 (instruction). The run took 8.2 hours at one request
  at a time.
- **AIME 2025.** 28 of 30 correct. One reply finished with a wrong answer (848 for AIME II
  problem 13); the other reached the 122,880-token budget while still working on AIME II problem
  15, so it has no answer. The replies had a median of 19,642 output tokens. The run took 1.6 hours at two
  requests at a time.

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
