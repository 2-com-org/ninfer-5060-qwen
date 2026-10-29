# Evaluation

**Status: not run yet.** No capability scores exist yet for this engine with the IQ3_XXS artifact,
or for NInfer-all on this card. The tables below are placeholders.

## Plan

The repository's evaluation coordinator ([`eval/`](../eval/README.md)) runs EvalScope 1.10.0
against the server's OpenAI API. The GSQ-RCO suite
([`eval/configs/qwen3_8_27b_gsq_rco_reasoning.yaml`](../eval/configs/qwen3_8_27b_gsq_rco_reasoning.yaml))
covers four benchmarks:

- IFBench;
- AIME 2025 and AIME 2026;
- GPQA-Diamond.

It runs with thinking on, temperature 1.0, top-p 0.95, top-k 20, seed 42, and output budgets up to
122,880 tokens.

The configuration was written for an RTX 5090: four concurrent requests at a 262,144-token
context. On the 16 GB card, four requests fit at most 196,608 tokens each. The run needs one of two
adjustments:

- four requests at 196,608 tokens;
- or two requests at 262,144 tokens, slower overall.

Its output budgets need checking against those contexts.

To compare like with like, both engines run the same artifact and sampling settings:

- **NInfer-Qwen**: `scripts/qwen38/serve.sh --concurrency N --context C --temperature 1 --thinking`.
- **NInfer-all**: its current `master`, built for `sm_120a`, serving the same `.ninfer` with its
  own measured settings for this card.

EvalScope and its datasets are downloaded on first use (`eval/requirements.txt`). Expect about a
day of GPU time per engine for the full text suite at this card's speed.

## Results

| Benchmark | NInfer-Qwen (IQ3_XXS, RTX 5060 Ti) | NInfer-all (IQ3_XXS, RTX 5060 Ti) |
|---|---:|---:|
| IFBench | *pending* | *pending* |
| AIME25 | *pending* | *pending* |
| AIME26 | *pending* | *pending* |
| GPQA-Diamond | *pending* | *pending* |

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
