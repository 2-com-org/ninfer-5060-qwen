# Performance

All measurements are on one **GeForce RTX 5060 Ti 16 GB** (`sm_120a`, 448 GB/s) in a desktop
running Windows 11 + WSL2 Ubuntu 26.04, with CUDA 13.1 and driver 610.88/621.01. The display was
off during timed runs: under WSL2, display and desktop activity change the GPU memory budget (see
[GPU memory and Windows](Serving.md#gpu-memory-and-windows)).

## Method

- **Paper prompts.** The prompts are long technical papers: 30,018 tokens, and the first
  160,016 tokens of a 224K-token Navier-Stokes and Euler text. The reply is 512 tokens,
  generated greedily with thinking off at the shipped single-request settings (262,144 context,
  MTP with 4 drafts, 31 copy tokens).
- **Reply checks.** Replies are compared by sha256. A change that alters a reply is flagged,
  because decode speed depends on the reply's draft acceptance.
- **Rates.** Prefill and decode rates come from the server's own request timings: prompt
  processing up to the first token, then generation after it.
- **Run order.** Comparisons run arms alternately (A B B A, or alternating starts) on the same
  binaries and artifact. Each arm is started at least 90 s after desktop activity.

## This line (build p151, commit `88844810`)

| Workload | Result | Notes |
|---|---|---|
| 30K-token prompt, prefill | 3,644-3,648 tok/s | 3 fresh starts; the time to first token is 8.2 s |
| 30K-token prompt, decode | 88.70-88.71 tok/s | identical replies |
| 160K-token prompt, prefill | 2,143 tok/s | the time to first token is 74.7 s |
| 160K-token prompt, decode | 65.95 tok/s | 65.6-70.3 across builds whose replies differ |
| Short prompt, decode | 87.0-87.7 tok/s | builds p140-p145 |
| First request after start (14 tokens), time to first token | 102.7-104.7 ms | p150: 259-280 ms, one start 2.0 s |
| Returning turn in a 156K conversation, time to first token | 163-168 ms | the catalog cache in RAM |
| Startup, weights read to ready (warm) | 9.7-10.7 s | +6 s to pin a 22 GiB RAM cache |
| 8 requests, total | 342 tok/s | short essays, MTP with 3 drafts |
| 16 / 24 / 32 requests, total | ~430 / ~510 / ~625 tok/s | about 20 tok/s each at 32 |

## Against NInfer-all

### At the fork point (`e7b7e669`), same card and same GGUF

The fork point is upstream NInfer-all with this card's automatically calibrated route profile.

| | NInfer-all `e7b7e669` | NInfer-Qwen p151 |
|---|---:|---:|
| Largest context | 196,608 (262,144 fails: 608 MiB short) | 262,144 |
| Prefill, 18-30K prompts | 975-1,000 tok/s (64K context) | 3,644-3,648 tok/s (30K) |
| `ninfer_bench` pp2048 | 1,085 tok/s | not re-measured |
| Short-prompt decode | 75.5 tok/s (MTP 3, 64K context) | 87.0-87.7 tok/s (MTP 4, 262K context) |
| 160K-prompt prefill / decode | not measured ¹ | 2,143 / 65.95 tok/s |
| 8 requests, total | not measured ¹ | 342 tok/s |

¹ The first build of this line was the fork point plus this card's build preset and device
profile, before any kernel work. At a 196,608 context it measured 865 tok/s prefill and 62.4 tok/s
decode on the 160K prompt, and 275 tok/s total with 8 requests.

These comparisons are **not like for like**:

- the old builds could not run the shipped settings (262,144 context, 4 drafts);
- the baseline converted the MTP head as Q6_K and had no coarse proposal head;
- the 30K decode baseline was only recorded as time per round (38.3 ms).

They show where each build stood at its best measured configuration.

### Current NInfer-all `master`

*To be measured.* NInfer-all's `master` is 366 commits ahead of the fork point. A same-card,
same-artifact, same-settings run (prefill and decode at 30K and 160K, short decode, and 8 and 32
requests) is on the [Roadmap](Roadmap.md).

## Where the time goes

At 160K, prompt attention is about half of prefill time, running at about 65% of the card's tensor
peak. The weight GEMMs are most of the rest, and they now run in a warp-specialized i-quant tile
kernel.

Decode is bound by memory bandwidth:

- an MTP round streams about 11.8 GB of weights;
- the text GEMVs run at about 95% of the 393-412 GB/s this card streams in one kernel;
- host overhead is about 1% of a round.

[Changes from NInfer-all](Changes-from-NInfer-all.md) lists what each change contributed.

## Quality

Perplexity from `ninfer-perplexity`, as paired per-window deltas with standard errors:

| | Perplexity | Change |
|---|---:|---|
| BF16 KV, 64K windows (perplexity-1m, 16 streams) | 4.6011 | reference |
| `rk4v4` KV (shipped), 64K windows | 4.6155 | +0.315% ± 0.120% against BF16 |
| `rk8v4` KV, 160K windows (papers, 224,767 tokens) | 5.3112 | reference |
| `rk4v4` KV, 160K windows | 5.3012 | -0.19% ± 0.40% against `rk8v4` |
| `rk2v4-e8` KV (rejected) | | +3.5% at 64K, +9.9% at 160K |

- **NVFP4 prefill.** NVFP4 prefill activations, on by default for speed, cost +1.6% held-out
  perplexity. `-- --no-prefill-nvfp4` prefills with MXFP8 activations instead, which is slower.
- **Exact kernels.** Every kernel change was qualified against an FP32/FP64 oracle under the
  repository's [Op qualification contract](../docs/maintainer/op-development.md). Changes meant to
  be exact (GDN replay, value decode, startup module loading) were checked for bit-identical
  outputs.
- **Capability evaluations** are pending: see [Evaluation](Evaluation.md).

## Reproducing

The published path reproduces these results. A fresh clone of `qwen38-5060ti` (`d214185d`) was
converted by `prepare-model.sh` and built by `build.sh`, and it gave the same replies as p151 on
both papers:

| Prompt | Prefill | Decode |
|---|---:|---:|
| 160K | 2,148 tok/s | 66.0 tok/s |
| 30K | 3,484 tok/s | 88.6 tok/s |

The 30K prefill is a single first prompt on a fresh server. `serve.sh` and the Windows launcher
each started at 262,144 tokens, with 233 MiB free, and answered a request.

The test and benchmark workflows are documented in [tests/README.md](../tests/README.md) and
[bench/README.md](../bench/README.md).

- Build with the `rtx5060-dev` preset to get `ninfer_bench`, the Op benchmarks and the tests.
- `ninfer-perplexity` is built by `scripts/qwen38/build.sh`;
  [docs/perplexity.md](../docs/perplexity.md) describes its corpora and options.
- Serving measurements post the paper prompts through the OpenAI API and read the server's
  request log (`-- --request-log-jsonl FILE`) for prefill, decode, rounds and acceptance.
