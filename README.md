# NInfer-Qwen

**Qwen3.8-27B at its full 262,144-token context on a 16 GB GeForce RTX 5060 Ti.**

NInfer-Qwen is a from-scratch C++/CUDA inference engine for one model on one card. It is a fork
of [NInfer-all](https://github.com/iamwavecut/ninfer-all), itself a continuation of
[NInfer](https://github.com/Neroued/ninfer). The model is
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in ISTA-DASLab's
[GSQ-RCO IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) quantization, and
the card is the RTX 5060 Ti 16 GB (Blackwell, `sm_120a`). Every kernel route, memory setting and
launch default here was measured on that card.

The server provides:

- the OpenAI Chat Completions and Responses APIs, the Anthropic Messages API and the llama.cpp API;
- a built-in WebUI with a live monitor;
- speculative decoding with the model's own multi-token-prediction (MTP) head, plus n-gram copy
  drafts for text a reply repeats;
- a 4-bit KV cache (`rk4v4`);
- a pinned-RAM cache that keeps finished conversations, so a returning conversation skips its
  history;
- up to 32 requests at once.

## Performance

Measured on an RTX 5060 Ti 16 GB under Windows 11 + WSL2 (CUDA 13.1, driver 610.88/621.01),
display off. Generation is greedy, one request, at the shipped launch settings (262,144 context,
MTP with 4 drafts and 31 copy tokens).

The prompts are the start of a mathematical paper: 30,018 and 160,016 tokens, with 512 output
tokens. Full method and raw ranges are on the [Performance](wiki/Performance.md) page.

| | **NInfer-Qwen** (this branch) | NInfer-all at the fork point,<br>same card and model | NInfer-all `master`,<br>same card and model |
|---|---:|---:|---:|
| Largest context that starts | **262,144** | 196,608 (262,144 is 608 MiB short) | 233,472 with no memory left (262,144 is 493 MB short) |
| Prefill, 30K-token prompt | **3,644-3,648 tok/s** (8.2 s) | 975-1,000 tok/s ¹ | 873-942 tok/s (32-34 s) ⁶ |
| Prefill, 160K-token prompt | **2,143 tok/s** (75 s) | 865 tok/s ² | 680-693 tok/s (231-235 s) ⁶ |
| Decode after the 30K prompt | **88.7 tok/s** | ~70 tok/s (38.3 ms per round) ³ | 72.3 tok/s ⁶ |
| Decode after the 160K prompt | **66.0 tok/s** | 62.4 tok/s ² | 58.1-58.4 tok/s ⁶ |
| Decode, short prompt | **87.0-87.7 tok/s** ⁴ | 75.5 tok/s ³ | 71.5-72.0 tok/s ⁶ |
| Time to first token, 14-token first request | **103-105 ms** | not measured | 1.52-1.57 s ⁶ |
| 8 / 32 requests at once, total decode | **342 / ~625 tok/s** ⁵ | 275 tok/s ² / limit 8 | ~240 tok/s ⁷ / limit 8 |

¹ 18-30K-token prompts at a 65,536 context. ² The first build of this line: the fork point plus
this card's build preset and device profile, before any kernel work. The fork point itself cannot
start at these settings. ³ At a 65,536 context with 3 drafts; the 30K rate is estimated from its
measured time per round and the short prompt's 2.72 tokens per round. ⁴ Measured on builds p140-p145;
later commits did not change decode. ⁵ Short essays, every lane busy, MTP with 3 drafts.
⁶ `master` `33163f05` at a 221,184-token context, which leaves it a free-memory margin like this
branch's at 262,144, with this branch's single-request settings except `--embedding-host` (master
lacks it). Same artifact, a fresh server per run, alternating with this branch's fresh-clone build,
which measured 3,209-3,647 and 2,081-2,147 tok/s prefill, 88.2-88.7, 66.0 and 87.8 tok/s decode
and 117-124 ms to the first token in the same runs. ⁷ At a 163,840-token context (196,608 does not
start). Only 7 of the 8 requests decoded together; the 8th waited 29.9 s. Eight 1,024-token
replies took 45.9 s, against 26.2 s on this branch.

The fork-point column is not like for like: those builds could not run the shipped settings, so it
shows their best measured configuration. The `master` column uses the same artifact and settings,
apart from the context and `--embedding-host`. [Performance](wiki/Performance.md#current-ninfer-all-master)
has the per-round times and draft acceptance.

**Quality.** The 4-bit KV cache costs +0.32% ± 0.12% perplexity against a BF16 cache in 64K
windows, and matches the 8-bit `rk8v4` cache within 0.4% in 160K windows. Prefill with NVFP4
activations (on by default; `--no-prefill-nvfp4` turns it off) costs +1.6% held-out perplexity.
Details are on [Performance](wiki/Performance.md#quality).

### Evaluations

IFBench is done; the other evaluations are running. They use the repository's EvalScope suite
([`eval/`](eval/README.md)) with Qwen's thinking-mode settings (temperature 1.0, top-p 0.95,
top-k 20) at a 262,144-token context, running as many requests at once as the KV pool holds whole
output budgets: three for IFBench, two for AIME and LiveCodeBench, one for GPQA-Diamond. Every
benchmark is scored by rule or by running code in a Docker sandbox; none needs an LLM judge.

| Benchmark | NInfer-Qwen, IQ3_XXS | Qwen3.8-27B card | GSQ-RCO IQ3_S on NInfer-all (RTX 5090) | Official Qwen3.8-27B NInfer artifact |
|---|---:|---:|---:|---:|
| IFBench | 79.33 | 79.5 | 80.33 | 77.67 |
| AIME25 | *pending* | — | 100 | 96.67 |
| AIME26 | *pending* | — | 100 | 96.67 |
| GPQA-Diamond | *pending* | 89.2 | 88.38 | 87.37 |
| LiveCodeBench v6 | *pending* | 90.3 | — | — |

The card's scores are Qwen's own runs of the full-precision model. The last two columns come from
[docs/gguf.md](docs/gguf.md); the IQ3_S column is a larger quantization than the IQ3_XXS this
engine serves. See [Evaluation](wiki/Evaluation.md) for the protocol.

## Requirements

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti **16 GB** (tested). Other RTX 50 cards build and run but are not tuned. RTX 30/40 builds compile but are unmeasured with this model. |
| OS | Windows 11 with WSL2 Ubuntu (tested), or native Ubuntu 24.04 or newer (not yet tested) |
| Driver and toolkit | NVIDIA driver 580 or newer; CUDA Toolkit **13.1 or newer** (12.8/12.9 miscompile `sm_120a`) |
| Host | CMake 3.28+, Ninja, GCC 13-15, FFmpeg and libcurl development packages (`install-deps.sh` installs them) |
| Memory | 32 GB of RAM or more recommended: the conversation cache pins up to 22 GiB (45% of RAM when that is less) |
| Disk | About 30 GB: 10.4 GB of sources and a 10.8 GB model (the sources can be deleted afterwards), plus the build tree |

## Quick start

Inside Ubuntu (under WSL2 on Windows: `wsl --install -d Ubuntu`, then open Ubuntu), clone into a
path without spaces:

```bash
git clone --branch qwen38-5060ti https://github.com/2-com-org/ninfer-qwen.git ~/ninfer-qwen
```

```bash
~/ninfer-qwen/scripts/qwen38/setup.sh
```

`setup.sh` does three things, each also available as its own script:

1. Installs the build packages and a Python environment for the converter (`install-deps.sh`).
2. Builds the server for your GPU (`build.sh`, 30-60 minutes).
3. Downloads the 10.4 GB GGUF and the base model's config files at pinned revisions, checks
   them, and converts them into `models/qwen38-27b-gsq-iq3xxs.ninfer` (`prepare-model.sh`,
   about 10 minutes on the CPU, with 11 GB of RAM).

To skip step 3, run `install-deps.sh` and `build.sh` on their own, then download the converted
artifact from [Hugging Face](https://huggingface.co/FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer)
(10.8 GB):

```bash
~/ninfer-qwen/.venv-convert/bin/hf download FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer qwen38-27b-gsq-iq3xxs.ninfer --local-dir ~/ninfer-qwen/models
```

Install the CUDA Toolkit first: [NVIDIA's downloads](https://developer.nvidia.com/cuda-downloads)
have a "WSL-Ubuntu" package for WSL2. Under WSL2, do not install a Linux driver.

Start the server:

```bash
~/ninfer-qwen/scripts/qwen38/serve.sh
```

The server takes 10-30 seconds to start. Then:

- **WebUI:** <http://localhost:8080>
- **Monitor:** <http://localhost:8080/monitor/>
- **OpenAI-compatible API:** `http://localhost:8080/v1`, model `qwen3.8-27b`

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

On Windows, double-click `scripts\qwen38\windows\Start-Qwen38.cmd`. In Explorer, the checkout is
at `\\wsl.localhost\Ubuntu\home\<you>\ninfer-qwen`. The launcher starts the server inside WSL,
shows its log, opens the WebUI, and stops the server when you close the window. It takes the same
options as `serve.sh`, for example `Start-Qwen38.cmd -Concurrency 4`.

## Common options

`serve.sh --help` lists every option. Each default and limit below was measured on the
16 GB card; [Serving](wiki/Serving.md) explains them.

| Option | Default | |
|---|---|---|
| `--context N` | 262,144 | tokens per request; lower it with more requests |
| `--concurrency N` | 1 | 1-32 requests at once; above 3 the context shrinks (196,608 up to 8, 131,072 up to 12, 98,304 above) |
| `--thinking` | off | think by default (requests can override) |
| `--temperature X` | 0 | default sampling temperature (requests can override) |
| `--ram-cache-gib N` | 22 | pinned RAM for finished conversations |
| `--local-only` | off | listen on 127.0.0.1 only |
| `--prefix-cache alt` | catalog | with 2-8 requests: concurrent requests share a common prompt prefix |

## Documentation

The `wiki/` folder holds the project wiki:

- [Home](wiki/Home.md)
- [Installation](wiki/Installation.md)
- [Model preparation](wiki/Model-Preparation.md)
- [Serving](wiki/Serving.md)
- [Performance](wiki/Performance.md)
- [Evaluation](wiki/Evaluation.md)
- [Changes from NInfer-all](wiki/Changes-from-NInfer-all.md)
- [Troubleshooting](wiki/Troubleshooting.md)
- [Roadmap](wiki/Roadmap.md)

The engine's reference documentation is inherited from NInfer-all and lives in [docs/](docs/README.md):

- architecture, Op qualification, KV and context caches;
- CLI and serving references;
- the upstream README, as [docs/ninfer-all.md](docs/ninfer-all.md).

## Model weights

The engine serves a `.ninfer` artifact converted from two sources, both Apache-2.0:

- the GSQ-RCO GGUF, whose blocks are kept exactly as stored;
- the base model's tokenizer and config.

The conversion makes two changes. It requantizes the MTP head from Q6_K to IQ4_XS
([override](tools/convert/overrides/qwen3_8_mtp_iq4xs.py)), and it adds a coarse IQ2_XXS proposal
head for the shortlisted output layer. [Model preparation](wiki/Model-Preparation.md) has the exact
command.

The artifact this conversion makes is published on Hugging Face as
[FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer](https://huggingface.co/FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer),
with its conversion record, under the sources' Apache-2.0 license.

## License and credits

Apache-2.0 ([LICENSE](LICENSE)). This engine builds on:

- **NInfer** by Neroued, and **NInfer-all** by iamwavecut and its contributors
  ([CONTRIBUTORS.md](CONTRIBUTORS.md)): the engine.
- **ISTA-DASLab** for GSQ-RCO and the quantized GGUF ([arXiv:2604.18556](https://arxiv.org/abs/2604.18556),
  [arXiv:2605.00649](https://arxiv.org/abs/2605.00649)).
- **The Qwen team** for Qwen3.8-27B.
- **ggml / llama.cpp** for the vendored quantization formats, the MMQ kernels and the WebUI.
