# NInfer-Qwen

NInfer-Qwen serves **Qwen3.8-27B GSQ-RCO IQ3_XXS** at its full **262,144-token context** on a
single **GeForce RTX 5060 Ti 16 GB**. It is a C++/CUDA engine forked from
[NInfer-all](https://github.com/iamwavecut/ninfer-all), which descends from
[NInfer](https://github.com/Neroued/ninfer), and it is tuned for this one model on this one card.

## At a glance

| | |
|---|---|
| Model | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), ISTA-DASLab's [GSQ-RCO IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) GGUF with its MTP head (10.4 GB) |
| Card | RTX 5060 Ti 16 GB (`sm_120a`), Windows 11 + WSL2 |
| Context | 262,144 tokens with one to three requests; up to 32 requests with smaller contexts |
| Speed (one request) | 3,646 tok/s prefill and 88.7 tok/s decode on a 30K-token prompt; 2,143 and 66.0 at 160K |
| APIs | OpenAI Chat Completions and Responses, Anthropic Messages, llama.cpp; WebUI and monitor |
| Features | MTP speculative decoding, n-gram copy drafts, 4-bit KV cache, pinned-RAM conversation cache, tool calls, structured output (JSON schema, regex, GBNF), reasoning-loop guard |

## Pages

- [Installation](Installation.md): the prebuilt release, or WSL2, CUDA and the build.
- [Model preparation](Model-Preparation.md): sources, licenses and the conversion.
- [Serving](Serving.md): launch options, concurrency, RAM cache, the Windows launcher and API
  examples.
- [Performance](Performance.md): measurements, method, quality and the comparison with
  NInfer-all `master`.
- [Evaluation](Evaluation.md): the capability evaluation protocol and results (IFBench and AIME
  2025 so far).
- [Changes from NInfer-all](Changes-from-NInfer-all.md): what this line adds and removes.
- [Troubleshooting](Troubleshooting.md)
- [Roadmap](Roadmap.md): open work and the parked optimisation queue.

## Quick start

```bash
git clone https://github.com/2-com-org/ninfer-5060-qwen.git ~/ninfer-5060-qwen
```

```bash
~/ninfer-5060-qwen/scripts/qwen38/setup.sh
```

```bash
~/ninfer-5060-qwen/scripts/qwen38/serve.sh
```

Once it is running, open <http://localhost:8080> for the WebUI or use the API at
`http://localhost:8080/v1`.
