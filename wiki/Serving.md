# Serving

```bash
scripts/qwen38/serve.sh [options] [-- more ninfer-serve arguments]
```

`serve.sh` turns a few choices into the `ninfer-serve` command line that was measured on the RTX
5060 Ti 16 GB. `--print` shows that command without running it. Arguments after `--` go to
`ninfer-serve` unchanged; the full reference is [docs/serving.md](../docs/serving.md).

## Endpoints

| URL | |
|---|---|
| `http://localhost:8080/` | WebUI (llama.cpp's chat app) |
| `http://localhost:8080/monitor/` | live monitor: throughput, acceptance, time to first token, KV and cache use, request history |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, JSON schema / regex / GBNF constraints, logprobs |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages |
| `POST /completion`, `/tokenize`, `/apply-template` | llama.cpp-compatible endpoints |
| `GET /health`, `/v1/models`, `/stats`, `/metrics` | health, model info, counters, Prometheus metrics |

The model id is `qwen3.8-27b`. By default the server listens on every network interface and
accepts any API key. On a network you do not trust, pass `--local-only`, or require a key with
`-- --api-key KEY`; clients then send it as an OpenAI bearer token or an Anthropic `x-api-key`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Explain GDN attention in two sentences."}],
    extra_body={"enable_thinking": True},  # per-request thinking
)
print(reply.choices[0].message.content)
```

Claude Code, Codex and other agents can use the Anthropic or OpenAI base URL directly. Thinking is
off by default: turn it on per request (`enable_thinking`, or Anthropic's `thinking`) or for the
whole server with `--thinking`.

## Options and why they are set

### One request (default)

| Setting | Value | Measured reason |
|---|---|---|
| Context | 262,144 | It fits with about 270 MiB of GPU memory to spare. |
| KV cache | `rk4v4`: 4-bit Lloyd-Max keys, 4-bit values | +0.3% perplexity against BF16 KV, and needed for the context to fit. |
| GDN state | FP16 | |
| Token embedding | host memory (`--embedding-host`) | Frees about 1 GiB of the card. |
| Prefill chunk | 2,304 | The automatic 4,608 left only 136 MiB, and Windows paged GPU memory when the desktop took more. 2,304 tolerates about 128 MiB of later use and is 2% slower on a 160K prompt. |
| Speculation | MTP, 4 drafts | 4 drafts beat 3 by +1.2% short and +3% at 30K and 160K. 5 or more cause GPU paging at 262,144. |
| Copy tokens | 31 | Returning edited code ran 455 vs 347 tok/s against 15. Prose is unchanged. 63 costs 108 MiB more. |
| Progress anchors | every 27,648 tokens | A cancelled long prefill resumes from the last anchor. The spacing is a multiple of every chunk used. |
| CUDA Graph allowance | 32 MiB | The measured need; the engine's default reserves 13-15 times that. |

### Several requests (`--concurrency N`)

Requests share one KV pool, sized to the free GPU memory after startup. Each lane holds its own
model state, so the pool shrinks as lanes are added. One request's prompt is processed while the
others generate.

The pool sizes below were measured with 262,144-token requests at 2-3 lanes and 196,608 at 4-8:

| Requests | Default context per request | Pool (measured) | Prefill chunk | Drafts / copy tokens |
|---|---:|---:|---:|---|
| 1 | 262,144 | dedicated | 2,304 | 4 / 31 |
| 2-3 | 262,144 | 263-265K tokens | 2,304 (2), 1,152 (3) | 3 / 15 |
| 4-8 | 196,608 | 202-236K | 2,304 | 3 / 15 |
| 9-12 | 131,072 | 162-193K | 3,456 | 3 / 15 |
| 13-16 | 98,304 | 121K | 3,456 | 3 / 15 |
| 17-24 | 98,304 | 109-161K | 3,456 | 3 / 7 |
| 25-32 | 98,304 | 101-136K | 3,456 | 3 / 0 |

- **Admission.** A request reserves its prompt plus its whole output budget, and it queues while
  the pool lacks those pages. A request without `max_tokens` gets the largest budget within its
  1/N share of the pool.
- **Lease growth.** `--lease-growth` instead reserves the prompt plus 4,096 tokens and grows the
  reservation between rounds. An answer that cannot grow ends with `finish_reason=length`.
- **Throughput.** Total throughput rises with lanes while each request slows down. With short
  requests and every lane busy, the totals were about 82 tok/s at 1 request, 342 at 8, 430 at 16,
  510 at 24 and 625 at 32 (about 20 tok/s each).
- **Startup.** Startup takes up to 25 s longer at 16 or more requests.

### Conversation cache (`--ram-cache-gib`)

Finished conversations keep their KV and model state in pinned system RAM. A returning
conversation continues from where it stopped instead of reprocessing its history. Nothing is
written to disk.

The default is 22 GiB, or 45% of RAM when that is less. The budget holds up to 64 conversations,
and up to 128 model states (75 MiB each) take half of it.

Measured with 12 conversations of 8-28K-token documents over 4 turns:

| Setting | Average first token of a returning turn |
|---|---|
| 3 concurrent requests, 21-24 GiB | 0.9-1.0 s |
| 3 concurrent requests, engine defaults | 11.0 s |
| 1 request, 6 conversations, this cache | 0.19 s |
| 1 request, 6 conversations, engine defaults | 4.1 s |

Pinning 22 GiB adds about 6 s to startup.

### Shared prompt prefixes (`--prefix-cache alt`)

With 2-8 requests, a request whose prompt starts like one still being processed can wait for that
work and reuse it. One example is an orchestrator's subagents that share a system prompt.

With a 30K-token shared system prompt, two to four such requests sent together finished in 11-13 s
instead of 19-21 s. The option costs a smaller pool and a maximum context of 196,608, or 131,072
at 5 or more requests.

### Sampling and thinking

`--temperature` (default 0, greedy) and `--thinking` (default off) are server defaults; requests
override both. Qwen recommends temperature 1.0, top-p 0.95 and top-k 20 when thinking.
`--reasoning-loop stop|conclude` (passed after `--`) ends thinking that repeats itself.

## Windows launcher

`scripts\qwen38\windows\Start-Qwen38.cmd` (`windows\Start-Qwen38.cmd` in the Linux release archive)
runs `serve.sh` inside WSL from a Windows console. The Windows release's `Start-Qwen38.cmd` and its
Start menu entry run the same launcher inside the `ninfer-5060-qwen` distribution and take the same
parameters. It:

- shows the server log and saves it under `%LOCALAPPDATA%\ninfer-5060-qwen\logs`;
- waits for `/health`, then opens the WebUI;
- prints the addresses other devices on the network can use (unless `-LocalOnly`);
- stops the server when the window closes or Ctrl+C is pressed. A Windows job object and a
  lifetime pipe make sure no server is left behind.

Its parameters mirror `serve.sh`: `-Context`, `-Concurrency`, `-Spec`, `-DraftTokens`,
`-CopyTokens`, `-PrefixCache`, `-LeaseGrowth`, `-RamCacheGiB`, `-Temperature`, `-Thinking`, `-Port`,
`-LocalOnly`, `-Model` and `-Server`. It adds three of its own:

- `-NoBrowser`;
- `-Dashboard`, a fuller monitor (`tools/monitor/monitor.py`, in a checkout only) at
  <http://localhost:8090>;
- `-Distribution`, which picks the WSL distribution. By default it is the one holding the
  checkout or release.

```powershell
\\wsl.localhost\Ubuntu\home\<you>\ninfer-5060-qwen\scripts\qwen38\windows\Start-Qwen38.cmd -Concurrency 4 -LocalOnly
```

For a desktop shortcut, point one at `Start-Qwen38.cmd`.

## GPU memory and Windows

The 262,144 settings leave about 250-300 MiB of the 16 GB free. Under WSL2 the Windows driver
(WDDM) silently pages GPU memory to system RAM when other programs take more: games, a browser
drawing on this GPU, other CUDA programs. Every kernel then slows by 20-60%, and decode by 15-45%.

If speed drops:

- close other GPU programs;
- view the WebUI from another device;
- or pass a smaller `--context`.

`nvidia-smi dmon -s t` showing PCIe receive traffic during decode is the telltale sign. See
[Troubleshooting](Troubleshooting.md).
