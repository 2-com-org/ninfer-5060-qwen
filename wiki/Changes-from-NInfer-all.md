# Changes from NInfer-all

This line branches from NInfer-all at `e7b7e669` and is developed on the branch `qwen38-5060ti`.
It stays a branch: this repository's `master` mirrors NInfer-all. On 2026-10-31 the branch merged
that `master`, keeping its own tree. It adds two kinds of commits:

- this line's own work, including seven commits ported from the author's earlier RTX 5060 Ti fork
  (5060ninfer);
- 34 fixes and features picked from NInfer-all after the fork point: 23 while the branch was
  developed and 11 at that merge.

Kernel changes were qualified against FP32/FP64 oracles
([Op development](../docs/maintainer/op-development.md)) and measured end to end on the RTX
5060 Ti. Commit messages hold each change's measurements.

## Scope

- **Removed:** the Gemma 4 and Qwen3.8-Flash-Next model families, and their uncalled Ops (`d3274706`,
  `6146b5a2`). Their documents remain in [docs/](../docs/README.md) for reference but describe code
  this line no longer carries.
- **Kept:** the Qwen3.5-architecture models (Qwen3.6/3.8 27B, 35B-A3B) and every upstream product
  surface (CLI, server, perplexity, benchmarks, converter). Only Qwen3.8-27B GSQ-RCO IQ3_XXS on the
  RTX 5060 Ti is tuned and measured.
- **Added:**
  - build presets (`rtx5060`, `rtx5060-dev`) and an embedded RTX 5060 Ti device profile, with
    measured kernel routes;
  - the `scripts/qwen38` setup and launch scripts;
  - the committed MTP IQ4_XS conversion override.

## Prefill

| Change | Commit | Measured effect |
|---|---|---|
| Two-pass GGUF prefill: weights decoded into L2-resident MXFP8/NVFP4 slabs, then a cuBLASLt block-scaled GEMM. NVFP4 activations from fused RMSNorm/SwiGLU producers, and a load-time scale table | `7f189ba2` | Part of 865 → ~2,200 tok/s at 160K (with the items below) |
| `sm_120` warp-specialized 128×128 tile kernel for i-quant products (IQ3_XXS, IQ3_S, IQ2_*, IQ4_XS) with fused epilogues | `60cc3f9b` | +17% prefill at 18-30K, +14% at 160K |
| Warp-specialized `rk4v4` prompt attention with e4m3 PV (picked from 5060ninfer) | `0037accd` | Prefill up at every length, most at 225K |
| Automatic prefill chunk, a segmented `staged_fp8` prompt image for 262K contexts, a fused output gate, vectorized conv1d, a two-CTA GDN recurrence | `7f189ba2` | |
| Small-T attention modules loaded at Program startup | `88844810` | First-request time to first token 259-280 → 103-105 ms; +1.2% on a fresh server's 30K prompt |
| First token published before the MTP drafts land | `a2eed564` | Time to first token -3 to -7 ms on returning turns |
| n-gram index built beside the prefill, in huge pages | `57583f6e`, `2242e45d` | Preparing a returning 3.6K turn: 29-36 → 1-2 ms (time to first token -13 to -25 ms) |

## Decode

| Change | Commit | Measured effect |
|---|---|---|
| Narrow GGUF product for 5-32 columns (speculative verification, several requests); 256-bit activation loads, staged decoder tables, compiled epilogues | `7f189ba2` | |
| Four-row vector products routed by a measured per-type table | `3d7f513f` | ≥2% on the routed shapes |
| Coarse IQ2_XXS proposal head with exact rescoring (`shortlist_argmax`) | `7f189ba2` | Rounds 3.8% shorter, identical replies |
| e4m3 PV in the `rk4v4` small-T decode attention (picked from 5060ninfer) | `4b6adb6c` | Decode time per round -0.5% / -1.4% / -5.2% at 18K / 30K / 225K |
| Bit-exact GDN replay, exact-order gating, conversion-free int4 value decode | `85a0b23b` | Shorter rounds with bit-identical outputs |
| `rk4v4` small-T widths 6-8 kept on the 12×1×32 tier past 8K | `4073da41` | A trade-off. When Windows pages GPU memory, a 156K turn is 48 ms faster; in the normal state it is about 3 ms slower |
| GDN state operands rounded before the `sm_120` TF32 MMA (picked from 5060ninfer) | `aa3eea55` | Correctness |
| MTP head requantized Q6_K → IQ4_XS (conversion override) | `fa678571` | -117 MiB; +1.4% decode at 160K |

Short-prompt decode went from 75.5 tok/s at the fork point (64K context, 3 drafts) to 87-88 tok/s
(262K context, 4 drafts). See [Performance](Performance.md) for the conditions.

## Memory: 262,144 tokens on 16 GB

At the fork point, 262,144 tokens failed by 608 MiB. These changes made it fit:

- `--embedding-host`, which keeps the token table in host memory;
- a measured CUDA Graph allowance (32 MiB instead of the default's 13-15 times the need);
- no extra device checkpoint states;
- the 2,304-token prefill chunk.

Pinned host pools for KV and state now prefer 2 MiB huge pages (`33d02523`), which shortens
startup pinning.

## Serving

- **Up to 32 concurrent requests** (`b5f96ebf`): 342 → ~625 tok/s total from 8 to 32 requests.
- **Context cache:**
  - a Host capture lands behind its lane's next prefill pass (`8a217005`);
  - a cancelled prefill keeps its progress for the retry (`b412b7f5`);
  - prefix shortlist digests are mixed a word at a time (`d4f7e869`).
- **WebUI monitor and request history** (`b288fcfc`), styled with
  [Strata](https://github.com/Niko1221/Strata)'s WebUI stylesheet (MIT). The WebUI is shipped in Linux packages and
  Docker images, with `--ui-config-file` (`ee40e32e`).
- **Frontend speed** (`36157551`, `666bb88c`, `a021e478`, `f570d05b`): NFC normalization only
  between boundaries, Jinja `strip` decoding only what it trims, and reuse of ordinary-text runs
  across tokenizer calls.

## Picked from NInfer-all `master` (23 commits)

These upstream changes landed after the fork point and were cherry-picked:

- **Structured output and reasoning:** regex, choice and GBNF constraints, strict tool arguments
  (`0c4c4f5a`); the `--reasoning-loop off|stop|conclude` guard (`d7c2d2ac`, `460cd456`).
- **API conformance:** Chat, Responses and Anthropic requests aligned with the current specs
  (`80120470`); tool-call parsing fixes (`a507d9da`, `60538fea`, `211204d6`, `59371a10`);
  tool-request sampling (`82355e15`).
- **Engine and cache fixes:** `65f45be1`, `d982bfd4`, `e1642b73`, `b608abc4`, `d32c6d1e`,
  `8eb3380d`, `42fe061b`, `29013405`, `bb481e47`.
- **Performance:** `ee87b19c`, `a021e478`, `f570d05b`.
- **Packaging:** `ee40e32e`.

## Picked at the merge of `master` (11 commits)

`master`, mirroring NInfer-all, was 254 commits past this line; upstream NInfer-all had 38 more. Each
was reviewed. `master` was then merged without taking its tree, and the commits that reach this
model and card were picked on top (upstream hashes):

- **Correctness:** CUDA 13 gives every translation unit its own copy of a kernel, and only the
  first copy had its dynamic shared-memory limit raised; the limit is now raised per kernel and
  device (`b471e287`, ported by hand). A TMA pipeline stage is released only after the consumers'
  shared-memory reads (`febfc636`). A synchronous image encode no longer fails while another
  lane's image holds the vision overlay window (`615d16a8`). PPM, PGM and PBM images decode again
  (`53788e2d`).
- **Security:** the WebUI's MCP relay requires the API key and reads bounded bodies (`f71e4f3d`).
  The Monitor page itself stays open; its data routes need the key.
- **GDN gating residency** comes from the device's occupancy instead of one card's constants
  (`ec26e643`, `446dec21`).
- **Converter:** GGUF source values kept in the logical column order, so re-quantising `ssm_out`
  no longer writes permuted columns (`be919c76`); `--device cpu` named when CUDA is unavailable
  (`348f9a10`), one concatenation fewer (`53e909f8`).
- **Packaging:** Linux builds compile FFmpeg 9.0.2 and dav1d 1.5.4 from pinned release archives
  and link them statically (`5d80c03a`), so the server needs no FFmpeg libraries at run time.

The rest serve other model families, other cards, tensor parallelism, pipeline stages or CI. The
[Roadmap](Roadmap.md) lists the ones worth porting later.

The full list of this line's commits is in `git log e7b7e669..master ^33163f05`.

## Tried and rejected

Experiments that lost are recorded so they are not repeated:

- **KV formats:** a 2-bit `rk2v4-e8` cache cost +3.5% perplexity at 64K and +9.9% at 160K.
- **Proposal heads:** a 32K-row head was 2% slower overall, because its acceptance fell.
- **Drafter:** a Q4 DFlash2 drafter needs new drafter kernels.
- **GEMV prototypes:** two tensor-core decode GEMVs were 0.3-0.9 times the vector kernels.
- **Prompt attention:** several redesigns (p61, p78, p80, ping-pong) lost.
- **Module loading:** an uncompressed fatbin and threaded module loading did not speed it up.

The [Roadmap](Roadmap.md) lists what remains open.
