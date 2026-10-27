# GGUF block formats

NInfer serves GGUF releases that give every tensor its own ggml quantization type, such as
ISTA-DASLab's [GSQ-RCO models](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
without converting their weights to another format. The converter recipe `qwen3_8_27b_gguf` copies
each quantized tensor's blocks of a Qwen3.8-27B release into the artifact unchanged (see
[weight conversion](weight-conversion.md#a-mixed-precision-qwen38-27b-gguf)), and the runtime
multiplies them in place. The Qwen3.8-Flash-Next releases convert the same way with
`qwen3_8_flash_next_gguf`, their routed experts kept as expert-major banks
([Qwen3.8-Flash-Next](qwen3-8-flash-next.md)).

## Formats

Eighteen block types: the fifteen llama.cpp writes for dense models, `Q8_0`, `Q2_K`, `Q3_K`,
`Q4_K`, `Q5_K`, `Q6_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL` and `IQ4_XS`, and `Q2_0`, `Q4_0` and `Q5_0`, which the Flash-Next releases use, stored as
the `gguf_*` formats of the `gguf_blocks_v1` layout
([tensor formats](maintainer/tensor-formats.md#35-gguf-block-formats),
[storage layouts](maintainer/storage-layouts.md#6-gguf_blocks_v1)). A projection may mix types:
the parts of a fused projection that share a type are multiplied together, the others one by one
against the same activation.

## Products

Every product quantizes its BF16 activation to ggml's `q8_1` numbers, one scale and one sum per 32
values, as llama.cpp does, and accumulates in FP32; the weights' represented values are exactly
those of `ggml-quants.c`. The one exception is the tile kernel's MXFP8 form below.

- **Up to eight columns** (decode, speculative verification, small batches), the vector kernel
  (`src/ops/linear/gguf/ggml_bridge_vec.cuh`) decodes each 32-value slice of a row once into int8
  words and dots it with every column. A warp owns two rows; a `[gate; up]` pair of one type is one
  launch that writes `silu(gate) * up`.
- **More columns** (prompts) take llama.cpp's integer tensor-core kernel (`mul_mat_q`, vendored
  unmodified in `third_party/ggml-quants` under its MIT license) with its Ampere tile table.
  `IQ1_M` has no such kernel; its wide products dequantize to BF16 and run cuBLAS.
- **On sm_120, 5 to 32 columns** (the decode verification of several sequences, their drafts,
  short prompt pieces) of `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S` and `Q2_K` with K a
  multiple of 1024 take the narrow kernel (`src/ops/linear/gguf/ggml_bridge_narrow.cuh`) instead of
  either, from `gguf::narrow_min_columns` (5 for `IQ3_S` and `Q2_K`, 7 for `IQ2_XXS`, 6 for the
  others): the vector kernel's `q8_1` arithmetic on integer tensor cores (`mma.m16n8k32`), each 16
  rows x 32 values of a weight decoded once and multiplied against every column, the product's
  epilogue fused. A CTA owns units of 16 TW rows; each product takes the TW (2, 3 or 4) of least
  modeled time -- the units an SM runs in turn, each costing its weight bytes and a fifth of its
  activation bytes -- and each width and TW has its own measured configuration. On the RTX 5060 Ti,
  over the 27B's products of these types (weighted by their count), it is 17% faster than the
  vector kernel at 8 columns and 38-41%, 32% and 22% faster than the matrix kernel at 9-16, 20-24
  and 28-32. From each type's first width to 7 columns a type is 2-17% faster, while single shapes
  lose up to 5% (`IQ3_S` 6144 x 5120 at 5 columns, `IQ2_S` 5120 x 17408 at 7). The kernel is
  instruction-issue bound: each thread's copies are planned once rather than derived every step,
  and the activation's d is held as FP32 (the same FP16-rounded value), which made it a further 3%
  faster at 8 columns, 7-9% at 9-16 and 13-14% at 17-32 (count-weighted; `IQ3_XXS` 17408 x 5120 at
  17-24 columns 140 -> 110 us), with one shape 5% slower (`IQ3_XXS` 5120 x 6144 at 6-8 columns). With MTP drafts of 3
  tokens, a server decoding 2-8 requests at once produced 14-40% more tokens per second than on
  the vector and matrix kernels, one request the same.
- **On sm_120**, from 33 columns (`gguf::kTileMinColumns`), `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`,
  `IQ4_XS`, `IQ2_XS` and `IQ2_S` take the tile kernel (`src/ops/linear/gguf/ggml_bridge_tile.cuh`):
  the same `q8_1` arithmetic in 128 x 128 tiles, warp-specialized (producer warps copy raw blocks
  and decode them with the vector kernel's decoders, consumer warps run `mma.m16n8k32`), with the
  product's epilogue fused (no FP32 plane). It runs one CTA per SM without stream-k, so
  `gguf::takes_tile_kernel` keeps the matrix kernel where the tile grid fills less than 80%
  (`IQ4_XS` 90%) of its waves; on an RTX 5060 Ti a 1024-column product is 1.06-1.30x faster than
  the matrix kernel's. A narrower product pads its one column tile: the 27B's prompts of 33-64
  tokens prefilled 6-8% faster than on the matrix kernel and 72-127 tokens 24-26%, while at 32
  tokens and fewer the matrix kernel stays faster (4% at 32, 13% at 16).
- **MXFP8.** On a 120a build, a product whose Use permits A8 activations (every GGUF import stores
  at least that permission) runs the tile kernel's MXFP8 form instead (unless its A4 permission
  selects NVFP4, below), unless `--no-prefill-mxfp8` is given.
  The activation becomes e4m3 under one power-of-two scale per 32 values; the producer warps write
  each decoded 32-value weight slice as e4m3 under its own power-of-two scale (the minimum terms of
  `Q2_K` and `Q4_K` folded in, so those two take the tile kernel too); the consumers run the
  block-scaled e4m3 `mma` (`kind::mxf8f6f4`), which applies both scales and accumulates in FP32.
  The i-quants and `Q2_K` decode through byte-permute lookups: each slice rounds its few code
  values (grid magnitudes, `IQ4_XS`'s 16 levels, `Q2_K`'s 4) times its scale to e4m3 once, so a
  weight costs a lookup instead of a conversion. On the RTX 5060 Ti the form runs 1.26-1.41x the
  `q8_1` tile rate for the one-scale i-quants, about 2x for `IQ2_XS` and `IQ2_S`, and 1.6-2.9x the
  matrix kernel for `Q4_K` and `Q2_K`, at 3.4-3.8% relative RMS error per product against
  `q8_1`'s 0.9%; `Q6_K` stays on `q8_1`. What that costs and buys a whole model is in
  [quality trades](maintainer/quality-trade-experiments.md).
- **Two-pass MXFP8.** From 1024 columns (`gguf::lt_min_columns`) the MXFP8 form's BF16 products
  (projections, `linear_add`, and gated `[gate; up]` pairs) run in two passes per slab of rows
  instead: the slab's blocks decode with the tile kernel's arithmetic (the same e4m3 values and
  scales) into an MXFP8 image of up to 10.5 MiB in scratch, small enough to stay in L2, and
  cuBLASLt's block-scaled GEMM (`VEC32_UE8M0`) multiplies it with FP32 accumulation into BF16. A
  gated pair stacks its gate and up rows in one image and one GEMM and forms `act(gate) * up` from
  the BF16 products, so it needs no FP32 gate plane. cuBLASLt offers no FP32 output for this GEMM,
  so products into FP32 planes keep the tile kernel. A product's passes take the rows (up to the
  slab) that run its GEMMs in the fewest waves of 128 x 128 tiles on the device: at 1152 columns on
  36 SMs, K = 6144's 5120 rows take 1536-row passes (3 + 3 + 3 + 1 waves, 5.5% faster than 1792-row
  ones at 4 + 4 + 3). On the RTX 5060 Ti the 27B's products ran 4-20% faster than the tile kernel
  at 1024 columns and 16-20% at 2048, but some ran slower at 256-768 columns (the image pass costs
  the same while a narrow GEMM fills the device poorly). A 30K-token prompt prefilled 15% faster
  and a 160K one 11%; the image pass is ~9% of a 1152-token chunk, which a 2304-token chunk halves
  per token.
- **Two-pass NVFP4** (the default; `--no-prefill-nvfp4` turns it off). The Qwen3.8-27B recipe
  stores A4 permission for every imported matrix but the vocabulary head and the token table;
  those Uses' two-pass products run in NVFP4 instead of MXFP8: e2m1 values under an E4M3 scale per 16 values and one
  power-of-two global per operand (`max|x| * g <= 2688`), multiplied by cuBLASLt's NVFP4 GEMM
  (`VEC16_UE4M3`, about 340 TOPS on the RTX 5060 Ti against MXFP8's 170). Each 16 values take
  the scale with the least squared error among four candidates (`RN(max/6)`, `RN(max/4)` and the
  next E4M3 code above each). Which one wins depends only on the weight and its global, so the
  model's native parameters find every composed operand's choices once at startup (two bits per
  16 values in pinned host memory, `pinned_fp4_choice_bytes` in the load summary, about 0.4 GB
  for the 27B recipe) and each chunk's image pass reads them instead of searching, byte for
  byte the same image; a product whose global is not the one its table was built at searches.
  The image decodes with the vector kernel's decoders, so the form
  also serves `IQ1_S`, `IQ1_M` and `Q6_K` (whose prompts otherwise take `mul_mat_q` or BF16); the
  weight's global comes from its max |w|, which the loader measures once per matrix
  (`ops::gguf_max_abs`). An activation is encoded in one pass, each 16 values at their own power
  of two, which also finds its max |x|; a scale pass then moves each scale code by whole octaves to
  the activation's global (the arithmetic is invariant under that move while every candidate scale
  stays a normal, unsaturated E4M3 code) and encodes the few groups outside that range again from
  the input. In the dense FFN (`ops::linear_swiglu_add`) the kernel that forms `act(gate) * up`
  encodes those products as the down projection's activation, so down skips its own encode pass;
  the result is byte-identical to `linear_swiglu` followed by `linear_add`. With
  `--no-prefill-nvfp4` an A4 Use takes the MXFP8 form, and with `--no-prefill-mxfp8` `q8_1`. At
  its GEMM's rate the image pass pays for itself far sooner than MXFP8's, so the form takes
  products from 129 columns (`gguf::lt_min_columns`): on the RTX 5060 Ti the 27B's prompts of
  129-384 tokens prefilled 6-11% faster than on the matrix and tile kernels, 512 tokens 32%, 768
  57% and 1000 74%, and a server's first answer to a 518- or 918-token prompt came 30-34% sooner;
  at 128 tokens and fewer those kernels stay faster (the tile kernel by 42% at a whole 128-column
  tile). Its cost and gain are in [quality trades](maintainer/quality-trade-experiments.md).
- **Token table** rows are dequantized exactly.

The scales of `IQ2_XXS`, `IQ2_XS`, `IQ2_S` and `IQ3_XXS` are applied in FP32 where llama.cpp's
vector kernels round an integer rescale, so a decode step's logits can differ from llama.cpp's in the
last bits; prompts use llama.cpp's own arithmetic.

## Serving

A converted artifact starts like any other; MTP, DFlash2 and Vision work as with the official
artifact. The GSQ-RCO IQ3_S conversion is published as
[WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3):

```bash
./build/apps/ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --kv-dtype rk8v4 --gdn-state-fp16 --spec mtp --draft-tokens 3
```

## Measurements

GSQ-RCO IQ3_S (3.5 bits per weight, 10.95 GiB of weights) against the official Qwen3.8-27B NInfer
artifact (Q4/Q5 groups, 15.9 GiB of weights).

WikiText-2 test perplexity with the protocol of the GSQ-RCO model card (the test rows joined by
blank lines, disjoint 2,048-token windows, `ninfer-perplexity --disjoint`, BF16 KV):

| Artifact | Perplexity |
|---|---:|
| GSQ-RCO IQ3_S in NInfer | 7.071 |
| GSQ-RCO IQ3_S, model card | 7.07 |
| BF16 base model, model card | 7.05 |
| Official Qwen3.8-27B NInfer artifact | 7.286 |

The reasoning benchmarks of the [capability evaluation](../eval/README.md), with the protocol the
official artifact was scored with (EvalScope, temperature 1.0, one sampled run each, an RTX 5090):

| Benchmark | GSQ-RCO IQ3_S in NInfer | Official Qwen3.8-27B NInfer artifact |
|---|---:|---:|
| IFBench (prompt-level strict) | 80.33% | 77.67% |
| AIME 2025 | 100.00% | 96.67% |
| AIME 2026 | 100.00% | 96.67% |
| GPQA-Diamond | 88.38% | 87.37% |

The GSQ-RCO card reports 100 on AIME 2025 and 89.39 on GPQA-Diamond with its own protocol (BF16:
100 and 89.90). One run of GPQA-Diamond's 198 questions varies by about two points.

Speed against the official artifact on the same card in the same sitting, `rk8v4` KV, one request,
greedy, thinking off, with the reference client (`tools/bench/refbench.py`): an RTX 3090 at 420 W
and an RTX 4090 at 450 W with a 176,128-token window (the official DFlash2 row on the RTX 4090 at
167,936), an RTX 5090 at 450 W with 262,144. Short chat is the mean decode rate over five
512-token answers; GSQ-RCO IQ3_S / official:

| | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---:|---:|---:|
| short chat, no speculation, tok/s | 59.9 / 40.3 | 70.6 / 55.0 | 107.5 / 88.1 |
| short chat, MTP, 3 drafts, tok/s | 108.7 / 82.7 | 146.4 / 109.3 | 221.4 / 178.1 |
| short chat, DFlash2, 5 drafts, tok/s | 115.6 / 109.2 | 169.4 / 141.3 | 226.5 / 222.5 |
| decode after 32K tokens, no speculation, tok/s | 50.4 / 38.4 | 66.0 / 52.2 | 100.3 / 83.2 |
| time to first token, 32K prompt, s | 20.5 / 22.2 | 9.3 / 12.8 | 8.6 / 9.3 |

The lead comes from reading 31% fewer weight bytes per token. Verification of several drafts gains
less than plain decode because the i-quants cost more arithmetic per byte than the official Q4/Q5
groups, and every extra column adds its own dot products; DFlash2's six-token verification is
where the two meet.

