# Model preparation

NInfer serves its own `.ninfer` artifacts. `scripts/qwen38/prepare-model.sh` builds the one this
line is measured with from public sources.

## Sources

| Source | Revision | What is used | License |
|---|---|---|---|
| [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | `d562806d` | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`, 10,442,827,840 bytes, sha256 `63f29a21…4093262`: the weights with the 15-tensor MTP head | Apache-2.0 |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2` | the config, tokenizer, chat template and processor files only, about 23 MB (no weights) | Apache-2.0 |

GSQ-RCO is ISTA-DASLab's mixed-precision quantization
([arXiv:2604.18556](https://arxiv.org/abs/2604.18556), [arXiv:2605.00649](https://arxiv.org/abs/2605.00649)).
Each tensor keeps the GGUF block type it was given. The IQ3_XXS file holds 866 tensors:

| Types | Tensors |
|---|---|
| IQ3_S and IQ3_XXS | 180 |
| IQ2_S, IQ2_XS and IQ2_XXS | 134 |
| IQ4_XS | 33 |
| Q2_K | 28 |
| Q4_K | 19 |
| Q6_K (the MTP projections) | 8 |
| IQ1_S and IQ1_M | 8 |
| BF16 and F32 (norms and small tensors) | 456 |

## Steps

```bash
scripts/qwen38/prepare-model.sh
```

The script runs four steps:

1. **Download.** It downloads both sources at the pinned revisions into `models/`, or into
   `--models-dir`.
2. **Check.** It checks the GGUF's size and sha256. A stamp file records a passed check, so later
   runs skip it.
3. **Convert.** It converts on the CPU in about 10 minutes (516 s on 12 cores, 11 GB peak RAM) into
   `models/qwen38-27b-gsq-iq3xxs.ninfer`, 10.8 GB.
4. **Record.** It writes `qwen38-27b-gsq-iq3xxs.ninfer.conversion.json` next to the artifact: the
   provenance, and every object with its source tensors, format and method.

With `--skip-download` it uses sources already in the models directory, as long as they pass the
same checks. The sources can be deleted afterwards; the server reads only the `.ninfer` file.

## What the conversion does

```bash
python -m tools.convert --recipe qwen3_8_27b_gguf \
  --override tools/convert/overrides/qwen3_8_mtp_iq4xs.py \
  --model models/Qwen3.8-27B --source gguf=models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  --components text,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --proposal-coarse gguf_iq2_xxs \
  --ranking tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 \
  --name qwen3.8-27b --device cpu --out models/qwen38-27b-gsq-iq3xxs.ninfer
```

- **`qwen3_8_27b_gguf`** imports every text tensor with its ggml blocks kept byte for byte. The
  engine runs those blocks natively, through llama.cpp's MMQ formats and NInfer's own vector and
  tile kernels. The loader never repacks weights at runtime.
- **The MTP override** requantizes the MTP head's projections from Q6_K (6.56 bits per weight) to
  IQ4_XS (4.25). The head is read once per draft, four times per round. On this card the change
  saved 117 MiB and made decode 1.4% faster after a 160K-token prompt: rounds were 2.4% shorter,
  and acceptance fell from 0.592 to 0.582. Its blocks decode exactly as gguf-py decodes IQ4_XS
  ([test](../tests/convert/test_qwen3_8_mtp_iq4xs.py)).
- **The proposal head** (`--proposal`) is a shortlist of the 131,072 most frequent tokens, ranked by
  the repository's frequency corpus. The speculative drafts use it instead of the full
  248,320-token vocabulary.
  - `--proposal-coarse gguf_iq2_xxs` adds an IQ2_XXS copy of that shortlist, which picks
    candidates before exact rescoring.
  - Together they made a round 1.23 ms (3.8%) faster. All 24 test replies stayed identical, with
    the same drafts accepted.
  - About 0.56% of reply tokens fall outside the shortlist. A draft cannot propose those tokens,
    so the decode cost is at most about 0.5%.
- **The chat template** is the repository's Qwen3.8 template. It adds tool-call and thinking
  handling that the serving layer relies on.

## Optional: DFlash2 drafter

NInfer can also draft with z-lab's DFlash2 companion model. Against MTP on this card it was 12%
faster on short prompts and 4% faster at 30K, about even at 64K, and 1-2% slower at 90-120K. Its weights need 2 GiB more of GPU
memory, so it serves one request at up to 131,072 tokens.

This line does not script or support it. The artifact would have to be converted without the
coarse proposal head (`--source dflash2=<Qwen3.8-27B-DFlash2> --components text,mtp,dflash2`,
without `--proposal-coarse`), because the loader refuses DFlash2 together with the coarse head.

## Prebuilt artifact

A prebuilt `.ninfer` is not on Hugging Face yet. Both sources are Apache-2.0, so publishing one is
allowed with attribution; it is on the [Roadmap](Roadmap.md).
