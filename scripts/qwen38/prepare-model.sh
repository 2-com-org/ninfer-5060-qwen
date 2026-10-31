#!/usr/bin/env bash
# Download the model sources and convert them into the artifact this engine serves.
#
#   scripts/qwen38/prepare-model.sh [--models-dir DIR] [--python PY] [--skip-download] [--force]
#
# Downloads about 10.5 GB into models/ (or --models-dir), from pinned revisions:
#   ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF  Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf, 10.4 GB: the GSQ-RCO
#                                         weights with the multi-token-prediction head
#   Qwen/Qwen3.8-27B                      config, tokenizer and processor files only (no weights)
# Both are Apache-2.0. Then converts them on the CPU (about 10 minutes, 11 GB of RAM) into
# models/qwen38-27b-gsq-iq3xxs.ninfer, 10.8 GB: the GGUF blocks as stored, the MTP projections
# requantized from Q6_K to IQ4_XS (tools/convert/overrides/qwen3_8_mtp_iq4xs.py) and a coarse
# IQ2_XXS proposal head for the shortlisted LM head. Needs about 22 GB free disk for the sources and
# the artifact; the sources can be deleted afterwards.
#
# --skip-download uses sources already in the models directory (checked by size and sha256).
# --python defaults to .venv-convert/bin/python (scripts/qwen38/install-deps.sh --converter creates it).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
models="$root/models"
python="$root/.venv-convert/bin/python"
download=1 force=0
while (( $# )); do
  case "$1" in
    --models-dir) models=${2:?--models-dir needs a value}; shift 2 ;;
    --python) python=${2:?--python needs a value}; shift 2 ;;
    --skip-download) download=0; shift ;;
    --force) force=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'prepare-model.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done
fail() { printf 'prepare-model.sh: %s\n' "$*" >&2; exit 1; }
[[ -x $python ]] || fail "no converter Python at $python: run scripts/qwen38/install-deps.sh --converter, or pass --python"

gguf_repo=ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
gguf_revision=d562806dbafae37109975e970aae91b43e73b440
gguf_name=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
gguf_size=10442827840
gguf_sha256=63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262
base_repo=Qwen/Qwen3.8-27B
base_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
base_files=(chat_template.jinja config.json generation_config.json merges.txt preprocessor_config.json
  tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json)

gguf="$models/$gguf_name"
base="$models/Qwen3.8-27B"
out="$models/qwen38-27b-gsq-iq3xxs.ninfer"
if [[ -f $out && $force == 0 ]]; then
  printf '%s exists; pass --force to convert again.\n' "$out"
  exit 0
fi
mkdir -p "$models"

if (( download )); then
  "$python" - "$models" "$gguf_repo" "$gguf_revision" "$gguf_name" "$base_repo" "$base_revision" \
    "${base_files[@]}" <<'EOF'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

models, gguf_repo, gguf_revision, gguf_name, base_repo, base_revision, *base_files = sys.argv[1:]
for name in base_files:
    hf_hub_download(base_repo, name, revision=base_revision, local_dir=Path(models) / "Qwen3.8-27B")
hf_hub_download(gguf_repo, gguf_name, revision=gguf_revision, local_dir=models)
EOF
fi

for name in "${base_files[@]}"; do
  [[ -f $base/$name ]] || fail "missing $base/$name"
done
[[ -f $gguf ]] || fail "missing $gguf"
[[ $(stat -L -c %s "$gguf") == "$gguf_size" ]] || fail "$gguf is not the pinned file (size differs)"
stamp="$gguf.sha256-verified"
if [[ ! -f $stamp || $stamp -ot $gguf ]]; then
  printf 'Checking %s\n' "$gguf"
  [[ $(sha256sum "$gguf" | cut -d ' ' -f 1) == "$gguf_sha256" ]] || fail "$gguf does not match sha256 $gguf_sha256"
  touch "$stamp"
fi

printf 'Converting into %s\n' "$out"
cd "$root"
"$python" -m tools.convert --recipe qwen3_8_27b_gguf \
  --override tools/convert/overrides/qwen3_8_mtp_iq4xs.py \
  --model "$base" --source "gguf=$gguf" --components text,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --proposal-coarse gguf_iq2_xxs \
  --ranking tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 \
  --name qwen3.8-27b --device cpu --out "$out"
printf 'Model ready: %s\nServe it with scripts/qwen38/serve.sh\n' "$out"
