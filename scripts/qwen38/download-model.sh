#!/usr/bin/env bash
# Download the converted Qwen3.8-27B GSQ-RCO IQ3_XXS artifact that serve.sh serves.
#
#   download-model.sh [--models-dir DIR]
#
# Fetches qwen38-27b-gsq-iq3xxs.ninfer (10.8 GB) and its conversion record from Hugging Face
# (FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer, Apache-2.0) at a pinned revision into models/ beside
# serve.sh in a release archive, or in the checkout (--models-dir overrides). An interrupted download
# resumes where it stopped; the finished file is checked against its size and SHA-256 before it is
# moved into place. Needs curl and about 11 GB of free disk. In a checkout,
# scripts/qwen38/prepare-model.sh converts the same weights from the original GGUF instead.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f $here/serve.sh && -d $here/bin ]]; then root=$here; else root="$(cd -- "$here/../.." && pwd)"; fi
models="$root/models"
while (( $# )); do
  case "$1" in
    --models-dir) models=${2:?--models-dir needs a value}; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'download-model.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done
fail() { printf 'download-model.sh: %s\n' "$*" >&2; exit 1; }
command -v curl >/dev/null || fail 'curl is missing (sudo apt-get install curl)'

repo=FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer
revision=73d153fab03ed94c5c76b329897a7450f7de601d
files=(
  'qwen38-27b-gsq-iq3xxs.ninfer 10846918144 e715439d19d0e156f43a128d724d50825ccd55a2b2c0f898d37716a7d269f819'
  'qwen38-27b-gsq-iq3xxs.ninfer.conversion.json 323911 be9aa3996f0bf1c46a4799cd11848feeabf33567e9d4b117a34665e95a81eba8'
)

mkdir -p -- "$models"
for entry in "${files[@]}"; do
  read -r name size sha256 <<< "$entry"
  target="$models/$name" part="$models/$name.part"
  if [[ -f $target && $(stat -c %s -- "$target") == "$size" ]]; then
    printf 'Checking %s\n' "$target"
    [[ $(sha256sum -- "$target" | cut -d ' ' -f 1) == "$sha256" ]] && { printf 'Present: %s\n' "$target"; continue; }
    fail "$target has the right size but the wrong SHA-256; delete it and run this again"
  fi
  have=0
  [[ -f $part ]] && have=$(stat -c %s -- "$part")
  (( have <= size )) || { rm -f -- "$part"; have=0; }
  free=$(( $(df --output=avail -B1 -- "$models" | tail -n 1) ))
  (( free > size - have + 268435456 )) ||
    fail "$models has $(( free >> 20 )) MiB free; $name needs $(( (size - have) >> 20 )) MiB more (pass --models-dir)"
  printf 'Downloading %s (%s bytes) into %s\n' "$name" "$size" "$models"
  # Each attempt resumes from what the previous ones wrote.
  for attempt in 1 2 3 4 5 6 7 8; do
    (( have < size )) || break
    curl --fail --location --continue-at - --output "$part" \
      "https://huggingface.co/$repo/resolve/$revision/$name" ||
      { printf 'Attempt %s stopped (curl exit %s); resuming\n' "$attempt" "$?"; sleep 5; }
    have=$(stat -c %s -- "$part" 2>/dev/null || echo 0)
  done
  (( have == size )) || fail "$part is $have bytes, expected $size; run this again to resume"
  printf 'Checking %s\n' "$part"
  if [[ $(sha256sum -- "$part" | cut -d ' ' -f 1) != "$sha256" ]]; then
    rm -f -- "$part"
    fail "$name failed its SHA-256 check and was deleted; run this again"
  fi
  mv -- "$part" "$target"
  printf 'Downloaded: %s\n' "$target"
done
