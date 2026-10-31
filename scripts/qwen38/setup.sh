#!/usr/bin/env bash
# One-shot setup on Ubuntu or WSL2: dependencies, build, model. Then run scripts/qwen38/serve.sh.
#
#   scripts/qwen38/setup.sh [--no-apt] [--convert]
#
# Runs install-deps.sh (with sudo for apt unless --no-apt), build.sh and download-model.sh, which
# fetches the converted model (10.8 GB) from Hugging Face; each can be rerun on its own. --convert
# makes the model from the original GGUF instead (prepare-model.sh): it installs the converter's
# Python environment (about 1 GB), downloads 10.4 GB of sources and converts them in about 10
# minutes on the CPU, into an artifact with the same weights. Needs the CUDA Toolkit (see
# install-deps.sh), about 15 GB of free disk (35 GB with --convert) and about an hour, most of it
# compilation.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deps=() convert=0
while (( $# )); do
  case "$1" in
    --no-apt) deps+=(--no-apt); shift ;;
    --convert) convert=1; deps+=(--converter); shift ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'setup.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done
"$here/install-deps.sh" "${deps[@]}"
"$here/build.sh"
if (( convert )); then "$here/prepare-model.sh"; else "$here/download-model.sh"; fi
printf '\nSetup complete. Start the server with:\n  %s/serve.sh\n' "$here"
