#!/usr/bin/env bash
# One-shot setup on Ubuntu or WSL2: dependencies, build, model. Then run scripts/qwen38/serve.sh.
#
#   scripts/qwen38/setup.sh [--no-apt]
#
# Runs install-deps.sh (with sudo for apt unless --no-apt), build.sh and prepare-model.sh with their
# defaults; each can be rerun on its own. Needs the CUDA Toolkit (see install-deps.sh), about
# 30 GB of free disk and an hour or two, most of it compilation and the 10.4 GB download.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deps=()
case "${1:-}" in
  --no-apt) deps=(--no-apt) ;;
  '') ;;
  -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
  *) printf 'setup.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
esac
"$here/install-deps.sh" "${deps[@]}"
"$here/build.sh"
"$here/prepare-model.sh"
printf '\nSetup complete. Start the server with:\n  %s/serve.sh\n' "$here"
