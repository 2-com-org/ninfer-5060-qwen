#!/usr/bin/env bash
# Install the build dependencies on Ubuntu (24.04 or newer, also under WSL2), and with --converter
# the converter's Python environment in .venv-convert/.
#
#   scripts/qwen38/install-deps.sh [--no-apt] [--converter] [--venv DIR]
#
# Not installed here: the NVIDIA driver and the CUDA Toolkit. Install the toolkit from
# https://developer.nvidia.com/cuda-downloads -- CUDA 13.1 or newer for the RTX 50 series, 12.8 or
# newer for the RTX 30/40 series. Under WSL2 choose the "WSL-Ubuntu" package and install no Linux
# driver: WSL uses the Windows driver (580 or newer).
#
# --converter creates the environment prepare-model.sh converts the model with: Python 3.11
# (python3.11, or uv, which fetches it; otherwise python3) with the CPU build of PyTorch, about
# 1 GB. download-model.sh, which setup.sh runs by default, needs none of it. --no-apt skips the
# system packages (no sudo).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
venv="$root/.venv-convert"
apt=1 converter=0
while (( $# )); do
  case "$1" in
    --no-apt) apt=0; shift ;;
    --converter) converter=1; shift ;;
    --venv) venv=${2:?--venv needs a value}; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'install-deps.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done

if (( apt )); then
  sudo apt-get update
  # meson, nasm and zlib build the FFmpeg and dav1d that the server links statically.
  sudo apt-get install --yes build-essential cmake ninja-build pkg-config git curl \
    meson nasm zlib1g-dev libcurl4-openssl-dev python3-venv
fi

if (( converter )); then
  if [[ ! -x $venv/bin/python ]]; then
    if command -v python3.11 >/dev/null; then
      python3.11 -m venv "$venv"
    elif command -v uv >/dev/null; then
      uv venv --seed --python 3.11 "$venv"
    else
      printf 'python3.11 and uv not found; using %s (the converter is maintained on 3.11)\n' "$(python3 --version)"
      python3 -m venv "$venv"
    fi
  fi
  "$venv/bin/python" -m pip install --upgrade pip
  "$venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
  "$venv/bin/python" -m pip install -r "$root/scripts/qwen38/requirements-convert.txt"
  printf 'Converter environment: %s\n' "$venv"
fi

nvcc=$(command -v nvcc || true)
[[ -n $nvcc ]] || nvcc=/usr/local/cuda/bin/nvcc
if [[ -x $nvcc ]]; then
  "$nvcc" --version | tail -n 2
else
  printf '\nThe CUDA Toolkit is missing: install it before scripts/qwen38/build.sh (see the top of this script).\n'
fi
