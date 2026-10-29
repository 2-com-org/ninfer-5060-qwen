#!/usr/bin/env bash
# Build ninfer-serve and ninfer-perplexity for this machine's GPU into build-qwen38/.
#
#   scripts/qwen38/build.sh [--arch 120a|89|86] [--build-dir DIR] [--jobs N] [--no-webui]
#
# The CUDA architecture comes from nvidia-smi unless --arch names it: 120a for the RTX 50 series
# (CUDA 13.1 or newer), 89 for the RTX 40 series, 86 for the RTX 30 series (CUDA 12.8 or newer).
# The engine is tuned and measured on the RTX 5060 Ti (120a); other cards build and run with the
# routes measured for them upstream, but this model's settings are unmeasured there.
#
# The WebUI (llama.cpp's released web app, b11165) is downloaded once into third_party/webui and
# compiled into the server; --no-webui builds the API alone. Compilation takes 30-60 minutes on
# 12 cores; each compiler job can take 3 GB of RAM, so the default job count is the smaller of the
# core count and a third of the RAM in GiB.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$root/build-qwen38"
arch='' jobs='' webui=1

fail() { printf 'build.sh: %s\n' "$*" >&2; exit 1; }
while (( $# )); do
  case "$1" in
    --arch) arch=${2:?--arch needs a value}; shift 2 ;;
    --build-dir) build_dir=${2:?--build-dir needs a value}; shift 2 ;;
    --jobs) jobs=${2:?--jobs needs a value}; shift 2 ;;
    --no-webui) webui=0; shift ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) fail "unknown option $1 (see --help)" ;;
  esac
done

# nvcc writes the source path into its command lines, which breaks on a space.
[[ $root != *' '* ]] || fail "the checkout path contains a space: $root. Clone into a path without spaces (for example ~/ninfer-qwen)."

nvcc=${CUDACXX:-$(command -v nvcc || true)}
[[ -n $nvcc ]] || nvcc=/usr/local/cuda/bin/nvcc
[[ -x $nvcc ]] || fail 'nvcc not found: install the CUDA Toolkit (13.1 or newer for the RTX 50 series), or set CUDACXX.'
cuda=$("$nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')

if [[ -z $arch ]]; then
  capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ') ||
    fail 'nvidia-smi cannot see a GPU; pass --arch.'
  case $capability in
    12.0) arch=120a ;;
    8.9) arch=89 ;;
    8.6) arch=86 ;;
    *) fail "no build for compute capability '$capability'; pass --arch 120a, 89 or 86." ;;
  esac
fi
case $arch in
  120a)
    awk -v v="$cuda" 'BEGIN { split(v, p, "."); exit !(p[1] > 13 || (p[1] == 13 && p[2] >= 1)) }' ||
      fail "CUDA $cuda miscompiles sm_120a kernels; the RTX 50 series needs CUDA 13.1 or newer."
    ;;
  89|86) ;;
  *) fail "--arch takes 120a, 89 or 86" ;;
esac

if [[ -z $jobs ]]; then
  memory_gib=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
  jobs=$(( memory_gib / 3 ))
  (( jobs > $(nproc) )) && jobs=$(nproc)
  (( jobs < 1 )) && jobs=1
fi

options=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch"
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF)
# Programmatic dependent launch overlaps a kernel's start with its predecessor's tail (sm_90+).
[[ $arch == 120a ]] && options+=(-DNINFER_PDL=ON)
if (( webui )); then
  webui_root="$root/third_party/webui/b11165"
  index=$(find "$webui_root" \( -name index.html -o -name index.html.gz \) -print 2>/dev/null | head -n 1 || true)
  if [[ -z $index ]]; then
    "$root/scripts/fetch-webui.sh" b11165 "$webui_root"
    index=$(find "$webui_root" \( -name index.html -o -name index.html.gz \) -print | head -n 1)
  fi
  options+=(-DNINFER_WEBUI_DIR="$(dirname -- "$index")")
fi

printf 'Building for sm_%s with CUDA %s, %s jobs, into %s\n' "$arch" "$cuda" "$jobs" "$build_dir"
cmake -S "$root" -B "$build_dir" -G Ninja "${options[@]}"
cmake --build "$build_dir" --parallel "$jobs" --target ninfer-serve ninfer-perplexity
printf 'Built %s\n' "$build_dir/apps/ninfer-serve"
