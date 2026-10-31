#!/usr/bin/env bash
# Build the Linux release archive in a container: ninfer-serve for the RTX 50 series (sm_120a),
# packaged by scripts/package-release.sh into dist/.
#
#   scripts/qwen38/build-release.sh [--jobs N] [--work DIR]
#
# The build uses CUDA 13.1.1, the toolkit of the published measurements, on Ubuntu 24.04, whose
# glibc 2.39 is the oldest the archive runs on. It compiles the committed tree (HEAD, without local
# changes) in DIR/src, default build-release/ in this checkout, and keeps DIR/build between runs, so
# a second run rebuilds only what changed. Needs Docker and about 25 GB in DIR; compilation takes
# 30-60 minutes on 12 cores, and each compiler job can take 3 GB of RAM.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$root/build-release" jobs=''
while (( $# )); do
  case "$1" in
    --jobs) jobs=${2:?--jobs needs a value}; shift 2 ;;
    --work) work=${2:?--work needs a value}; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'build-release.sh: unknown option %s\n' "$1" >&2; exit 2 ;;
  esac
done
fail() { printf 'build-release.sh: %s\n' "$*" >&2; exit 1; }
command -v docker >/dev/null || fail 'docker is missing'
[[ -z $(git -C "$root" status --porcelain --untracked-files=no) ]] ||
  printf 'build-release.sh: building HEAD; local changes are not included\n' >&2

base=nvidia/cuda:13.1.1-devel-ubuntu24.04@sha256:9cf8694a27722418a1f175d90f85d5afb5a728fd4a9907d7f0565efecfa14d32
image=ninfer-5060-qwen-release:13.1.1-ubuntu24.04
docker build --quiet --tag "$image" - <<EOF
FROM $base
RUN apt-get update \\
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \\
        ca-certificates cmake curl file git libcurl4-openssl-dev make meson nasm ninja-build \\
        patchelf pkg-config python3 zlib1g-dev \\
    && rm -rf /var/lib/apt/lists/*
EOF

# A checksum copy keeps the mtimes of unchanged files, so the kept build tree stays incremental.
mkdir -p -- "$work/src" "$work/build" "$work/export"
rm -rf -- "$work/export"/*
git -C "$root" archive HEAD | tar -x -C "$work/export"
# Downloads the checkout already holds (the WebUI bundle and the media archives, both checked
# against pinned digests when fetched) are reused instead of fetched again.
for kept in third_party/webui third_party/media-sources; do
  [[ -d $root/$kept ]] && cp -a -- "$root/$kept" "$work/export/$(dirname -- "$kept")/"
done
rsync --recursive --links --checksum --delete -- "$work/export/" "$work/src/"
build_id="$(git -C "$root" describe --always --tags)"

docker run --rm --user "$(id -u):$(id -g)" --env HOME=/tmp --env NINFER_BUILD_ID="$build_id" \
  --volume "$work:/work" "$image" bash -c '
    set -euo pipefail
    /work/src/scripts/qwen38/build.sh --arch 120a --build-dir /work/build ${1:+--jobs "$1"}
    NINFER_BUILD_ROOT=/work/build /work/src/scripts/package-release.sh' release "$jobs"

mkdir -p -- "$root/dist"
tag="$(tr -d '[:space:]' < "$work/src/VERSION")"
mv -f -- "$work/src/dist/ninfer-5060-qwen-$tag-linux-x64.tar.gz" \
  "$work/src/dist/ninfer-5060-qwen-$tag-linux-x64.tar.gz.sha256" "$root/dist/"
printf 'Release archive: %s\n' "$root/dist/ninfer-5060-qwen-$tag-linux-x64.tar.gz"
