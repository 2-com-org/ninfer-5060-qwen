#!/usr/bin/env bash
# Package the Linux release archive: ninfer-serve for the RTX 50 series (sm_120a) with the CUDA
# libraries it links, the launch scripts and the run documentation. No tests, benchmarks or
# evaluation tools.
#
#   NINFER_BUILD_ROOT=<build dir> scripts/package-release.sh
#   scripts/qwen38/build-release.sh        builds in the release container, then runs this
#
# VERSION holds the release tag (for example 0.1.0), which names the archive and
# RELEASE_NOTES_<tag>.md; to cut a release, bump VERSION and write its notes. The build tree must be
# a Release build for 120a with the WebUI (scripts/qwen38/build.sh). The binary is linked on the
# build host, so its glibc sets the oldest system the archive runs on: build-release.sh uses
# Ubuntu 24.04. Needs patchelf, and python3 to read the media manifest.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fail() { printf 'package-release.sh: %s\n' "$*" >&2; exit 1; }

[[ -f $repo_root/VERSION ]] || fail "missing $repo_root/VERSION"
tag="$(tr -d '[:space:]' < "$repo_root/VERSION")"
[[ -n $tag ]] || fail "$repo_root/VERSION is empty"
notes="RELEASE_NOTES_$tag.md"
# Checked before anything is deleted or built into dist.
[[ -f $repo_root/$notes ]] || fail "missing $repo_root/$notes: write the release notes for $tag first"
command -v patchelf >/dev/null || fail 'patchelf is missing'

build_root="$(cd -- "${NINFER_BUILD_ROOT:?set NINFER_BUILD_ROOT to the build tree}" && pwd)"
cache="$build_root/CMakeCache.txt"
[[ -f $cache ]] || fail "$build_root is not a configured build tree"
setting() { sed -n "s/^$1:[A-Z]*=//p" "$cache"; }
[[ $(setting CMAKE_BUILD_TYPE) == Release ]] || fail 'the build tree is not a Release build'
[[ $(setting CMAKE_CUDA_ARCHITECTURES) == 120a ]] || fail 'the build tree does not target 120a'
[[ -n $(setting NINFER_WEBUI_DIR) ]] || fail 'the build tree has no WebUI (NINFER_WEBUI_DIR)'
server="$build_root/apps/ninfer-serve"
[[ -x $server ]] || fail "missing $server"
manifest="$build_root/third_party/media/sources.json"
[[ -f $manifest ]] || fail "missing $manifest: the build did not link the static FFmpeg"
cuda_root="$(cd -- "$(dirname -- "$(setting CMAKE_CUDA_COMPILER)")/.." && pwd -P)"

dist="$repo_root/dist"
name="ninfer-5060-qwen-$tag-linux-x64"
product="$dist/$name"
archive="$dist/$name.tar.gz"
checksum="$dist/$name.tar.gz.sha256"
mkdir -p -- "$dist"
# The checksum goes first and is written last, so a failed run leaves no stale pair to publish.
rm -rf -- "$product" "$archive" "$checksum"
mkdir -p -- "$product"/{bin,lib,licenses,sources,windows}

# The host symbol table is not needed at run time; the fatbins and the build-id note stay. DT_RPATH
# (not RUNPATH) is searched before LD_LIBRARY_PATH, so another CUDA Toolkit there cannot replace
# the bundled libraries.
install -m 0755 -- "$server" "$product/bin/ninfer-serve"
strip --strip-unneeded -- "$product/bin/ninfer-serve"
patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$product/bin/ninfer-serve"

# Every direct dependency is either a CUDA library bundled here, the driver's libcuda, or a library
# the archive README lists as a requirement. Anything else would make that list wrong.
system_libraries=' libcuda.so.1 libcurl.so.4 libz.so.1 libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 '
resolved="$(ldd "$server")"
while read -r needed; do
  case $needed in
    libcudart.so.*|libcublas.so.*|libcublasLt.so.*)
      path="$(awk -v n="$needed" '$1 == n && $2 == "=>" { print $3 }' <<< "$resolved")"
      [[ -n $path && $(readlink -f -- "$path") == "$cuda_root"/* ]] ||
        fail "$needed does not resolve into the CUDA Toolkit at $cuda_root"
      install -m 0644 -- "$(readlink -f -- "$path")" "$product/lib/$needed" ;;
    *) [[ $system_libraries == *" $needed "* ]] || fail "unexpected dependency $needed" ;;
  esac
done < <(readelf --dynamic --wide "$server" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
# libcublas loads libcublasLt; both must come from the same Toolkit.
[[ -f $product/lib/libcublas.so.13 && -f $product/lib/libcublasLt.so.13 && -f $product/lib/libcudart.so.13 ]] ||
  fail 'expected libcudart, libcublas and libcublasLt 13 among the dependencies'
if grep -E -q 'lib(avcodec|avformat|avutil|swscale|swresample|dav1d)\.so' <<< "$resolved"; then
  fail 'ninfer-serve links FFmpeg or dav1d dynamically'
fi

install -m 0755 -- "$repo_root/scripts/qwen38/serve.sh" "$repo_root/scripts/qwen38/download-model.sh" "$product/"
install -m 0644 -- "$repo_root/scripts/qwen38/windows/Start-Qwen38.cmd" \
  "$repo_root/scripts/qwen38/windows/Start-Qwen38.ps1" "$product/windows/"
install -m 0755 -- "$repo_root/scripts/qwen38/windows/wsl-supervisor.sh" "$product/windows/"
install -m 0644 -- "$repo_root/docs/release-archive-linux.md" "$product/README.md"
install -m 0644 -- "$repo_root/VERSION" "$repo_root/LICENSE" "$repo_root/$notes" "$product/"

# Licences of what the binary carries. FFmpeg (LGPL 2.1) and dav1d are linked statically, so their
# exact source archives ship beside them.
install -D -m 0644 -- "$cuda_root/EULA.txt" "$product/licenses/NVIDIA-CUDA-EULA.txt"
for component in cpp-httplib:LICENSE ggml-quants:LICENSE llama-jinja:LICENSE llama-jinja:UNICODE-LICENSE \
                 nlohmann:LICENSE.MIT spdlog:LICENSE spdlog:include/spdlog/fmt/bundled/fmt.license.rst \
                 utf8proc:LICENSE.md xgrammar:LICENSE xgrammar:NOTICE xgrammar:3rdparty/dlpack/LICENSE; do
  directory=${component%%:*} file=${component#*:}
  install -D -m 0644 -- "$repo_root/third_party/$directory/$file" \
    "$product/licenses/$directory/$(basename -- "$file")"
done
# The WebUI is llama.cpp's, under the same MIT licence as its other parts above.
install -D -m 0644 -- "$repo_root/third_party/llama-jinja/LICENSE" "$product/licenses/llama.cpp-webui/LICENSE"
while IFS=$'\t' read -r component archive_path licence; do
  install -m 0644 -- "$archive_path" "$product/sources/"
  tar -xJOf "$archive_path" "$licence" > "$product/licenses/$component-$(basename -- "$licence")"
done < <(python3 -c '
import json, sys
for name, entry in json.load(open(sys.argv[1])).items():
    print(name, entry["archive"], entry["license"], sep="\t")
' "$manifest")

(
  cd -- "$product"
  find . -type f ! -name SHA256SUMS.txt -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum -- > SHA256SUMS.txt
)
tar -C "$dist" --owner=0 --group=0 --numeric-owner -czf "$archive" "$name"
(cd -- "$dist" && sha256sum -- "$name.tar.gz" > "$(basename -- "$checksum")")
du -h -- "$archive"
cat -- "$checksum"
