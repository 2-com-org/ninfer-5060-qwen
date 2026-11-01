# Cutting a release

This line publishes one archive: Linux x86-64 for the RTX 50 series (`sm_120a`), which also runs
under WSL2. It holds `ninfer-serve` with the WebUI built in, NVIDIA's CUDA runtime and cuBLAS
libraries, `serve.sh`, `download-model.sh`, the Windows launcher for WSL2 and the archive README
[`release-archive-linux.md`](../release-archive-linux.md). The model is not in it:
`download-model.sh` fetches it from Hugging Face. Tests, benchmarks and evaluation tools stay out.

The container image (`Dockerfile`, `.github/workflows/docker.yml`) and the Windows packager
(`scripts/package-release.ps1`, [`release-archive-windows.md`](../release-archive-windows.md)) are
NInfer-all's. This line has not published either, and GitHub Actions has not run on this
repository.

## The flow

1. **Prepare.** On the default branch, set `VERSION` to the tag (for example `0.1.0`) and write
   `RELEASE_NOTES_<tag>.md`. The packager names the archive from `VERSION` and refuses to run
   without the matching notes.
2. **Build.** `scripts/qwen38/build-release.sh` exports the committed tree, builds it in a
   container from `nvidia/cuda:13.1.1-devel-ubuntu24.04` (pinned by digest) with
   `scripts/qwen38/build.sh --arch 120a`, and runs [`scripts/package-release.sh`](../../scripts/package-release.sh).
   The result is `dist/ninfer-5060-qwen-<tag>-linux-x64.tar.gz` with its `.sha256`. The build tree
   stays in `build-release/build`, so a rebuild after a small change is quick.
3. **Check on the card** (below).
4. **Publish.** Tag the commit (`git tag -a v<tag>`), push the tag, then
   `gh release create v<tag> dist/ninfer-5060-qwen-<tag>-linux-x64.tar.gz{,.sha256}
   -R 2-com-org/ninfer-5060-qwen --notes-file RELEASE_NOTES_<tag>.md --title "..."`.

## Gotchas

**Always pass `-R 2-com-org/ninfer-5060-qwen` to `gh`.** The checkout may also have NInfer-all's
remotes, and `gh` without `-R` can resolve to the parent repository.

**The build host's glibc is the archive's minimum.** That is why the build runs on Ubuntu 24.04
(glibc 2.39) and not on the newer system the measurements used. The archive README states it.

**The toolkit is part of the measurement.** The published speeds were measured with CUDA 13.1.1
(nvcc 13.1.115). Another toolkit generates different code; re-measure before changing the base
image.

**The bundled libraries.** The packager copies `libcudart`, `libcublas` and `libcublasLt` from the
build's toolkit into `lib/` and sets the server's `DT_RPATH` to `$ORIGIN/../lib`, which the loader
searches before `LD_LIBRARY_PATH`, so a Toolkit elsewhere on the system cannot replace them. Every
other direct dependency must be the driver's `libcuda` or one the archive README lists (libcurl,
zlib, the C and C++ runtimes); the packager fails on anything else, and on a dynamically linked
FFmpeg or dav1d. Update the README and the packager's list together.

**Redistributing NVIDIA's libraries has conditions.** The CUDA Toolkit EULA (Attachment A) lists
the CUDA runtime and cuBLAS as redistributable. The grant requires that the application has
material functionality of its own, that the libraries are used only by it, and that the
distribution terms are consistent with the EULA and protect NVIDIA's rights. The archive therefore
carries the toolkit's `EULA.txt` as `licenses/NVIDIA-CUDA-EULA.txt`, and its README names the
libraries as NVIDIA's, outside the project's Apache-2.0 licence. Re-read the EULA of the toolkit
that builds a release.

**FFmpeg is LGPL and linked statically.** The archive carries the exact FFmpeg and dav1d source
archives the build compiled (`sources/`) with their licences, and the README points to this
repository's source for relinking.

**Asset size.** The server links every kernel; with cuBLAS the archive is several hundred
megabytes. GitHub rejects release assets over 2 GiB.

## Checks before publishing

- A fresh unpack verifies (`sha256sum -c SHA256SUMS.txt`, and the outer `.sha256`).
- `ldd bin/ninfer-serve` resolves `libcudart`, `libcublas` and `libcublasLt` from `lib/`, and
  `bin/ninfer-serve --help` runs.
- `./download-model.sh` and `./serve.sh` with nothing overridden start on an RTX 5060 Ti 16 GB at
  262,144 tokens and serve a chat completion and the WebUI.
- Greedy replies to the paper prompts match the measured build's, and the 30K and 160K prefill and
  decode speeds match it within run-to-run variation.
- Under WSL2, `windows\Start-Qwen38.cmd` opened from `\\wsl.localhost\...` starts the server.
