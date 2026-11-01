# NInfer-Qwen 0.1.0

The first prebuilt release of NInfer-Qwen: Qwen3.8-27B GSQ-RCO IQ3_XXS at its full 262,144-token
context on a 16 GB GeForce RTX 5060 Ti, for Linux and for Windows through WSL2.

## What is in it

`ninfer-5060-qwen-0.1.0-linux-x64.tar.gz` holds `ninfer-serve` for the RTX 50 series (`sm_120a`)
with the WebUI built in, NVIDIA's CUDA runtime and cuBLAS libraries, `serve.sh` with the measured
launch settings, `download-model.sh` for the 10.8 GB model, and the Windows launcher for WSL2. It
needs the NVIDIA driver 580 or newer, glibc 2.39 or newer (Ubuntu 24.04 or newer) and libcurl; no
CUDA Toolkit, build tools or Python. Tests, benchmarks and evaluation tools are not included.

`ninfer-5060-qwen-0.1.0-windows-wsl-x64.zip` is the same server for Windows, with nothing typed in
Linux. Its `Install.cmd` checks the GPU and WSL 2, imports the Linux archive with Ubuntu 24.04 as
the WSL distribution `ninfer-5060-qwen`, downloads the model and adds Start menu entries;
`Start-Qwen38.cmd` starts the server and opens the WebUI, and `Uninstall.cmd` removes it all. It
needs WSL 2 and the NVIDIA Windows driver 580 or newer, but no Linux distribution of your own.

The server was built from this tag with CUDA 13.1.1 on Ubuntu 24.04, the toolkit of the published
measurements.

## Performance

Measured on an RTX 5060 Ti 16 GB under Windows 11 + WSL2, one request, greedy, at the shipped
settings ([Performance](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Performance)):

| | |
|---|---:|
| Prefill, 30K-token prompt | 3,644-3,648 tok/s |
| Prefill, 160K-token prompt | 2,143 tok/s |
| Decode after the 30K / 160K prompt | 88.7 / 66.0 tok/s |
| Decode, short prompt | 87.0-87.7 tok/s |
| 8 / 32 requests at once, total decode | 342 / ~625 tok/s |

## Since the development branch

This line merged NInfer-all's `master` to replace it, and picked the fixes that reach this model
and card: each kernel copy's shared-memory limit under CUDA 13, TMA stage release, GDN gating
residency from the device's occupancy, the vision overlay window, the WebUI MCP relay's API key
and bounded buffer, the converter's logical GGUF order, PPM/PGM/PBM decoding, and FFmpeg 9.0.2 with
dav1d 1.5.4 linked statically ([Changes from
NInfer-all](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Changes-from-NInfer-all)).

## Known limits

- Tested only under WSL2 on Windows 11. Native Linux uses the same binary and scripts but has not
  been run.
- Tuned and measured on the RTX 5060 Ti 16 GB only. Other RTX 50 cards run with routes not
  measured for them.
- Evaluations so far: IFBench 79.33 and AIME 2025 96.67 (93.33 within the 122,880-token budget;
  the one cut-off reply was correct when rerun at the full context). AIME 2026, GPQA-Diamond and
  LiveCodeBench v6 are still to run.
