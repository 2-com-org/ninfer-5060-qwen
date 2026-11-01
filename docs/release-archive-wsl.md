# NInfer-Qwen for Windows (WSL2) — release archive

This README ships **inside** the Windows release zip. NInfer-Qwen serves
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in ISTA-DASLab's GSQ-RCO IQ3_XXS
quantization at its full 262,144-token context on a 16 GB GeForce RTX 5060 Ti. The server is a
Linux program; this package installs it with everything it needs as its own WSL2 distribution,
`ninfer-5060-qwen`, and runs it from Windows. Nothing has to be typed inside Linux. The model
(10.8 GB) is downloaded during setup. Source, build guide and measurements:
[github.com/2-com-org/ninfer-5060-qwen](https://github.com/2-com-org/ninfer-5060-qwen) and its
[wiki](https://github.com/2-com-org/ninfer-5060-qwen/wiki).

## Requirements

- **Windows 11** with **WSL 2** (Windows 10 22H2 with WSL 2 should work but is untested). Without it, open a terminal as administrator,
  run `wsl --install --no-distribution` and restart; setup says so when it is missing. No Linux
  distribution has to be installed first.
- An **RTX 50 series** GPU: the package carries an `sm_120a` build. It is tuned and measured on the
  RTX 5060 Ti 16 GB; other RTX 50 cards run with routes not measured for them.
- The **NVIDIA Windows driver 580 or newer**. WSL uses it; install no driver inside Linux.
- **32 GB of RAM** or more recommended: the conversation cache pins up to 22 GiB, or 45% of the
  memory WSL gives Linux (half of the PC's by default) when that is less.
- **About 14 GB of disk** for the distribution and the model.

## Quick start

1. Right-click the zip, choose **Extract All**, and open the extracted folder.
2. Double-click **`Install.cmd`**. If Windows warns that the file came from the internet, allow it to
   run. Setup checks the GPU and WSL, installs the distribution, downloads
   the model (resuming if interrupted: run `Install.cmd` again), checks its SHA-256 and adds
   **NInfer-Qwen** to the Start menu.
3. Start **Qwen3.8-27B** from the Start menu, or double-click `Start-Qwen38.cmd`.

The server's terminal shows its log. When it is ready (10-30 seconds), the WebUI opens at
<http://localhost:8080/>. Its live monitor is at <http://localhost:8080/monitor/>, and API clients use
the base URL `http://localhost:8080/v1` with the model name `qwen3.8-27b`. The server speaks the
OpenAI Chat Completions and Responses APIs, the Anthropic Messages API and the llama.cpp API.
Closing the terminal, or Ctrl+C in it, stops the server and releases GPU memory.

After setup the extracted folder can be deleted: the Start menu entries use copies in
`%LOCALAPPDATA%\ninfer-5060-qwen`, where the server logs are kept too.

## Setup options

`Install.cmd` takes two options, for example from a terminal in the extracted folder:

| Option | Effect |
|---|---|
| `-InstallDir D:\NInfer-Qwen` | put the distribution's disk (about 13 GB with the model) there instead of `%LOCALAPPDATA%\ninfer-5060-qwen\wsl`. Prefer an SSD: the server reads the model from it at every start, which on a hard disk takes about a minute and a half |
| `-ModelFile PATH` | copy a `qwen38-27b-gsq-iq3xxs.ninfer` downloaded before instead of downloading it; its size and SHA-256 are checked |

## Settings

`Start-Qwen38.cmd` takes the server's settings, for example `Start-Qwen38.cmd -Concurrency 4` or
`Start-Qwen38.cmd -LocalOnly`. The defaults are the measured single-request settings: 262,144 tokens,
MTP speculative decoding with 4 drafts and 31 copy tokens, a 4-bit KV cache, and thinking off
unless a request turns it on.

| Option | Effect |
|---|---|
| `-Concurrency N` | serve 1-32 requests at once from one shared KV pool; the default context shrinks with N |
| `-Context N` | tokens per request |
| `-Thinking` | think by default; requests can still override it |
| `-Temperature X` | default sampling temperature (greedy by default) |
| `-Port N`, `-LocalOnly` | where the server listens |
| `-NoBrowser` | do not open the WebUI |

The server listens on every interface and requires no API key; on a shared network pass
`-LocalOnly`. The [Serving](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Serving) page explains
each setting, and [Performance](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Performance) lists
the speeds measured with them.

The 262,144-token context leaves about 270 MiB of GPU memory. Programs that take more GPU memory
after startup (games, other GPU programs, a busy browser) make the Windows driver page GPU memory to
system RAM, which can halve speed: close them, or pass a smaller `-Context`.

## Uninstall

`Uninstall.cmd`, or **Uninstall NInfer-Qwen** in the Start menu, asks first. It then removes the
`ninfer-5060-qwen` distribution with the server and the model, the Start menu entries and the
launcher copies in `%LOCALAPPDATA%\ninfer-5060-qwen`; the logs there stay. Other WSL distributions
are not touched. A newer release installs after the older one is uninstalled.

## What is in the zip

| File | Contents |
|---|---|
| `Install.cmd` | sets up the distribution and downloads the model (`scripts\Install.ps1`) |
| `Start-Qwen38.cmd` | starts the server and opens the WebUI |
| `Uninstall.cmd` | removes everything setup installed (`scripts\Uninstall.ps1`) |
| `ninfer-5060-qwen-VERSION-wsl.tar.gz` | the distribution: Ubuntu 24.04 with curl, and the Linux release in `/opt/ninfer-5060-qwen` |
| `licenses\` | third-party licences of the server |
| `SHA256SUMS.txt` | checksums of every file here |

Inside the distribution, `/opt/ninfer-5060-qwen` is the Linux release archive: `bin/ninfer-serve` with
the WebUI built in, NVIDIA's CUDA runtime and cuBLAS in `lib/`, `serve.sh`, `download-model.sh`, the
launcher in `windows\` that `Start-Qwen38.cmd` runs, and the model in `models/`. Explorer shows it at
`\\wsl$\ninfer-5060-qwen\opt\ninfer-5060-qwen`.

## Licences

NInfer-Qwen is Apache-2.0 (`LICENSE`). The model is Apache-2.0. The CUDA runtime and cuBLAS
libraries are NVIDIA's, redistributed under the CUDA Toolkit End User License Agreement
(`licenses\NVIDIA-CUDA-EULA.txt`) and not covered by the project's licence. FFmpeg (LGPL 2.1 or
later) and dav1d (BSD 2-clause) are linked statically; their exact source archives are in the
distribution at `/opt/ninfer-5060-qwen/sources`, and the project's source, with the build that
links them, is at the repository above. The distribution's operating system is Ubuntu 24.04's
packages, each under the licence in `/usr/share/doc/PACKAGE/copyright` inside it; their sources are
in the Ubuntu archive. The other components' licences are in `licenses\`.
