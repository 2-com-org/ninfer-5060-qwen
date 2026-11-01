# NInfer-Qwen for Linux and WSL2 — release archive

This README ships **inside** the release archive. NInfer-Qwen serves
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in ISTA-DASLab's GSQ-RCO IQ3_XXS
quantization at its full 262,144-token context on a 16 GB GeForce RTX 5060 Ti. The archive holds
the server, the CUDA libraries it needs and its launch scripts; the model (10.8 GB) is downloaded
separately. Source, build guide and measurements:
[github.com/2-com-org/ninfer-5060-qwen](https://github.com/2-com-org/ninfer-5060-qwen) and its
[wiki](https://github.com/2-com-org/ninfer-5060-qwen/wiki). Links here are absolute because the
archive has no `docs/` directory.

`VERSION` names this release and `RELEASE_NOTES_*.md` says what it contains.

## Requirements

- An **RTX 50 series** GPU: the archive carries an `sm_120a` build. It is tuned and measured on the
  RTX 5060 Ti 16 GB; other RTX 50 cards run with routes not measured for them.
- **NVIDIA driver 580 or newer**: the Linux driver, or under WSL2 the Windows driver (install no
  Linux driver inside WSL).
- **x86-64 Linux with glibc 2.39 or newer**, such as Ubuntu 24.04 or newer, natively or under WSL2.
- **libcurl 4** and **curl**, present on most systems (`sudo apt-get install libcurl4t64 curl` on
  Ubuntu 24.04).
- **32 GB of RAM** or more recommended: the conversation cache pins up to 22 GiB (45% of RAM when
  that is less).
- **About 12 GB of disk** for the model.

Nothing else: the CUDA runtime and cuBLAS are in `lib/`, and FFmpeg is linked into the server.

## Quick start

Unpack the archive into a path without spaces, download the model and start the server:

```bash
tar -xzf ninfer-5060-qwen-*-linux-x64.tar.gz
cd ninfer-5060-qwen-*-linux-x64
./download-model.sh
./serve.sh
```

`download-model.sh` resumes an interrupted download and checks the file's size and SHA-256. When
the server is ready, open the WebUI at <http://localhost:8080/>. Its live monitor is at
<http://localhost:8080/monitor/>, and API clients use the base URL `http://localhost:8080/v1` with
the model name `qwen3.8-27b`. The server speaks the OpenAI Chat Completions and Responses APIs,
the Anthropic Messages API and the llama.cpp API. Ctrl+C stops it.

The server listens on every interface and requires no API key. On a shared network pass
`--local-only`, or require a key:

```bash
./serve.sh --local-only
./serve.sh -- --api-key YOUR-KEY
```

## Windows (WSL2)

The Windows release, `ninfer-5060-qwen-VERSION-windows-wsl-x64.zip`, installs this archive as its
own WSL distribution from `Install.cmd` and starts it from the Start menu, with nothing typed in
Linux. To run this archive in an Ubuntu of your own under WSL2 instead:

1. Install WSL2 with Ubuntu from a Windows terminal: `wsl --install -d Ubuntu`. Open Ubuntu.
2. Inside Ubuntu, unpack the archive into your Linux home directory and run `./download-model.sh`
   there, as in the quick start. Keep the files in WSL's own file system: reading the model through
   `/mnt/c` makes startup much slower.
3. In Windows, open `windows\Start-Qwen38.cmd` inside the unpacked folder, for example
   `\\wsl.localhost\Ubuntu\home\YOUR-NAME\ninfer-5060-qwen-VERSION-linux-x64\windows\Start-Qwen38.cmd`.
   It starts the server, opens the WebUI and shows the server log. Closing the window stops the
   server and releases GPU memory.

`Start-Qwen38.cmd` takes the same settings as `serve.sh`, for example
`Start-Qwen38.cmd -Concurrency 4` or `Start-Qwen38.cmd -LocalOnly`; `Get-Help .\Start-Qwen38.ps1
-Detailed` lists them. Logs are kept in `%LOCALAPPDATA%\ninfer-5060-qwen\logs`.

Under Windows the 262,144-token context leaves about 270 MiB of GPU memory. Programs that take
more GPU memory after startup (games, other GPU programs, a busy browser) make the Windows driver
page GPU memory to system RAM, which can halve speed: close them, or pass a smaller `--context`.

## Settings

`./serve.sh --help` documents every option. The defaults are the measured single-request
settings: 262,144 tokens, MTP speculative decoding with 4 drafts and 31 copy tokens, a 4-bit KV
cache, and thinking off unless a request turns it on. The common changes:

| Option | Effect |
|---|---|
| `--concurrency N` | serve 1-32 requests at once from one shared KV pool; the default context shrinks with N |
| `--context N` | tokens per request |
| `--thinking` | think by default; requests can still override it |
| `--temperature X` | default sampling temperature (greedy by default) |
| `--port N`, `--local-only` | where the server listens |
| `-- ARGS` | pass further `ninfer-serve` options, as `bin/ninfer-serve --help` lists them |

The [Serving](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Serving) page explains each
setting, and [Performance](https://github.com/2-com-org/ninfer-5060-qwen/wiki/Performance) lists
the speeds measured with them.

## What is in the archive

| File | Contents |
|---|---|
| `serve.sh` | starts the server with the measured settings |
| `download-model.sh` | downloads and checks the model into `models/` |
| `bin/ninfer-serve` | the server, with the WebUI built in |
| `lib/` | NVIDIA's CUDA runtime and cuBLAS libraries, which the server loads from here |
| `windows/` | the Windows launcher for WSL2 |
| `licenses/`, `sources/` | third-party licences, and the FFmpeg and dav1d source archives linked into the server |
| `SHA256SUMS.txt` | checksums of every file here: `sha256sum -c SHA256SUMS.txt` |

## Licences

NInfer-Qwen is Apache-2.0 (`LICENSE`). The model is Apache-2.0. The libraries in `lib/` are NVIDIA's,
redistributed under the CUDA Toolkit End User License Agreement (`licenses/NVIDIA-CUDA-EULA.txt`)
and not covered by the project's licence. FFmpeg (LGPL 2.1 or later) and dav1d (BSD 2-clause) are
linked statically; their exact source archives are in `sources/`, and the project's source, with
the build that links them, is at the repository above. The other components' licences are in
`licenses/`.
