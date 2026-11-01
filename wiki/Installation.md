# Installation

This line was developed and measured on Windows 11 with WSL2 Ubuntu. Native Ubuntu uses the same
scripts but has not been tested yet.

## Prebuilt release (RTX 50 series)

The [releases](https://github.com/2-com-org/ninfer-5060-qwen/releases) carry two archives for the
RTX 50 series. Neither needs a CUDA Toolkit, build packages or Python; both need the NVIDIA driver
580 or newer.

**Windows:** `ninfer-5060-qwen-VERSION-windows-wsl-x64.zip`. Extract it and double-click
`Install.cmd`: it checks the GPU and WSL 2, imports the Linux archive below with Ubuntu 24.04 as its
own WSL distribution, `ninfer-5060-qwen`, downloads the model and adds **NInfer-Qwen** to the Start
menu. **Qwen3.8-27B** there (or `Start-Qwen38.cmd`) starts the server through the launcher in
[Serving](Serving.md#windows-launcher); `Uninstall.cmd` removes the distribution and the model. It
needs WSL 2 but no Linux distribution of your own; `Install.cmd -InstallDir D:\NInfer-Qwen` puts
its disk on another drive. The zip's README lists its options, contents and licences.

**Linux:** `ninfer-5060-qwen-VERSION-linux-x64.tar.gz`: `ninfer-serve` with the WebUI, the CUDA
runtime and cuBLAS it loads, `serve.sh`, `download-model.sh` and the Windows launcher. It needs
glibc 2.39 or newer (Ubuntu 24.04 or newer) and libcurl. It also runs in an Ubuntu of your own
under WSL2 (set up as in step 1 below); unpack it in your Linux home directory:

```bash
tar -xzf ninfer-5060-qwen-*-linux-x64.tar.gz
cd ninfer-5060-qwen-*-linux-x64
./download-model.sh
./serve.sh
```

Then `windows\Start-Qwen38.cmd` in the unpacked folder (open it through
`\\wsl.localhost\Ubuntu\home\...`) starts the server from Windows and opens the WebUI. The
archive's README lists its contents and licences. The rest of this page builds from source.

## 1. WSL2 and Ubuntu (Windows only)

In an administrator PowerShell, install Ubuntu, then restart if asked:

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu from the Start menu and create your Linux user. Keep the NVIDIA **Windows** driver
current (580 or newer). WSL uses it, so never install a Linux NVIDIA driver inside WSL. Check that
the GPU is visible:

```bash
nvidia-smi
```

WSL gives Linux half of the PC's RAM by default. The conversation cache pins up to 22 GiB of it,
or 45% of what Linux sees when that is less. For more, raise `memory=` in `%UserProfile%\.wslconfig`.

## 2. CUDA Toolkit

Install **CUDA 13.1 or newer** from [NVIDIA's downloads](https://developer.nvidia.com/cuda-downloads):

- Under WSL2: Linux → x86_64 → WSL-Ubuntu.
- On native Ubuntu: the Ubuntu package.

CUDA 12.8 and 12.9 miscompile `sm_120a` kernels, and the build refuses them for RTX 50 cards.
`build.sh` finds `nvcc` on `PATH` or in `/usr/local/cuda/bin`; `CUDACXX` overrides both.

## 3. Clone

Clone into a Linux path **without spaces**: nvcc's command lines break on them. Avoid
`/mnt/c/...`, because building through the Windows filesystem is several times slower.

```bash
git clone https://github.com/2-com-org/ninfer-5060-qwen.git ~/ninfer-5060-qwen
```

## 4. Setup

`setup.sh` runs the three steps below in order. Each one can also be run on its own:

```bash
~/ninfer-5060-qwen/scripts/qwen38/setup.sh
```

### `install-deps.sh`

This step installs the system packages with `apt-get` (pass `--no-apt` to skip it):

- the compilers, CMake and Ninja;
- meson, nasm and the zlib headers, which build FFmpeg and dav1d;
- the libcurl headers;
- `python3-venv`.

With `--converter` (which `setup.sh --convert` passes) it also creates `.venv-convert/` for the
weight converter, with PyTorch's CPU build (about 1 GB), NumPy, safetensors, gguf and
huggingface_hub at the versions in
[`requirements-convert.txt`](../scripts/qwen38/requirements-convert.txt). The default setup
downloads the converted model and needs none of it. It looks for Python in this order:

1. `python3.11`;
2. `uv`, which can fetch Python 3.11;
3. `python3`. The converter is maintained on 3.11, so a newer `python3` is not guaranteed to work.

### `build.sh`

`build.sh` reads the GPU's compute capability from `nvidia-smi`. It maps 12.0 to `120a`, 8.9 to
`89` and 8.6 to `86`; `--arch` sets it by hand. It then:

1. Downloads llama.cpp's released WebUI (b11165) into `third_party/webui/`.
2. Downloads the FFmpeg 9.0.2 and dav1d 1.5.4 release archives into `third_party/media-sources/`
   and checks their SHA-256 digests. The build compiles them and links them into the server, with
   only the decoders of the image and video formats it reads, so the server needs no FFmpeg
   libraries at run time.
3. Configures a Release build in `build-qwen38/`.
4. Builds `ninfer-serve` and `ninfer-perplexity`.

| Option | |
|---|---|
| `--arch 120a\|89\|86` | override the detected architecture |
| `--jobs N` | parallel compiler jobs. The default is the smaller of the core count and RAM/3 GiB, since one nvcc job can take 3 GB |
| `--build-dir DIR` | default `build-qwen38` |
| `--no-webui` | build the API without the WebUI |

Expect 30-60 minutes on 12 cores. The RTX 5060 Ti build turns on programmatic dependent launch
(`NINFER_PDL=ON`), as the measured binaries did.

### `download-model.sh`

This step downloads the converted model (10.8 GB) and its conversion record from
[Hugging Face](https://huggingface.co/FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer) at a pinned
revision into `models/`. It resumes an interrupted download and checks each file's size and
SHA-256. `--models-dir DIR` puts them elsewhere.

`setup.sh --convert` runs `prepare-model.sh` instead, which downloads the original GGUF and
converts it into an artifact with the same weights. [Model preparation](Model-Preparation.md)
covers it.

## Build by hand

`build.sh` is equivalent to the following, run in the checkout:

```bash
scripts/fetch-webui.sh b11165
```

```bash
scripts/fetch-media-sources.sh
```

```bash
cmake -S . -B build-qwen38 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_PDL=ON \
  -DNINFER_WEBUI_DIR=$PWD/third_party/webui/b11165/llama-b11165
```

```bash
cmake --build build-qwen38 --parallel 8 --target ninfer-serve ninfer-perplexity
```

The `rtx5060` and `rtx5060-dev` presets in `CMakePresets.json` set the same options. `rtx5060-dev`
also builds the tests and benchmarks; [tests/README.md](../tests/README.md) explains how to run
them. The full build-system reference is
[docs/maintainer/build-system.md](../docs/maintainer/build-system.md).

## Tested toolchain

The published measurements used this toolchain:

- Ubuntu 26.04 under WSL2;
- CUDA 13.1.115, GCC 15.2, CMake 4.2.3, Ninja 1.13.2;
- Windows driver 610.88 and 621.01;
- converter environment: Python 3.11.16, torch 2.14.1+cpu, numpy 2.4.6, gguf 0.19.0.
