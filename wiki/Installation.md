# Installation

This line was developed and measured on Windows 11 with WSL2 Ubuntu. Native Ubuntu uses the same
scripts but has not been tested yet.

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
git clone --branch qwen38-5060ti https://github.com/2-com-org/ninfer-5060-qwen.git ~/ninfer-5060-qwen
```

## 4. Setup

`setup.sh` runs the three steps below in order. Each one can also be run on its own:

```bash
~/ninfer-5060-qwen/scripts/qwen38/setup.sh
```

### `install-deps.sh`

This step installs the system packages with `apt-get` (pass `--no-apt` to skip it):

- the compilers, CMake and Ninja;
- the FFmpeg and libcurl headers;
- `python3-venv`.

It then creates `.venv-convert/` for the weight converter, with PyTorch's CPU build (about 1 GB),
NumPy, safetensors, gguf and huggingface_hub at the versions in
[`requirements-convert.txt`](../scripts/qwen38/requirements-convert.txt). It looks for Python in
this order:

1. `python3.11`;
2. `uv`, which can fetch Python 3.11;
3. `python3`. The converter is maintained on 3.11, so a newer `python3` is not guaranteed to work.

### `build.sh`

`build.sh` reads the GPU's compute capability from `nvidia-smi`. It maps 12.0 to `120a`, 8.9 to
`89` and 8.6 to `86`; `--arch` sets it by hand. It then:

1. Downloads llama.cpp's released WebUI (b11165) into `third_party/webui/`.
2. Configures a Release build in `build-qwen38/`.
3. Builds `ninfer-serve` and `ninfer-perplexity`.

| Option | |
|---|---|
| `--arch 120a\|89\|86` | override the detected architecture |
| `--jobs N` | parallel compiler jobs. The default is the smaller of the core count and RAM/3 GiB, since one nvcc job can take 3 GB |
| `--build-dir DIR` | default `build-qwen38` |
| `--no-webui` | build the API without the WebUI |

Expect 30-60 minutes on 12 cores. The RTX 5060 Ti build turns on programmatic dependent launch
(`NINFER_PDL=ON`), as the measured binaries did.

### `prepare-model.sh`

This step downloads and converts the model. [Model preparation](Model-Preparation.md) covers it,
and how to download the converted artifact instead.

## Build by hand

`build.sh` is equivalent to the following, run in the checkout:

```bash
scripts/fetch-webui.sh b11165
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
