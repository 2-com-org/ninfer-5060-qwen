# Troubleshooting

## Build

| Symptom | Cause and fix |
|---|---|
| `the checkout path contains a space` | nvcc's command lines break on spaces. Clone into a path without them, for example `~/ninfer-5060-qwen`. |
| `nvcc not found` | Install the CUDA Toolkit (13.1+), or set `CUDACXX=/usr/local/cuda-13.x/bin/nvcc`. |
| `CUDA 12.x miscompiles sm_120a kernels` | RTX 50 cards need CUDA 13.1 or newer. |
| The compiler is killed, or the machine stalls | Out of memory. Pass `build.sh --jobs 4` or fewer, or give WSL more memory in `%UserProfile%\.wslconfig`. |
| The build is very slow | The checkout is under `/mnt/c`. Clone inside the Linux filesystem instead. |

## Model preparation

| Symptom | Cause and fix |
|---|---|
| `does not match sha256` | The download is corrupt or incomplete. Delete the GGUF and rerun `prepare-model.sh`. |
| `no converter Python` | Run `scripts/qwen38/install-deps.sh`, or pass `--python` with a Python that has torch, numpy, safetensors and gguf. |
| The conversion is killed | Out of memory. Close other programs, or give WSL more memory. |

## Startup

| Symptom | Cause and fix |
|---|---|
| `requested Engine runtime reservation requires … bytes, but only … are available` | Another program holds GPU memory: under Windows the desktop, browsers, games and other CUDA programs. Close them, or pass a smaller `--context`. |
| Startup fails while pinning host memory | The RAM cache does not fit. Pass a smaller `--ram-cache-gib`. |
| `Port 8080 is already in use` | Another server is running. Stop it, or pass `--port`. |
| The first start takes 20-30 s | Cold file cache and CUDA module loading. Warm starts take about 10 s, plus about 6 s to pin a 22 GiB RAM cache. |

## Speed

| Symptom | Cause and fix |
|---|---|
| Decode or prefill suddenly 15-50% slower | Windows is paging GPU memory to system RAM because something took more memory after startup. `nvidia-smi dmon -s t` shows PCIe receive traffic during decode. Close GPU programs, view the WebUI from another device, or pass a smaller `--context`. |
| Slower for 30-60 s right after starting | Under WSL2, a server that starts within about 90 s of desktop activity gets a smaller GPU memory budget for a while. It recovers on its own. |
| Concurrent requests queue | Admission reserves each request's prompt plus its output budget. Set `max_tokens`, use `--lease-growth`, or use fewer or shorter requests. |

## Getting help

When reporting a problem, include:

- the `serve.sh --print` command;
- the server log (`%LOCALAPPDATA%\ninfer-5060-qwen\logs\…` with the Windows launcher);
- `nvidia-smi`;
- `nvcc --version`.
