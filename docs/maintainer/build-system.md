# Build system

NInfer builds from its source tree with CMake 3.28+ and C++/CUDA 20.
`CMAKE_CUDA_ARCHITECTURES` is `80`, `86` (the default), `89` or `120a`, one per build tree; configure
refuses anything else. CUDA 12.8 builds the `sm_8x` targets, and a `120a` build needs CUDA 13.1 or
newer.
Product commands and prerequisites are in the README's [Running](../ninfer-all.md#running) and
[Building](../ninfer-all.md#building) sections; test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. Architecture selection and validation happen before `project()` detects CUDA.
The default configuration is Release. Ninja links and archives share the single-slot
`ninfer_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `NINFER_BUILD_APPS` | ON | CLI, HTTP server, perplexity evaluator and `ninfer-calibrate` |
| `BUILD_TESTING` | OFF | C++ tests and registered Python interoperability tests |
| `NINFER_BUILD_BENCHMARKS` | OFF | Op, model, Engine and context-cost benchmarks |

The opt-in kernel, build and platform options (`NINFER_SM120_NATIVE`, `NINFER_PDL`,
`NINFER_MULTICALL`, the SFU and Windows options, and the rest) are listed in the
[Linux build guide](../build-linux.md#build-options).

Apps or tests enable the internal product support components: media acquisition, prompt input,
logging and serving. This is one derived condition, not a separate user option. FFmpeg belongs
to model media decoding and remains required when apps are disabled. Curl and spdlog are needed
only for product support. A Python 3 interpreter is found when tests are configured. The larger Python
pytest suites and conversion/evaluation scripts run separately from CMake and CTest.

Python 3.11 describes the maintained environment, not a configuration version gate. The one CUDA
version gate is 13.1 for a `120a` build: CUDA 12.8 and 12.9 miscompile `sm_120a` kernels.
CMake also discovers FFmpeg without imposing library version floors. Actual language/API support
is exercised by compilation and tests. The libcurl 7.85 minimum has a concrete API basis:
media acquisition uses `CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`, introduced in
[that release](https://curl.se/libcurl/c/CURLOPT_PROTOCOLS_STR.html).

The checked-in `CMakePresets.json` provides:

- `release`: Release product binaries, tests and benchmarks disabled;
- `dev`: Release products, tests and benchmarks enabled;
- matching build presets and a `dev` test preset with failure output enabled.

Both configure presets use `build/`. Switching between them explicitly resets all three build
options; it does not create independent build trees. Build with `cmake --build build -j`.
For a separate configuration, override the binary directory with `-B build-<name>` and use that
directory in subsequent build/test commands.

Keep machine paths in the ignored `CMakeUserPresets.json`. For example, replacing the interpreter
placeholder with the interpreter from the selected environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev",
      "cacheVariables": {
        "Python3_EXECUTABLE": "/path/to/python3"
      }
    }
  ]
}
```

Configure that environment with `cmake --preset local`. Compiler paths may also be supplied here;
changing compilers requires a fresh build directory. There is no tools option, installed SDK,
package export or configure-time dependency download.

## Targets and dependencies

`cmake/Dependencies.cmake` discovers system CUDA, Threads, FFmpeg and conditional curl, and
exposes the repository-pinned JSON/HTTP headers and conditional spdlog library. External include
requirements follow their consuming targets. `cmake/NinferTargets.cmake` provides private
project includes and the two CUDA archive policies.

`src/CMakeLists.txt` explicitly enters the component directories. Each component's
`CMakeLists.txt` owns its targets, dependency visibility and compile properties:

| Component | Targets |
|---|---|
| `core/` | `ninfer_core` |
| `artifact/` | `ninfer_artifact` |
| `text/` | `ninfer_text`, including bundled utf8proc C source |
| `media/` | `ninfer_media_decode` |
| `ops/` | `ninfer_ops`, `ninfer_nvfp4_non_rdc` |
| `models/` | `ninfer_model_loading`, `ninfer_model_runtime` |
| `runtime/` | `ninfer_runtime_support`, `ninfer_engine` and its `ninfer::engine` alias |
| `product/` | media acquisition, prompt input and logging libraries |
| `serve/` | `ninfer_serve` |

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register simple horizontal implementations in `basic_sources.cmake` and larger semantic
  families in their own manifests. A family also registers its files under `wrapper/`.
- Linear includes one manifest for each of BF16, FP8, NVFP4, Q4, Q5, Q6 and Q8. Each manifest lists
  all its shape filenames and their actual `.cpp`/`.cu` extensions.
- Models separate loading, frontend, execution and Program source lists while retaining the
  distinct immutable-loading and mutable-runtime targets. Checkpoints/recipes do not define
  additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.
- One manifest generates instantiation units: `softmax_attention/sources.cmake` configures
  `small_t_i8_unit.cu.in` once per query width, head geometry, cache input and INT8-family KV
  storage it enumerates (140 units in the build tree), each one explicit instantiation of the
  small-T partial launch. The dispatch over those parameters stays in C++ (`small_t.cu`); the
  units exist to keep each storage's kernels in their own module (see below).

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units.

## CUDA compilation boundaries

`ninfer_core` and `ninfer_ops` compile whole-program: no relocatable device code (RDC) and no
device link. Every device symbol is defined and used in one translation unit. A change that needs a
device symbol across translation units has to bring RDC back to its target and measure the cost:
under RDC ptxas lowers a shuffle from a computed lane to an out-of-line call and stops pipelining
loads across loop back edges.

`ninfer_nvfp4_non_rdc` explicitly disables separable compilation and device-symbol resolution. It
contains the warp-specialized Linear, LinearSwiGLU and causal-attention NVFP4 sources that depend
on `setmaxnreg` register transfer, which RDC would discard. Their owning family manifests register
these sources into that target; host launchers connect them to the normal Ops.

The CUDA archives and the microbenchmarks embed source-line tables (`-lineinfo`) only when
configured with `-DNINFER_CUDA_LINEINFO=ON`, for Nsight Compute source views. The tables cost run
time, not only binary size: CUDA loads a module lazily, when a process first uses one of its
kernels, and expands and parses the whole module then. On the RTX 5060 Ti (`120a`, CUDA 13.1, WSL2
driver) the INT8-family width-3, 24-head small-T append module (then 452 kernels, every storage's)
expands to 234 MB with the tables, three quarters of it PTX text and line tables, against 5.8 MB
without them; its first load took 318-378 ms against 94-106 ms. A served request pays that the first
time a prompt's last chunk reaches a module, inside its time to first token. Dropping the tables
(same SASS, byte-identical replies) took the server's first 17,936-token prompt from 5.2 to 4.8 s to
first token, its CUDA Graph startup from 4.1-4.5 to 1.4 s and its kernel image from 0.55 to 0.26 GB,
with later prompts unchanged (27B, rk4v4, 262,144 context). The embedded `compute_120a` PTX and the
fatbin compression mode did not change the load time.

The same cost is why the INT8-family small-T units are split per KV storage: a process uses one
storage, and a module holding all five carried four storages' kernels into every first load. On the
same card the width-4, 24-head rk4v4 modules went from 452 to 100 kernels (append 45.7 to 11.1 MB,
cached 34.3 to 7.8 MB) and their first load from 107-112 to 17-19 ms and from 82-83 to 12-13 ms; the
server's first request after startup (14 tokens) reached its first token in 181-199 ms instead of
386-475 and its CUDA Graph startup took 1.1-1.3 s instead of 1.5-1.7, with later requests
unchanged. The split costs host code per unit: the server binary grows from 405 to 409 MB.

Splitting also moved those loads into the requests that first reach each width: a returning turn
whose prefill first reached a width took 70-115 ms longer, and one start's first request waited
2.0 s. A Program therefore loads its modules at startup
(`ops::causal_softmax_attention_load_modules`, from the Program constructor): every appending width
up to its widest call and, with MTP, the drafter's cached width 1. `cudaGetKernel` on one kernel of
each unit parses the module without loading code into the context, so free device memory after
startup does not change. On the RTX 5060 Ti (27B, rk4v4, 262,144 context, three starts per arm) the
first request after start reached its first token in 103-105 ms instead of 259-280, the turns that
had carried a module load in 163-168 ms instead of 236-281 (156K context) and 104-105 instead of
174-202 (3.6K), and a fresh server's 30K-token first prompt prefilled at 3,644-3,648 tok/s instead
of 3,590-3,605; other turns, decode and paper replies were unchanged. The loads add 0.6-0.9 s to
Program initialization, of which the CUDA Graph phase (1.9 to 1.6-1.7 s) and the server warmup give
back 0.25-0.4 s. Neither an uncompressed fatbin nor loading modules from several host threads
(they serialize in the driver) made the parse faster.

Whole-program compilation is not a speed lever on every card. On the RTX 3090 (sm_86, CUDA 12.8, 27B
DFlash2 bundle, rk4v4 KV, three interleaved rounds) the base line measured the switch at
pp512/pp2048/pp8192 +0.4/-0.6/-1.4% and decode within +-0.3%, all inside run-to-run spread, while
the server binary grows from 498 to 637 MB; the sm_89 fork reported +2.7-4.9% prefill from it. What
it buys everywhere is the code generation described above; re-measure speed before attributing a
gain to it on a new architecture or toolchain.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines the four product executables, or with `NINFER_MULTICALL`
one multi-call executable that bundles them. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`build/tests/`, `build/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Linear, LinearAdd and LinearSwiGLU reuse compiled test support libraries. Oracle options apply
to those libraries themselves as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
