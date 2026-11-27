# Gemma 4 on the NVIDIA TensorRT-RTX backend: reproduce the build and the benchmarks

Tooling: `litert/vendors/nvidia/benchmark/` in the LiteRT checkout (its `README.md` is the full reference).
Platform ids: `5080` (RTX 5080, native Windows x64), `5080_wsl` (WSL2 Linux on the same host), `rtx_spark` (RTX Spark
N1X, native Windows ARM64).

## 1. Commits (verified 2026-11-17)

| | LiteRT | LiteRT-LM |
|---|---|---|
| RTX 5080 (Windows, WSL) | `fa232ccb4bab214ba0227c67e06a9268cad0be06` | `42531a013e039c39f53a523f4635770c2741883a` |
| RTX Spark | `fa232ccb4bab214ba0227c67e06a9268cad0be06` | `42531a01` + cherry-pick of PR #4459 (`2b9cff5b`), as one local commit |

| Build id (`BUILD_ID` of the staged runtime) | RTX 5080 `fa232ccb4-42531a01` · RTX Spark `fa232ccb4-<cherry-pick commit>` |
|---|---|

## 2. What to obtain

| Item | Version | Where |
|---|---|---|
| Gemma 4 26B LiteRT-LM model | `26b_drafter_vision.litertlm`, 16,906,960,896 bytes, SHA-256 `3d35318b4884cc40d5925612f6fd55e94ddafd774dbf8ce9ec3dc8b2ee233cd3` | from the model team; check the hash |
| TensorRT-RTX SDK | 1.7.1 (Windows 1.7.1.116; Linux 1.7.1.97) | NVIDIA |
| CUDA toolkit | 12.9 (RTX 5080 Windows and WSL), 13.4 (RTX Spark) | NVIDIA |
| Windows build tools | MSVC Build Tools 2022 (ARM64 tools on the Spark), Git for Windows, Bazelisk, Python 3.12+ | |
| llama.cpp | v1.0.0 = release `b12359` = commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` | below |
| GGUF models (llama.cpp) | `unsloth/gemma-4-26B-A4B-it-qat-GGUF`: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (SHA-256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`), drafter `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` | Hugging Face |

GGUF download (any machine):

```bash
pip install -U huggingface_hub
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf
# Linux: the runner reads the Hugging Face cache (~/.cache/huggingface/hub) directly.
# Windows: add --local-dir C:/models/gemma-4-26B-A4B-it-qat-GGUF and point config.json's llama.models at the files.
```

llama.cpp v1.0.0, Windows (the release binaries the tables used; the cudart/cublas DLLs go beside `llama-bench.exe`):

```bash
# RTX 5080 (x64); on the RTX Spark use the ...-win-cuda-13.5-arm64.zip pair.
gh release download b12359 -R ggml-org/llama.cpp -p 'llama-b12359-bin-win-cuda-13.5-x64.zip' -p 'cudart-llama-bin-win-cuda-13.5-x64.zip'
```

```powershell
$d = 'C:\llama\llama-b12359-bin-win-cuda-13.5-x64'
Expand-Archive llama-b12359-bin-win-cuda-13.5-x64.zip -DestinationPath $d
Expand-Archive cudart-llama-bin-win-cuda-13.5-x64.zip -DestinationPath $d -Force
```

llama.cpp v1.0.0, Linux (source build; not part of the 2026-11-17 verification):

```bash
git clone https://github.com/ggml-org/llama.cpp ~/odml/llama.cpp && git -C ~/odml/llama.cpp checkout v1.0.0   # b1b770d9
cmake -S ~/odml/llama.cpp -B ~/odml/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/odml/llama.cpp/build -j --target llama-bench llama-server
```

## 3. Checkouts

```bash
git config --global core.autocrlf false     # else the build rewrites cxxbridge_cmd/Cargo.lock and the build id gets '+'
git clone https://github.com/google-ai-edge/LiteRT    && git -C LiteRT    checkout --detach fa232ccb4bab214ba0227c67e06a9268cad0be06
git clone https://github.com/google-ai-edge/LiteRT-LM && git -C LiteRT-LM checkout --detach 42531a013e039c39f53a523f4635770c2741883a
```

RTX Spark only:

```bash
cd LiteRT-LM
git fetch https://github.com/google-ai-edge/LiteRT-LM pull/4459/head
git switch -c remote-20261117-arm64
git cherry-pick 2b9cff5be57f7daa5f8bf20bb4fdb7d17ea19302
```

## 4. Machine settings: how the local override works

| | Linux (WSL) | Windows |
|---|---|---|
| File | `litert/vendors/nvidia/benchmark/env.local.sh` | `litert/vendors/nvidia/benchmark/windows/config.json` |
| Tracked? | no (write it yourself; template `env.local.sh.example`) | no (git-ignored; template `windows/config.example.json`) |
| Read by | `env.sh`, sourced by `build.sh`, `bench.sh`, `matrix.sh`, `llama/*.sh` | `windows/build.ps1`, `windows/*.tool`, `llama/*.tool`, `report/*` |
| Precedence | `env.local.sh` is sourced first; a variable it exports wins over the shell's and over `env.sh`'s default (`${VAR:-default}`); unset = default. To let the shell win, write `export VAR=${VAR:-value}` | `--config FILE` (tools) / `-Config FILE` (`build.ps1`), else `LITERT_BENCH_WINDOWS_CONFIG`, else `windows/config.json` |

### WSL (RTX 5080 host): `env.local.sh`

```bash
export LITERT_LM_DIR=$HOME/odml/llm/LiteRT-LM                       # the LiteRT-LM checkout
export TENSORRT_RTX_ROOT=$HOME/sdk/TensorRT-RTX-1.7.1.97
export CUDA_HOME=/usr/local/cuda-12.9
export LITERT_BENCH_ROOT=$HOME/.local/state/litert-nvidia-benchmark   # builds, model caches, measured/
export LITERT_BENCH_MODELS=$HOME/odml/bench_models                    # .litertlm files (layout below)
export LITERT_BENCH_JOBS=6
export LITERT_BENCH_BUILD_RAM_MB=6000
export LLAMA_CPP_DIR=$HOME/odml/llama.cpp                             # llama.cpp v1.0.0 checkout
export LLAMA_BIN=$LLAMA_CPP_DIR/build/bin
export LLAMA_HF_HUB=$HOME/.cache/huggingface/hub                      # where `hf download` put the GGUFs
```

All variables (all optional; default in parentheses): `LITERT_RT_DIR` (the tree `env.sh` is in), `LITERT_LM_DIR`
(`../../llm/LiteRT-LM` next to it), `TENSORRT_RTX_ROOT` (`~/opt/tensorrt-rtx-sdk`), `CUDA_HOME` (`/usr/local/cuda`),
`LITERT_BENCH_ROOT` (`~/.local/state/litert-nvidia-benchmark-wl`), `LITERT_BENCH_MODELS` (`$LITERT_LM_DIR/../models`),
`LITERT_BENCH_RUNTIME_LIBS` (`build.sh`'s output), `LITERT_BENCH_JOBS`, `LITERT_BENCH_BUILD_RAM_MB`,
`LITERT_BENCH_CC`/`CXX` (`/usr/bin/clang`, `clang++`), `LITERT_BENCH_DISTDIR`, `LITERT_BENCH_PYTHON` (`python3`),
`LLAMA_CPP_DIR` (`~/odml/llama.cpp`), `LLAMA_BIN` (`$LLAMA_CPP_DIR/build/bin`), `LLAMA_HF_HUB`
(`~/.cache/huggingface/hub`), `LLAMA_LOCAL_MODELS` (`$LITERT_BENCH_MODELS`).

Model layout: `--model 26b` reads a fixed name under `LITERT_BENCH_MODELS`:

```bash
mkdir -p ~/odml/bench_models/gemma-4-26B-moe-mtp-external
ln -s /path/to/26b_drafter_vision.litertlm \
      ~/odml/bench_models/gemma-4-26B-moe-mtp-external/gemma-4-26B-moe-mtp.external.litertlm
```

### RTX 5080 (native Windows x64): `windows/config.json`

```json
{
  "platform": "5080",
  "host_mode": "diagnostic",
  "runtime_dir": "C:/Users/<user>/dev/runtime/fa232ccb4-42531a01",
  "cli": "litert_lm_advanced_main.exe",
  "sdk_bin_dirs": [
    "C:/Users/<user>/dev/sdk/TensorRT-RTX-1.7.1.116/bin",
    "C:/Users/<user>/dev/toolchains/cuda-12.9/bin"
  ],
  "state_root": "C:/Users/<user>/dev/bench_state",
  "litert_dir": "C:/Users/<user>/dev/LiteRT",
  "litert_lm_dir": "C:/Users/<user>/dev/LiteRT-LM",
  "prompts_dir": null,
  "models": {
    "26b": "C:/Users/<user>/dev/models/26b_drafter_vision.litertlm"
  },
  "host": { "require_ac": false, "overlay_ac_scheme": null },
  "llama": {
    "bin": "C:/llama/llama-b12359-bin-win-cuda-13.5-x64",
    "release_tag": "v1.0.0",
    "commit": "b1b770d90d94cad9e9ea5b993bdbc383a522fb28",
    "threads": 20,
    "models": {
      "26b": {
        "main": "C:/models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "draft": "C:/models/gemma-4-26B-A4B-it-qat-GGUF/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
      }
    }
  },
  "build": {
    "bazel": "bazelisk.exe",
    "msvc": { "script": "C:/BuildTools/VC/Auxiliary/Build/vcvars64.bat", "args": [] },
    "cuda_home": "C:/Users/<user>/dev/toolchains/cuda-12.9",
    "tensorrt_rtx_root": "C:/Users/<user>/dev/sdk/TensorRT-RTX-1.7.1.116",
    "env": {
      "BAZEL_VC": "C:/BuildTools/VC",
      "BAZEL_VS": "C:/BuildTools",
      "BAZEL_SH": "C:/Program Files/Git/bin/bash.exe",
      "GIT_LFS_SKIP_SMUDGE": "1"
    },
    "subst": null,
    "litert": {
      "startup_args": ["--output_user_root=C:/bz/user", "--output_base=C:/bz/rt"],
      "args": [
        "-c", "opt", "--jobs=8", "--local_ram_resources=8192",
        "--noincompatible_enable_android_toolchain_resolution",
        "--python_path=C:/Users/<user>/AppData/Local/Programs/Python/Python312/python.exe",
        "--action_env=PYTHON_BIN_PATH=C:/Users/<user>/AppData/Local/Programs/Python/Python312/python.exe",
        "--config=windows",
        "--define=protobuf_allow_msvc=true",
        "--repo_env=CUDA_HOME=C:/Users/<user>/dev/toolchains/cuda-12.9",
        "--repo_env=TENSORRT_RTX_ROOT=C:/Users/<user>/dev/sdk/TensorRT-RTX-1.7.1.116"
      ]
    },
    "litert_lm": {
      "startup_args": ["--output_user_root=C:/bz/user", "--output_base=C:/bz/lm"],
      "args": [
        "-c", "opt", "--jobs=8", "--local_ram_resources=8192",
        "--noincompatible_enable_android_toolchain_resolution",
        "--python_path=C:/Users/<user>/AppData/Local/Programs/Python/Python312/python.exe",
        "--action_env=PYTHON_BIN_PATH=C:/Users/<user>/AppData/Local/Programs/Python/Python312/python.exe",
        "--define=litert_webgpu_dawn_dynamic_loader=false"
      ]
    }
  }
}
```

### RTX Spark (native Windows ARM64): `windows/config.json`

`C:/work` holds the checkouts, SDKs, tools and models; `subst` maps it to `L:` for short build paths.

```json
{
  "platform": "rtx_spark",
  "host_mode": "diagnostic",
  "runtime_dir": "C:/work/runtime/fa232ccb4-<cherry-pick commit>",
  "cli": "litert_lm_advanced_main.exe",
  "sdk_bin_dirs": [
    "C:/work/sdk/TensorRT-RTX-1.7.1.116/bin",
    "C:/work/sdk/cuda-13.4/bin",
    "C:/work/sdk/cuda-13.4/bin/arm64"
  ],
  "state_root": "C:/work/bench_state",
  "litert_dir": "C:/work/LiteRT",
  "litert_lm_dir": "C:/work/LiteRT-LM",
  "prompts_dir": null,
  "models": {
    "26b": "C:/work/models/26b_drafter_vision.litertlm"
  },
  "host": { "require_ac": true, "overlay_ac_scheme": "ded574b5-45a0-4f42-8737-46345c09c238" },
  "llama": {
    "bin": "C:/work/llama/llama-b12359-bin-win-cuda-13.5-arm64",
    "release_tag": "v1.0.0",
    "commit": "b1b770d90d94cad9e9ea5b993bdbc383a522fb28",
    "threads": 20,
    "models": {
      "26b": {
        "main": "C:/work/models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "draft": "C:/work/models/gemma-4-26B-A4B-it-qat-GGUF/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
      }
    }
  },
  "build": {
    "bazel": "C:/work/tools/bazel.exe",
    "msvc": {
      "script": "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat",
      "args": ["-arch=arm64", "-host_arch=arm64", "-no_logo"]
    },
    "cuda_home": "L:/sdk/cuda-13.4",
    "tensorrt_rtx_root": "L:/sdk/TensorRT-RTX-1.7.1.116",
    "env": {
      "BAZEL_VC": "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC",
      "BAZEL_VC_FULL_VERSION": "14.44.35207",
      "BAZEL_SH": "L:/tools/portablegit/bin/bash.exe",
      "BAZELISK_HOME": "C:/work/cache/bazelisk"
    },
    "subst": { "drive": "L:", "root": "C:/work" },
    "litert": {
      "startup_args": ["--output_user_root=L:/b/user", "--output_base=L:/b/rt", "--nowindows_enable_symlinks"],
      "args": [
        "--jobs=10", "--local_ram_resources=14000", "--noenable_runfiles",
        "--config=windows_arm64",
        "--repo_env=HERMETIC_PYTHON_VERSION=3.13", "--repo_env=USE_HERMETIC_CC_TOOLCHAIN=0",
        "--repo_env=CUDA_HOME=L:/sdk/cuda-13.4", "--repo_env=TENSORRT_RTX_ROOT=L:/sdk/TensorRT-RTX-1.7.1.116",
        "--action_env=PATH", "--action_env=INCLUDE", "--action_env=LIB",
        "--python_path=C:/work/venv/Scripts/python.exe",
        "--action_env=PYTHON_BIN_PATH=C:/work/venv/Scripts/python.exe"
      ]
    },
    "litert_lm": {
      "startup_args": ["--output_user_root=C:/lrt-build/u", "--output_base=C:/lrt-build/lr", "--windows_enable_symlinks"],
      "args": [
        "--jobs=12", "--local_resources=memory=16000",
        "--override_repository=litert=L:/LiteRT",
        "--//build_config:enable_constrained_decoding=false",
        "--noenable_runfiles", "--config=windows_arm64",
        "--repo_env=HERMETIC_PYTHON_VERSION=3.13", "--repo_env=USE_HERMETIC_CC_TOOLCHAIN=0",
        "--repo_env=CUDA_HOME=L:/sdk/cuda-13.4", "--repo_env=TENSORRT_RTX_ROOT=L:/sdk/TensorRT-RTX-1.7.1.116",
        "--action_env=PATH", "--action_env=INCLUDE", "--action_env=LIB",
        "--python_path=C:/work/venv/Scripts/python.exe",
        "--action_env=PYTHON_BIN_PATH=C:/work/venv/Scripts/python.exe",
        "--define=litert_webgpu_dawn_dynamic_loader=false"
      ]
    }
  }
}
```

### Config notes

| Field | Note |
|---|---|
| `models` | add `e2b`, `e4b`, `12b` the same way to measure those; `llama.models` per profile: `main`, `draft` |
| `prompts_dir` | `null` = the benchmark's `prompts/` (the files the tables used) |
| `host` | `python windows\common.tool check` fails a run when another benchmark runs, or (`require_ac`) off AC power, or (`overlay_ac_scheme`) not on that power overlay (the Spark's Best performance overlay) |
| `host_mode` | recorded in each run (`CI_HOST_MODE` overrides) |
| `build.litert_lm.args` | `build.ps1` adds `--override_repository=litert=<litert_dir>` unless one is given (the Spark gives the `L:` form) |
| `--define=litert_webgpu_dawn_dynamic_loader=false` | required: without it the CLI links every `wgpu` symbol twice (LNK2005) |

## 5. Build

```bash
# Linux
cd LiteRT/litert/vendors/nvidia/benchmark && ./build.sh          # --rt-only (plugins) | --lm-only (CLI)
# -> $LITERT_BENCH_ROOT/cache/build/litert/runtime-libs/  (BUILD_ID fa232ccb4-42531a01)
```

```powershell
# Windows
cd LiteRT\litert\vendors\nvidia\benchmark
powershell -ExecutionPolicy Bypass -File windows\build.ps1        # [-Config FILE] [-RuntimeDir DIR]
# -> runtime_dir with BUILD_ID and build.json. From fresh output bases: ~50 min (RTX 5080), ~33 min (RTX Spark)
```

Build pitfalls met on 2026-11-17:

| Symptom | Fix |
|---|---|
| build id ends in `+` | `core.autocrlf false` before the checkout (the build rewrites `cxxbridge_cmd/Cargo.lock`) |
| a fresh LiteRT-LM output base fails in `build_bazel_rules_android`'s patch | Git's `bin` (bash) on `PATH` |
| LNK1181 on a Rust object (RTX Spark) | a short LiteRT-LM output base (`C:/lrt-build/lr`; paths past 260 characters fail) |
| genrule `generate_build_identifier` fails | a `python3` on `PATH` (a script that execs `python.exe "$@"`, outside the CUDA `bin`) |
| symlink / runfiles errors | Developer Mode and `LongPathsEnabled`, or `--nowindows_enable_symlinks --noenable_runfiles` |

## 6. Benchmark one model at one context length

Workloads (`workloads.json`):

| Workload | Prefill tokens | Decode tokens | Context | Iterations |
|---|---|---|---|---|
| `short` | 1,024 | 256 | 2,048 | 8 |
| `8k` | 8,192 | 256 | 9,216 | 4 |
| `16k` | 16,384 | 256 | 17,408 | 4 |
| `30k` | 30,720 | 256 | 31,744 | 4 |
| `32k` | 32,768 | 256 | 33,792 | 4 |
| `128k` | 128,001 | 128 | 130,048 | 4 |
| `256k` | 256,001 | 128 | 262,144 | 4 |

LiteRT-LM:

```bash
B=LiteRT/litert/vendors/nvidia/benchmark
$B/bench.sh --model 26b --workload 8k --metric latency --out /tmp/runs                                   # Linux
python windows\bench.tool --model 26b --workload 8k --metric latency --out C:\runs                       # Windows
# MTP (acceptance needs real text):
... --workload 8k --metric latency --mtp --prompt-file $B/prompts/prompt_8k.txt --iterations 4 --label 26b_8k_mtp_text
# RTX 5080, 128k / 256k (no MTP; experts partly in host memory):
... --workload 128k --context 131072 --cli-arg=--allow_context_length_expansion=true --env LITERT_NVIDIA_MOE_DEVICE_RESERVE_MB=5120
... --workload 256k --context 262144 --cli-arg=--allow_context_length_expansion=true --env LITERT_NVIDIA_MOE_DEVICE_RESERVE_MB=10240
# RTX Spark, 128k / 256k: the same without the --env (every expert stays on the device); --mtp allowed
```

| Option | Effect |
|---|---|
| `--model e2b\|e4b\|12b\|26b\|FILE` | a profile or a `.litertlm` file |
| `--metric latency\|memory\|cold` | speed; one iteration with GPU memory sampling; one iteration from an empty cache |
| `--context N` | override the context capacity |
| `--mtp`, `--prompt-file FILE` | speculative decoding; real text (`prompts/prompt_1024.txt`, `prompt_8k.txt`, `prompt_16k.txt`, `prompt_30k.txt`, `prompt_128k.txt`) |
| `--iterations N`, `--label NAME`, `--pool DIR` | iterations; run directory name; model cache pool |
| `--cli-arg ARG`, `--env NAME=VALUE` | extra CLI flag / environment variable (repeatable) |
| `--cooldown C` | Windows: start with the GPU at most C °C |

llama.cpp:

```bash
$B/llama/bench.sh --model 26b --workload 8k --metric latency --out /tmp/runs                             # Linux
python llama\bench.tool --model 26b --workload 8k --metric latency --out C:\runs                         # Windows
... --workload 8k --metric latency --mtp --prompt-file $B/prompts/prompt_8k.txt --label 26b_8k_mtp_text  # MTP (llama-server)
```

| Option | Effect |
|---|---|
| `--workload` | `short\|8k\|16k\|30k\|128k` (Windows also `256k`) |
| `--metric latency\|memory\|cold` | llama-bench prefill + decode at depth; one `-pg` pass under GPU sampling; server start to ready |
| `--mtp` | the drafter GGUF (needs `--prompt-file`: llama-server completions) |
| `--sampling greedy\|model`, `--arg ARG` | the completions' sampler; an extra llama-bench/llama-server flag |

Output: `<out>/<label>/` (`meta.json`, `cli.log`, `time.txt`/`time.json`, `gpu.tsv`; llama.cpp: `pp.json`, `tg.json`,
`mem.json` or `client.N.json`).

## 7. The matrix and the tables

Only runs under the machine's `measured/` directory reach the tables; the newest valid run of each cell wins.

```bash
# Linux
B=LiteRT/litert/vendors/nvidia/benchmark
$B/matrix.sh       "$LITERT_BENCH_ROOT/measured/matrix_$(date -u +%Y%m%dT%H%M%S)" 26b   # BENCH_GROUPS="context mtp init verify"
$B/llama/matrix.sh "$LITERT_BENCH_ROOT/measured/llama_$(date -u +%Y%m%dT%H%M%S)"  26b   # LLAMA_GROUPS="context mtp init"
python3 $B/report/tables.tool --platform 5080_wsl --runs "$LITERT_BENCH_ROOT/measured" --out ./tables
```

```powershell
# Windows (RTX Spark: --platform rtx_spark)
python windows\matrix.tool <state_root>\measured\matrix_20261117 26b --llama --cooldown 55
python report\tables.tool --platform 5080 --runs <state_root>\measured\matrix_20261117 <state_root>\measured\matrix_20261117_llama --out C:\tables --sources C:\tables\sources.json
```

| | |
|---|---|
| Models | default `e2b e4b 12b 26b` |
| Groups | `context` (latency and memory at the model's workloads), `mtp` (plain and MTP on real prompts; also 128k/256k where `platforms.json` sets `long_context_mtp`: RTX Spark), `init` (cold and warm start), `verify` |
| Output | `<out>/<platform>/gemma4_moe.md` (`gemma4_dense.md` for E2B/E4B/12B); `--sources` lists the run behind each cell |
| Duration | the 128k/256k cells take 10-25 min each on the RTX Spark |

Conditions for reproducible numbers:
- One GPU job at a time; nothing else running (on the RTX Spark the CPU and GPU share one power budget).
- RTX 5080: Windows power plan High performance; at least 20 GiB free host memory for 128k/256k.
- RTX Spark: AC power, Best performance overlay, idle 10 minutes before timing (it caps its own power after sustained
  load), `--cooldown 55`.
- WSL timings are rough (the clock drifts and the CPU is shared); the official tables come from native Windows.

## 8. Unit tests (Linux)

The 49 Linux `cc_test` targets of `litert/vendors/nvidia` (`windows_compat_test` runs only on Windows). GPU tests are
tagged `manual`: name them explicitly. Use `build.sh`'s Bazel settings:

```bash
cd LiteRT        # the same output base as build.sh, so the libraries are reused
bazel --batch --output_base=$LITERT_BENCH_ROOT/cache/build/litert test -c opt \
  --noincompatible_enable_android_toolchain_resolution \
  --repo_env=CC=/usr/bin/clang --repo_env=CXX=/usr/bin/clang++ --action_env=CC=/usr/bin/clang --action_env=CXX=/usr/bin/clang++ \
  --repo_env=CUDA_HOME=$CUDA_HOME --repo_env=TENSORRT_RTX_ROOT=$TENSORRT_RTX_ROOT \
  --test_output=errors --cache_test_results=no --local_test_jobs=1 \
  --test_env=LD_LIBRARY_PATH=$TENSORRT_RTX_ROOT/lib:$CUDA_HOME/lib64 \
  //litert/vendors/nvidia/trtllm:quant_gemm_test //litert/vendors/nvidia/compiler:moe_plugin_test   # ... the targets to run
```

The targets: the `cc_test` rules of `litert/vendors/nvidia/{,compiler,dispatch,trtllm,tools}/BUILD`, 50 in all, 49 on Linux.
