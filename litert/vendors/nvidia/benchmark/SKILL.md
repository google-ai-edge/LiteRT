---
name: litert-nvidia-benchmark
description: Build and measure Gemma 4 through LiteRT TensorRT-RTX, run its bounded CI suite, and inspect performance and cache evidence.
---

# NVIDIA (TensorRT-RTX) benchmarks of the Gemma 4 models

One measurement is one run of the unchanged LiteRT-LM advanced CLI in benchmark mode on the
NVIDIA backend; the CLI's own benchmark report is the data. These scripts build the pieces,
choose the flags of a workload, keep a run's files together and turn run directories into tables.

| File | Role |
|---|---|
| `env.sh` | paths (the LiteRT tree, LiteRT-LM, the SDK, CUDA, the state directory, the model files) and the TensorRT-RTX environment of a model cache pool; an optional `env.local.sh` next to it sets a machine's paths |
| `build.sh` | builds the compiler plugin, the dispatch library and the LiteRT-LM CLI with Bazel into the state directory |
| `bench.sh` | one measurement: `--model e2b\|e4b\|12b\|26b\|FILE --out DIR [--workload short\|8k\|16k\|30k\|32k\|128k] [--metric latency\|memory\|verify\|cold] [--mtp] [--prompt-file FILE] [--iterations N] [--label NAME] [--cli-arg ARG] [--env NAME=VALUE]` |
| `ci.sh` | `ci.sh OUT_DIR`: 12B cold 128k, 12B hot 32k, 26B cold MTP 32k; builds once, validates and prints results |
| `matrix.sh` | the data points of the record: `matrix.sh OUT_DIR [MODEL...]`, `BENCH_GROUPS="context mtp init verify"` selects the groups |
| `report.py` | tables from run directories: `python3 report.py [--json] RUN_DIR...` |
| `prompts/` | the real prompts of the MTP measurements: about 1k, 8k, 16k, 30k and 128k tokens of documentation ending in a question |

## Setup

```
cp env.local.sh.example env.local.sh   # or export the variables below
```

`env.sh` reads, all optional: `LITERT_RT_DIR` (this tree; default the tree the script is in),
`LITERT_LM_DIR` (a LiteRT-LM checkout; default `../../llm/LiteRT-LM` next to the tree),
`TENSORRT_RTX_ROOT` (the SDK; default `~/opt/tensorrt-rtx-sdk`), `CUDA_HOME`
(default `/usr/local/cuda`), `LITERT_BENCH_ROOT` (builds, caches and runs; default
`~/.local/state/litert-nvidia-benchmark-wl`), `LITERT_BENCH_MODELS` (the litertlm files; default
`$LITERT_LM_DIR/../models`), `LITERT_BENCH_DISTDIR` (a Bazel `--distdir` for offline builds).
Put them in `env.local.sh` (sourced first by `env.sh`) or export them before calling a script.

## Bounded CI

Run from this directory after configuring `env.local.sh`:

```sh
./ci.sh "${LITERT_BENCH_ROOT:-$HOME/.local/state/litert-nvidia-benchmark-wl}/runs/ci_$(date -u +%Y%m%dT%H%M%SZ)"
```

Use a fresh output directory. `ci.sh` calls the existing build and benchmark
scripts and prints shell-escaped commands, step elapsed seconds, the report,
and `rg` excerpts of initialization and cache events. It needs Bash, Python 3,
Bazel, GNU time, flock, rg and nvidia-smi, plus the configured models and SDK.
It does not require the measurements repository or modify its reports.

The three checks are 12B cold at 128,001 prompt tokens; 12B hot at 30,720
prompt tokens with three iterations; and 26B cold with MTP on the fixed
`prompts/prompt_30k.txt` (about 30k actual tokens). The historical “32k” row
uses workload `30k`, with 31,744 context capacity. The separate `32k` preset
means 32,768 synthetic prompt tokens. The 128k check decodes 128 tokens;
the others request 256. Real-text runs honor stop tokens.

Cold uses a new artifact/runtime/CUDA pool. Model bytes are deliberately
pre-read into the OS page cache. Hot has an explicit preparation invocation
for the same workload and pool, followed by the measured invocation. CI
requires cold compilation and hot AOT hits with no compilation, complete
iterations, populated prefill and decode, total initialization, and MTP
statistics where applicable. There are no timing regression thresholds yet.

Host preparation is external: on cuda-wsl use `zsh -lic benchmark` before a
formal measurement and set `CI_HOST_MODE=formal`. Without formal preparation,
leave the default `CI_HOST_MODE=diagnostic` and label timings accordingly.
The runners refuse competing CUDA compute processes. Do not drop global OS
caches or stop unrelated applications. Model-file cache, compiled artifacts,
process warm-up and prompt/KV reuse are distinct states; “hot” describes the
preparation above, not every possible cache combination.

`report.py` prints preparation and measured processes as separate rows, with
whole-process wall time, total initialization, executor time, compiler and
engine generation time, AOT hits/misses, compiled partitions, prefill/decode,
MTP acceptance and memory when sampled. A dash means unavailable; compile and
engine durations overlap initialization and must not be added to it.

## Measuring

```
./build.sh                                     # once per change of the trees (minutes)
./bench.sh --model e4b --out RUNS              # decode and prefill at 2k, the default workload
./bench.sh --model e4b --workload 8k --metric latency --mtp \
    --prompt-file prompts/prompt_8k.txt --iterations 3 --out RUNS   # MTP on real text
./bench.sh --model 26b --metric verify --out RUNS                   # the Paris answer check
python3 report.py RUNS                         # the table of every run under RUNS
BENCH_GROUPS="context mtp" ./matrix.sh RUNS e4b 12b   # the record's data points for two models
```

Each run directory holds the CLI's log and benchmark report (`cli.log`), GNU time's accounting
(`time.txt`), the nvidia-smi samples of a memory run (`gpu.tsv`) and the run's settings
(`meta.json`). Compiled engines, runtime caches and the CUDA JIT cache live in a pool under
`$LITERT_BENCH_ROOT/cache/models/<model>/<build id>`, the build id being the two trees' commits
(a `+` marks local changes), so a changed tree compiles afresh and an unchanged one starts warm.
The TensorRT-RTX settings of a pool (`nb_runtime_env` in `env.sh`: the Gemma 4 partition policy,
BF16 activations, the CUDA GEMV plugins, the shared weight store) are the same for every model;
`--env NAME=VALUE` overrides one for an experiment, in a pool of its own (`--pool DIR`).

Use one `LITERT_BENCH_ROOT` for the standalone scripts and any measurement or evaluation
wrapper. Source/build identities already separate model pools; a branch does not need a
separate benchmark root. Put experimental pools under `$LITERT_BENCH_ROOT/cache/models/`
with `--pool`, and use `--metric cold` for a new pool under `cache/scratch/`.

Within the root, `cache/build/` holds Bazel outputs and the runnable binaries;
`cache/models/<model>/<build id>/` holds `aot/`, `runtime/`, `compiler/cuda/`, and `cpu/`
caches. `runs/` holds measurement evidence. Retired model pools and completed scratch
pools can be removed without deleting the corresponding runs; keep pools and build
outputs needed for current work. Model downloads stay in their configured model directories.
