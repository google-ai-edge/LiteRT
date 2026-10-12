# NVIDIA (TensorRT-RTX) benchmarks of the Gemma 4 models

One measurement is one run of the unchanged LiteRT-LM advanced CLI in benchmark mode on the
NVIDIA backend; the CLI's own benchmark report is the data. These scripts build the pieces,
choose the flags of a workload, keep a run's files together and turn run directories into tables.

| File | Role |
|---|---|
| `env.sh` | paths (the LiteRT tree, LiteRT-LM, the SDK, CUDA, the state directory, the model files) and the TensorRT-RTX environment of a model cache pool; an optional `env.local.sh` next to it sets a machine's paths |
| `build.sh` | builds the compiler plugin, the dispatch library and the LiteRT-LM CLI with Bazel into the state directory |
| `bench.sh` | one measurement: `--model e2b\|e4b\|12b\|26b\|FILE --out DIR [--workload short\|8k\|16k\|30k\|32k\|128k] [--metric latency\|memory\|verify\|cold] [--mtp] [--prompt-file FILE] [--iterations N] [--label NAME] [--cli-arg ARG] [--env NAME=VALUE]` |
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
`~/.local/state/litert-nvidia-benchmark`), `LITERT_BENCH_MODELS` (the litertlm files; default
`$LITERT_LM_DIR/../models`), `LITERT_BENCH_DISTDIR` (a Bazel `--distdir` for offline builds).
Put them in `env.local.sh` (sourced first by `env.sh`) or export them before calling a script.

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
