# NVIDIA TensorRT-RTX measurements

Performance and evaluation measurements of LiteRT-LM on LiteRT's NVIDIA (TensorRT-RTX)
backend against llama.cpp: Gemma 4 E2B, E4B, 12B and 26B-A4B, and Qwen 3.8 27B, on the
RTX 5080 (native Windows, and WSL2 Linux on the same host) and the RTX Spark N1X (native
Windows ARM64). These files live in `litert/vendors/nvidia/benchmark/` beside LiteRT's own
`env.sh`, `build.sh`, `bench.sh`, `matrix.sh`, `report.py` and `prompts/`; in the staging
repository outside that tree, `env.sh` is a shim that forwards to LiteRT's
([The merge into LiteRT](#the-merge-into-litert)).

Two parts, kept apart:

- **Generation** runs the models and writes run directories into a machine's cache. LiteRT's
  own `bench.sh` / `matrix.sh` on Linux, `windows/bench.tool` / `windows/matrix.tool` on native
  Windows; `llama/` for llama.cpp; `eval/` for LiveCodeBench and GPQA. Nothing generated is
  tracked.
- **Processing** turns the cached runs into the tables under `google/`: `report/tables.tool`
  (performance) and `report/eval.tool` (evaluations). It is deterministic over the cache, so
  running it again without new runs reproduces the same files; a new run replaces the cell it
  measures.

| Tracked | Role |
|---|---|
| `models.json`, `workloads.json`, `platforms.json` | the models, the workload presets (bench.sh's), the measured platforms |
| `report/runs.tool`, `report/tables.tool`, `report/eval.tool` | processing |
| `windows/common.tool`, `windows/bench.tool`, `windows/matrix.tool`, `windows/config.example.json` | native Windows generation (x64 and ARM64) |
| `llama/env.sh`, `llama/bench.sh`, `llama/serve.sh`, `llama/matrix.sh`, `llama/client.tool` | llama.cpp generation on Linux |
| `llama/bench.tool`, `llama/matrix.tool` | llama.cpp generation on native Windows |
| `llama/report.tool` | the llama.cpp run parser (LiteRT's `report.py` parses LiteRT runs) |
| `eval/lcb/*`, `eval/gpqa/*` | LiveCodeBench v6 and GPQA Diamond: fetch, select, run, grade |
| `google/<platform>/*.md`, `google/lcb/*.md`, `google/gpqa/*.md` | the generated tables |
| `test_tables.tool` | tests of the processing on synthetic runs (`python3 test_tables.tool`) |
| `env.sh`, `env.local.sh.example` | staging only: the shim, and a machine's Linux paths |

`.tool` files are Python; run them with `python3 file.tool` (`python file.tool` on Windows).

## Platforms

| Id (`platforms.json`) | Machine | Generation | Host memory counter |
|---|---|---|---|
| `5080` | RTX 5080, native Windows x64 (cuda-wsl's host) | `windows/*.tool` | peak working set |
| `5080_wsl` | RTX 5080, WSL2 Linux on the same host | `bench.sh`, `llama/*.sh` | GNU time's peak RSS |
| `rtx_spark` | RTX Spark N1X, native Windows ARM64 | `windows/*.tool` | peak working set |

## Setup

**Linux (WSL).** LiteRT's `benchmark/env.local.sh` names the machine's paths (`env.local.sh.example`
lists them, the `LLAMA_*` ones included). Build with LiteRT's `build.sh`; the llama.cpp
binaries come from a checkout at `LLAMA_CPP_DIR`.

**Windows.** Copy `windows/config.example.json` to `windows/config.json` (not tracked) and set
the runtime directory (the CLI and the two NVIDIA plugin libraries, `.dll` or `.so` names; a
`BUILD_ID` file there names the build, else the binaries' digest does), the SDK `bin`
directories (TensorRT-RTX, the CUDA runtime), the state root, the checkouts, the model files
and the llama.cpp release. `python windows/common.tool check` prints the host's state and fails
when another benchmark runs (or, with `host.require_ac`, off AC power). The machine builds its
binaries with its own scripts for now ([Open items](#open-items)).

Host preparation stays outside the runners: on the 5080 the owner's `benchmark` shell
preparation (High Performance, ASUS Profile 1, the cleanup allowlist) before a formal batch;
on the RTX Spark, AC power and the Best performance overlay, nothing else running (the CPU
and the GPU share one power budget, so a download or a build lowers GPU clocks), and
`--cooldown 55` so every run starts at most at 55 °C. The runners record the host's state in
each run's `meta.json`; `host_mode` (environment `CI_HOST_MODE`) says what was prepared.

## Generate

One matrix invocation writes one batch directory under the machine's measured-run cache:
`$LITERT_BENCH_ROOT/measured/<batch>` on Linux, `<state_root>/measured/<batch>` on Windows.
Diagnostic runs go elsewhere (`--out` is free); only what is under `measured/` reaches the
tables.

```sh
# Linux: LiteRT, then llama.cpp (one GPU job at a time)
"$BENCH/matrix.sh" "$LITERT_BENCH_ROOT/measured/matrix_$(date -u +%Y%m%dT%H%M%S)" e2b e4b 12b 26b
llama/matrix.sh    "$LITERT_BENCH_ROOT/measured/llama_$(date -u +%Y%m%dT%H%M%S)" e2b e4b 12b 26b
```

```powershell
# Windows: LiteRT and llama.cpp
python windows/matrix.tool C:\state\measured\matrix_20261108 e2b e4b 12b 26b --llama --cooldown 55
```

A single measurement is `bench.sh` / `windows/bench.tool --model M --workload W --metric
latency|memory|verify|cold [--mtp] [--prompt-file F]`, the same flags on both. The cold
metric starts from an empty cache pool; the other metrics reuse the pool of the model and
build, and a pool without compiled engines is prepared by a separate one-iteration run that
is never reported (`prepare.*` in the run directory).

A run directory holds `meta.json` (the workload, the command, the build ids, the host),
`cli.log` (the CLI's output; its benchmark report is the data), `time.txt` (GNU time) or
`time.json` (wall time and the process's peak working set and commit), and `gpu.tsv` (the
device's memory in use, sampled every 100 ms, for the memory and cold metrics). llama.cpp runs
hold llama-bench's JSON (`pp.json`, `tg.json`, `mem.json`) or the server's completions
(`client.N.json`) and start-to-ready times (`server/serve.json`, `cold/`, `warm/`).

## Process

```sh
python3 report/tables.tool --platform 5080_wsl --runs "$LITERT_BENCH_ROOT/measured"
python3 report/eval.tool --platform 5080_wsl --lcb "$LITERT_BENCH_ROOT/eval/lcb/runs" --gpqa "$LITERT_BENCH_ROOT/eval/gpqa/runs"
```

(`--bench-dir` names LiteRT's benchmark directory when `report.py` is not beside this one.)
`tables.tool` writes `google/<platform>/gemma4_dense.md`, `gemma4_moe.md` and `qwen3_8_27b.md`
for the model families that have runs; `eval.tool` writes `google/lcb/<platform>.md` and
`google/gpqa/<platform>.md`. The files hold a generated provenance line (platform, build ids,
dates, run count) and the tables, nothing else; everything a reader needs beyond the numbers
is in this file.

**Selection.** Every run under the given roots is read. A run is valid when its process
exited 0, it ran every iteration its `meta.json` asks for, and (LiteRT) its log shows the
NVIDIA dispatch executing a fully delegated graph (the CLI can exit 0 after falling back to
the CPU). Each table cell shows the newest valid run of its kind, by start time; older runs
stay in the cache and are not shown. The kinds, from a run's meta: synthetic latency, memory
(with the model file mapped, and without mmap for llama.cpp's host-memory column), real-text
plain and MTP latency, cold start, answer check. `--sources FILE` writes the run behind every
cell.

## What the tables show

**Workloads** (`workloads.json`, bench.sh's presets): 1k = 1,024 synthetic prefill tokens and
256 decode tokens at 2,048 context capacity; 8k = 8,192 / 256 / 9,216; 16k = 16,384 / 256 /
17,408; 30k = 30,720 / 256 / 31,744; 128k = 128,001 / 128 / 130,048; 256k = 256,001 / 128 /
262,144. The 12B and Qwen 3.8 27B add 128k; the 26B's 128k and 256k are real-text runs with
context expansion of the 32k export and a 5,120 / 10,240 MiB device reserve for the experts.
llama-bench sizes its own capacity per phase (prompt + generation + depth, rounded up), so
the memory comparison includes different KV capacities.

**Decode and prefill speed** are the CLI's own rates (`Decode Speed`, `Prefill Speed`) over the
measured iterations: 8 iterations with 2 discarded at 1k, 4 with 1 discarded above, because
TensorRT-RTX captures an engine's CUDA graph at its second execution and a 1k prompt is one
prefill chunk per iteration. llama.cpp's columns are llama-bench's means (`-p N -n 0` for
prefill, `-p 0 -n T -d N` for decode at that depth; Windows runs that declare warm-up
repetitions discard them). The speedup is TensorRT-RTX over llama.cpp.

**MTP** rows use real text (the prompt files of LiteRT's benchmark, about 1k, 8k, 16k, 30k and
128k tokens of documentation ending in a question) with stop tokens honored: three
conversations with one discarded for plain decoding, four with two discarded under speculative
decoding (the decode engine's capture falls in the second conversation). Acceptance is
accepted drafted tokens over drafted tokens; it measures agreement, not correctness. E2B and
E4B sample greedily; 12B, 26B and Qwen use top-k 64, top-p 0.95, temperature 1.0 (their
files' policy); llama-server's completions use the same sampler with seeds 1, 2, ... in the
model's conversation template. The "actual input tokens" column is the tokenized prompt, which
differs between runtimes and from the nominal label.

**Memory** is the device-wide `nvidia-smi memory.used` peak of a dedicated one-iteration run,
desktop allocations included, and the process's peak host memory: GNU time's peak RSS on
Linux (file mappings counted), the peak working set on Windows. llama.cpp's host figure comes
from a run without mmap (`-lm none`); its GPU peak from the mapped run where there is one. The
26B's llama.cpp runs use `-lm none` and 512-token physical batches; the dense models 1,024.
Long-context 26B runs on TensorRT-RTX place expert weights in mapped host memory.

**Start-up** comes from the cold run (a fresh AOT, runtime and CUDA cache pool: the compiler
phase, the engine builds, the executor's init and the whole process) and from the 1k
synthetic latency process as the warm start (its whole-process time includes its
iterations). Model files are read into the OS file cache before a model's group, so neither
is a cold disk read. **Answer check**: the CLI's Paris question on the NPU and on the CPU.

**llama.cpp settings** everywhere: `-fa on -ngl 99 -t 20 -b 2048`, `-ncmoe 0` for the 26B,
GGUFs with the matching MTP drafter GGUFs (`--spec-type draft-mtp --spec-draft-n-max 3 -ngld
99`). The GGUF and the LiteRT export of a model are different quantizations; speed ratios
do not establish equal accuracy.

**Platform notes.** On native Windows the figures include the desktop's GPU use; the 5080's
Smart App Control blocked freshly built unsigned DLLs until it was turned off (2026-11-03);
"CUDA - Sysmem Fallback Policy" is at its driver default (the runner marks a llama.cpp run
whose process sits within 1 GiB of the device with 0.5 GiB shared). The RTX Spark is a laptop
whose GPU is power-capped under sustained load (clocks fall from about 2.4 to 1.65–1.8 GHz),
so sustained rates sit below first-iteration ones; one LiteRT process launched from an SSH
session there failed once with `0xC0000142` and succeeded on repeat. The WSL tables
(`5080_wsl`) were measured on a host whose monotonic clock ran 1–5% fast between 2026-10-20
and 2026-10-28 (the kernel tick above 10,000 µs): their tokens/s read low by that much.
`clock_check.tool` (`python3 clock_check.tool 3` prints the state) records the host's clocks with
every Linux run and refuses an unfit one for a formal run; the Windows runners do not call it.

## Evaluate

**LiveCodeBench v6** (`eval/lcb`): `fetch.sh` downloads release v6's 175 problems, `select.tool
--all` writes `selection.json` (tracked), the prompts and the decoded tests; `run.sh MODEL
plain|mtp [MAX_TOKENS] [LABEL]` generates one answer per problem through LiteRT-LM on
TensorRT-RTX (`run_llama.sh` through llama-server), `grade.tool RUN_DIR` runs each answer's
tests the way LiveCodeBench does and writes `grades.json`. Runs live under
`$EVAL_RUNS` (default `$LITERT_BENCH_ROOT/eval/lcb/runs`). One question, predictably:

```sh
eval/lcb/question.sh 26b plain 139          # the 139th problem of selection.json
eval/lcb/question.sh 26b mtp abc397_a 30720 # by id, with a 30,720-token output limit
```

`report/eval.tool` shows, per model, runtime, mode, output limit and sampler, the newest run
that answered every selected problem: passed / attempted (a problem passes when every test
passes), answers without code, decode speed, output length and draft acceptance, and the
per-problem verdicts.

**GPQA Diamond** (`eval/gpqa`): `fetch.sh` downloads the authors' release (keep the questions
out of logs and commits); `run.tool --model qwen27b --label L [--count 10 | --question ID|N]`
runs each question through the platform's bench runner by Artificial Analysis's published
method (its four-option prompt, its answer regexes with the last match taken, zero-shot
pass@1, temperature 0.6 with top-k 20 / top-p 0.95 and a fixed seed per question, up to
32,768 output tokens), scoring against the dataset's answer key. A 32k reasoning answer takes
about 40 minutes at 13 tokens/s; LiteRT-LM's default 10-minute session wait must be raised for
such runs.

## The merge into LiteRT

The tree maps onto `litert/vendors/nvidia/benchmark/` as it is: `report/`, `windows/`,
`llama/`, `eval/`, `google/`, the three JSON registries and this README join LiteRT's
`env.sh`, `build.sh`, `bench.sh`, `matrix.sh`, `report.py` and `prompts/`. `env.sh` here is
not copied (LiteRT's is the real one), nor is `env.local.sh.example` beyond what LiteRT's
already lists. Ported Python carries the `.tool` suffix; nothing imports it by module name
(the files load each other with `importlib`'s `SourceFileLoader`).

## History

The earlier form of this repository kept processed snapshots, validations, diagnostics and
per-platform runners; they are in its history, last at commit `24fee92`: the WSL record
(`bench/`, `MEASUREMENTS.md`), the Windows 5080 matrix and its diagnostics (`windows_5080/`),
the RTX Spark harness, Qwen report and nvguard check (`rtx_spark/`), the minimal LiteRT-LM
study (`lm_minimal_20261102/`), the LiveCodeBench provenance (`eval/lcb/`), and the importer
that moved the published runs into the caches (`tools/import_legacy.py`). The caches hold
those runs: `~/.local/state/litert-nvidia-benchmark-wl/measured` and `eval/` on cuda-wsl,
`C:\Users\lijin\dev\windows_5080\measured` on the 5080, and
`work\rtx_spark\measured` and `work\rtx_spark\eval` on the RTX Spark.

## Open items

- The Windows builds still run through each machine's own scripts (the 5080's
  `build_litert_rtx.ps1`, the Spark's `build-rt.ps1` / `build-lm.ps1`, in the history above); a
  `windows/build.ps1` with the architecture in `config.json` would complete the Windows side.
- The RTX Spark's mobile-QAT llama.cpp comparison and 31B attempt, and the 5080's mobile-QAT
  comparison, are not part of the tables; their records are in the history.
