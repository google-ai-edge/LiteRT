# NVIDIA model reports

`run_head.sh` builds the NVIDIA compiler/dispatch libraries and the existing,
unchanged LiteRT-LM advanced CLI. It supports Gemma E2B and 12B with one entry.
No input JSON, extra executable, model download, or Git branch movement is needed.

## Setup and commands

Use Linux, Bash, Python 3, GNU time, flock, Bazel, Clang, CUDA 12.9 and a compatible
TensorRT-RTX SDK (tested with 1.5.0). Install dependencies using the repositories'
normal build instructions. Set checkout/SDK locations once:

```bash
export LITERT_LM_DIR=/home/lijin/odml/llm/LiteRT-LM
export TENSORRT_RTX_ROOT=/home/lijin/opt/tensorrt-rtx-sdk
export CUDA_HOME=/usr/local/cuda
export LITERT_BENCH_ROOT=/home/lijin/.local/state/litert-nvidia-benchmark
cd /home/lijin/odml/rt_g3/LiteRT

./litert/vendors/nvidia/run_head.sh report --profile e2b
./litert/vendors/nvidia/run_head.sh report --profile 12b
./litert/vendors/nvidia/run_head.sh report --profile 12b --workload 32k --prefill 128
./litert/vendors/nvidia/run_head.sh report --profile 12b --workload 32k --prefill 1024
./litert/vendors/nvidia/run_head.sh report --profile 12b --workload 128k
./litert/vendors/nvidia/run_head.sh report --profile 12b --workload 128k --prefill 1024
./litert/vendors/nvidia/run_head.sh report --profile 12b --prefill 128 --mtp
./litert/vendors/nvidia/run_head.sh report --profile 12b --metrics latency,memory --cache-state all

./litert/vendors/nvidia/run_head.sh cache list
./litert/vendors/nvidia/run_head.sh cache clear --profile 12b --kind runtime
./litert/vendors/nvidia/run_head.sh cache clear --profile 12b --kind runtime --yes
./litert/vendors/nvidia/run_head.sh cache clear --all       # preview
./litert/vendors/nvidia/run_head.sh cache clear --all --yes # apply
```

LiteRT is always the checkout containing the script and is explicitly passed as
LiteRT-LM's Bazel repository override. By default LiteRT-LM is inferred as
`../../llm/LiteRT-LM` relative to LiteRT. Models are inferred under the sibling
`models/gemma-4-{E2B,12B}-it-litert-lm/` directories, with the corresponding
`gemma-4-{E2B,12B}-it.litertlm` filename. Use `--model-file PATH` to override.
Legacy JSON, `RUN_ROOT`, `G4MODEL`, `LITERT_G3_HEAD`, `LITERT_LM_G3_HEAD`, and inherited
NVIDIA tuning variables do not select these paths/settings.

## What is measured

`--metrics` accepts `verify`, `latency`, `memory`, comma-separated subsets, or `all`
(the default). Every invocation builds incrementally and records build wall time,
GNU time maximum RSS, binary sizes/hashes and paths. Clearing `--kind build`
forces the next build to start from empty outputs. Remote/disk action caches are
disabled; dependency archives and the OS page cache may remain warm. No redundant
binary copy or stripped-size experiment is made.

| Preset | Synthetic input | Decode steps | Capacity | Prefill batch |
|---|---:|---:|---:|---:|
| E2B or 12B short | 1,024 | 256 | 2,048 | 1,024 or 128 |
| 12B 32k | 32,768 | 256 | 34,818 | 128 |
| 12B 32k | 32,768 | 256 | 33,792 | 1,024 |
| 12B 128k | 128,001 | 128 | 130,816 | 128 |
| 12B 128k | 128,001 | 128 | 130,048 | 1,024 |

Short throughput uses eight iterations, excluding two warmups. Long throughput
uses one warmup and three measured iterations; 32k repeats in two processes.
Native counts and every iteration are saved. Final native all-iteration aggregates
are excluded from the wrapper's warmup-filtered statistics. Long inputs use the
CLI's synthetic resizing, so they establish new baselines rather than replaying
the old exact-token-ID fixtures. Decode counts are native runtime steps, not a
trace of delivered IDs. `estimated_ttft_seconds` is prefill plus average decode
per token; it does not measure first delivery or the MTP first-token stall.

Verification uses a real capital-city prompt, synthetic overrides disabled,
and the CLI's expected-answer check on CPU and NVIDIA. Both responses are saved.
This is an answer smoke test, not logits-level numerical equivalence.
The default E2B baseline uses no signature selection. Explicit E2B `--prefill 128`
selects `prefill_128,decode`: the static executor ignores the batch hint when
both prefill graphs are available. 12B selects prefill plus decode; `--mtp` also
selects verify and enables speculation. An independent real-generation probe
requires runtime log evidence that drafting or speculative cycles actually ran;
merely accepting the flag is not success. Unsupported current model/backend combinations fail without fallback.
The script does not fetch an experimental PR to enable MTP.

At the tested LiteRT-LM revision `2f8284d5ea323278989facfaaed1c476b174760a`,
the 12B NPU MTP probe fails with `Unsupported backend: 6` in the native drafter.
Non-MTP tests pass; speculative performance is unavailable at that revision.

`--residency auto` uses resident engines for short and for the non-MTP long
presets; MTP long presets use lazy engines. This is a fixed preset policy, not
available-VRAM detection. Explicit `lazy` and `resident` remain available.
Resident mode avoids prefill/decode engine eviction and reload costs but keeps
both engines' weight allocations in GPU memory.
On the tested 16 GB RTX 5080, 32k/prefill 128 resident reached 78.08 decode steps/s
with 13.51 GiB sampled GPU peak (13.80 GiB at backend checkpoints). 128k/prefill
128 reached 75.2 decode steps/s with 14.69 GiB resident, against 41.0 decode
steps/s with 9.69 GiB lazy, and 128k/prefill 1024 resident peaked at 14.96 GiB.
These were diagnostic background-activity measurements; hosts with less free
GPU memory can select `lazy`.
Precision is BF16, GEMV is CUDA, shared weights are enabled, and 12B retains the
512 MiB FC cap. Shared weights do not eliminate per-engine GPU weight copies.

Memory runs are separate invocations with one native synthetic iteration. CPU
maximum RSS includes initialization; build RSS is the largest child observed by
GNU time, not the sum of concurrent workers. A 100 ms requested sampler interval
records runtime RSS and device-wide GPU usage; subprocess/query overhead increases
the actual spacing. Reports retain the baseline, samples, observed peak and CUDA
backend checkpoints. GPU memory includes desktop/background allocations and may
miss short peaks. No exact engine/session/prefill/decode phase attribution or
baseline-subtracted process-allocation claim is made. Existing compiler begin/end
events provide compiler wall intervals when present; process wall time and TensorRT
engine-build events are recorded separately. Missing metrics fail the case.

## Cache states and storage

`--cache-state warm` (default) prepares compatible AOT/runtime caches on a miss.
`runtime-cold` reuses/prepares AOT but measures with empty runtime/compiler caches.
`cold` gives every measured pass its own empty generated-cache directories.
`all` runs warm, runtime-cold and cold sequentially. In-process warmup and on-disk
cache state are distinct. Cold verification, throughput and memory each require
separate AOT compilation when all metrics are selected; the script announces this.
Use `--metrics latency,memory --cache-state all` for a cache/memory study after an
ordinary correctness run. Compatible warm invocations do not compile new AOT.

```text
$LITERT_BENCH_ROOT/
  cache/build/{litert,litert-lm}/
  cache/models/<profile>/<compatibility-key>/{aot,runtime,compiler,cpu}/
  cache/scratch/<run>/<case>/{aot,runtime,compiler,cpu}/
  reports/<run>/{report.md,results.json,logs/}
  reports/cleanup-<timestamp>.json
```

The default root is `~/.local/state/litert-nvidia-benchmark`. Resolved paths are
printed before expensive work; actual commands, settings, source fingerprints,
model/binary hashes, SDK/device identity, cache observations and status are in
`results.json`. Inspect sizes with `ls -lh` and `du -sh`. Cache compatibility uses
actual model/binary hashes and backend/workload/device/SDK identity, not just Git
HEAD. Reports remain readable after cleanup, but cleared artifact paths in them
will no longer exist. A new invocation rebuilds those artifacts.

Cleanup requires an owned root, excludes active runs, previews by default,
rejects symlinks in model cache ownership/data paths and uses Linux's mount table
to refuse nested mounts, including bind mounts on the same filesystem.
Bazel's legitimate output symlinks are unlinked without following their targets.
Model filters do not select shared build outputs; use `--kind build` or `--all`.
Sources, models, reports and unowned legacy folders are preserved. The one-time
legacy migration cleanup is not a feature of this script.

## Host preparation and build limits

On cuda-wsl (hostname `Cuda`), the script runs `zsh -lic benchmark`, retaining its
log. Protected applications remain untouched. Other hosts receive a GPU-process
and thermal readiness check and should use their documented local performance
preparation. A compute process or thermal contention blocks work.

`LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY` defaults to `1`: a strict idle failure
caused by background activity prints a warning and labels the report
`diagnostic_background_activity_exception`. Power-state transitions and nonzero
memory-controller utilization do not block this exception. The preparation log
and observed GPU state are retained; these are not clean-preflight benchmarks.
Set the variable to `0` to require the strict idle check. This replaces the old
`LITERT_BENCH_ALLOW_BACKGROUND_IDLE` variable.

Unknown CUDA processes, GPU utilization above 5%, temperatures at or above 70 C,
and failures to prepare High Performance/ASUS Profile 1 still block execution.
The exception applies only to the installed command's idle-window failure.

Optional build-resource variables: `LITERT_BENCH_JOBS` (8),
`LITERT_BENCH_BUILD_RAM_MB` (8192), `LITERT_BENCH_DISTDIR` (existing dependency
archives), and `LITERT_BENCH_MEMORY_LIMIT` (e.g. `25G`, using systemd user scopes
with an 8G swap cap; defaults to 25G on cuda-wsl). No runtime deadline is imposed.

The old action names and build/AOT/memory-state switches are retired. Use `report`
with the three metric names and one `--cache-state`. Run focused tests with
`python3 litert/vendors/nvidia/run_head_test.py` on Linux.
