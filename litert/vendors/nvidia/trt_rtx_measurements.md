# Permanent-worktree verification measurements, 2026-09-05

These measurements use the September 5 upstream bases. The September 6 history
cleanup and newer upstream bases are recorded in `trt_rtx_migration.md`; this
report does not claim a new benchmark after that cleanup.

The migration builds and runs successfully. Same-day paired throughput shows
no steady-state regression, and warm CPU memory is effectively unchanged.
Cold-start latency is more variable; the cold measurements below are not a
blanket claim of initialization-latency equivalence.

## Build and functional evidence

Both NVIDIA shared libraries and both LiteRT-LM executables were rebuilt with
`-c opt` and Clang 18, using the new LiteRT-LM worktree's explicit
`--override_repository=litert=/home/lijin/odml/rt_g3/LiteRT_trt_rtx`.
The plugin build took 196.7 seconds and the LM build 352.9 seconds.

Eleven LiteRT test targets passed: bytecode, profiling, AOT artifact integrity,
compiler plugin, sub-byte GEMV plugin, graph builder, CUDA head kernels,
composite inlining, model IR, C++ compiled model, and dispatch delegate kernel.
Both LM targets passed: signature selection and executor settings. These
include CPU-reference GPU kernel checks for the 12B shapes, including the full
262144-by-3840 vocabulary projection.

The 12B CPU reference and both fresh-AOT and reused-AOT generation returned
`Paris`. `run_head.sh verify-aot` passed its cache miss/hit, no-recompilation,
read-only artifact, unchanged identity, refitting, and expected-output checks.
The 12B in-memory JIT path completed a 1024-token prefill and 256-token decode.
E2B, with all four signatures retained, returned `PARIS` on both CPU and NPU.
The E2B run actually exercised the older external W2 head, confirming it is
still needed for that model.

After retiring the old working copies and restoring both original main
checkouts, the permanent worktrees again returned `Paris` on NPU and CPU.
The NPU log confirms an AOT hit without recompilation. This final smoke test
used only the permanent source/library paths; its logs are in
`post_cleanup/logs/` and `build/logs/post_cleanup_driver.log`.

These are focused tests and prompt-output agreement, not an exhaustive model
quality evaluation. All build, test, and failed/setup-attempt logs are retained.

## Fixed configuration

- RTX 5080, driver 581.95, TensorRT-RTX 1.5.0.114, CUDA 12.9.
- Model: `gemma-4-12B-it.litertlm`, 6,883,278,368 bytes; SHA-256
  `58fd31b778ca2c21c80d634fb34fc5a89d11d563a38dfd3cbf1b40dbf252a8b6`.
- Selected roots `prefill_1024,decode`; inactive roots `prefill_128,verify`.
  The log confirms 957 retained reachable subgraphs and four TRT partitions.
- BF16 activations, `cuda_gemv`, shared weights enabled, 512 MiB FC weight cap,
  2048-token context, 1024-token prefill, and 256-token decode.
- Each formal run passed the host's `benchmark` preflight: High Performance,
  original temperature-responsive ASUS Profile 1, idle GPU, 761 MiB device
  baseline. Protected applications/services were not manipulated.
- Memory-intensive runs used a user-systemd scope with `MemoryMax=22G` and
  `MemorySwapMax=8G`. Cgroup-accounted memory and whole-process RSS are not
  identical, particularly for shared/file-backed pages.

## Uninstrumented throughput

Each run has eight iterations; the following means discard the first two.
Memory/dispatch/layer profiling is disabled. The final pair used the same
AOT artifacts and populated runtime caches, with an explicit sequential read
of those artifact files before the GPU-idle preflight. This is scoped input
priming, not a global filesystem-cache drop.

```text
Run                             Prefill tok/s   Decode tok/s   CPU HWM MiB
------------------------------  -------------   ------------   -----------
Retained implementation, paired       3,792.72          65.345       6,482.11
Permanent worktrees, paired           3,803.63          66.115       6,482.79
Change                                +0.29%          +1.18%          +0.69

Permanent worktrees, warm repeat      3,800.64          66.372       6,486.27
Permanent worktrees, first run        3,784.69          64.250      23,133.38
```

The first run unexpectedly compiled once because it explicitly set
`JIT_HANDLE=0`, while `verify-aot` had unset that variable. Those values have
different cache identities. Its high process HWM includes compilation, and
one decode iteration was slower. It is retained, not presented as an AOT-hit
initialization measurement. The paired rows are the primary throughput check;
their small positive differences are not claimed as a new optimization.

All four runtime-cache files from the paired old/new runs are byte-identical.
Their SHA-256 values are retained in `build/logs/paired_runtime_cache_sha256.txt`.
All four independently compiled AOT artifacts also passed a byte-by-byte
comparison (11,358,228,140 bytes total), with SHA-256 recorded in
`build/logs/independent_artifact_sha256.txt`.
The old plugins were checked against the saved pre-migration runtime copies;
both hashes matched. Binary hashes are in `build/logs/binary_hashes.txt`.

## Granular memory and startup

All memory values below are MiB. The first four rows use the new worktrees.
The AOT-hit runs use a fresh LiteRT compiler-cache directory, so they exercise
the NVIDIA AOT manifest path rather than a preexisting LiteRT cache shortcut.

```text
Path                        CPU HWM   Compile/pack s   Init s   After unmap RSS   Decode median RSS   Max swap
-------------------------  ---------  --------------  -------  ----------------  -----------------  --------
Fresh AOT build             21,996.7          379.0      413.6          10,556.7           10,649.4       135.9
AOT hit, TRT cache cold      8,040.5       not run       82.0           2,318.3            2,409.9         0.0
AOT hit, both caches warm    6,478.3       not run       14.5             787.2              878.6         0.0
Fresh in-memory JIT         23,458.5          416.0      437.5       not applicable        15,790.6     5,793.3

TRT cache cold, files primed 7,943.0       not run       27.1           2,252.6            2,344.6         0.0
Retained old fresh AOT      23,442.8          301.5      335.7           see raw trace      see raw trace   0.0
Fresh AOT, input primed     23,981.0          378.6      416.9          12,576.6           12,668.9       129.6
```

`Init s` means process launch through the first dispatch invocation, including
any compilation. `Compile/pack s` is a subset of that interval; it includes
persistence for AOT and packing/handle preparation for JIT. These times must
not be added together. `After unmap` is the checkpoint immediately after the
last AOT artifact is unmapped, not the lifetime high-water mark. Decode RSS is
the median after the first large decode partition invocation through the end
of the benchmark, not an assumption that every decode token was instrumented.
JIT has no AOT mapping to unmap. Every listed run completed successfully.

All runs observed the same 13,716 MiB device-wide NVIDIA peak, against the
761 MiB baseline. This is not a precise per-process or per-component VRAM
allocation measurement. RSS and swap maxima occur independently and must not
be added as if they were simultaneous memory requirements.

The warm memory result reproduces the prior report's approximately 6476.5 MiB
HWM and 877.4 MiB steady decode RSS. No new memory optimization was introduced
by the migration. The lower HWM of the first cold AOT run must not be credited
as an optimization: its compiler-peak sample contained about 3781.6 MiB of
file-backed RSS, versus 5795.5 MiB in the same-day old control. Page residency,
reclaim, and allocation timing differ under the memory guard.

### What occupies the warm peak

```text
New-worktree stage                  Sampled RSS peak   Anon at that peak   File-backed at that peak
---------------------------------  -----------------  ------------------  ------------------------
TRT cache cold: four-context init             8,043.0             2,003.9                   5,961.1
Both caches warm: four-context init           6,480.0               479.7                   5,922.3
Both caches warm: first prefill                 873.8               511.4                     236.4
Both caches warm: steady decode                 878.6               515.7                     236.9
```

Anonymous and file-backed RSS are sampled at the same peak row, not maxima
from unrelated instants. Shared-memory RSS accounts for the remaining amount.
Most of the warm initialization peak is transient file-backed artifact
residency. After unmapping, that residency disappears; runtime-cache reuse
also avoids much of the anonymous allocation seen with an empty TRT cache.
The traces do not identify every internal TensorRT allocation or prove whether
its remaining anonymous memory is live SDK state or allocator retention.

### Timing caveats and the three cache states

AOT artifacts, TensorRT runtime caches, and the OS filesystem page cache are
three different things. The first TRT-cache-cold hit read about 6350.4 MiB from
storage and took 87.1 seconds end to end. Repeating with an empty TRT runtime
cache but primed AOT files read about 391.9 MiB and took 32.3 seconds; four-
context initialization fell from 80.3 to 23.3 seconds. With both caches warm,
the run read about 4.7 MiB and took 19.3 seconds end to end. No code change
separates these runs.

The first new cold-AOT process took 421.2 seconds, versus 341.3 seconds for the
same-day retained implementation and the historical 272.8 seconds. A second
new run explicitly primed the source model before its preflight; it took
422.9 seconds and peaked at 23,981.0 MiB. Its compiler/packing/persistence
interval was 378.6 seconds, compared with the old control's 301.5 seconds
(about 26% slower). Priming the input did not resolve this difference. The new
runs also swapped about 130 MiB, whereas the old control did not.

**Cold compilation latency remains an unresolved verification finding.**
The available checks establish functional equivalence and no steady-state
throughput regression, not cold-start equivalence. The NVIDIA compiler source
is unchanged apart from include ordering, the SDK/toolchain versions match,
and independently compiled artifact contents are identical. Those facts do
not explain away the timing difference. Isolating old/new LM hosts and old/new
plugin binaries under matched memory/cache conditions is a separate follow-up;
no speculative compiler or integrity change was made to hide the result.

## Measurement method and raw evidence

The process sampler requests `/proc/<pid>/status` and IO counters every 10 ms,
`smaps_rollup` every 250 ms, and `nvidia-smi` device memory every 100 ms. Actual
intervals include polling overhead. NVIDIA's existing stage markers label the
timeline. The sampler uses a monotonic elapsed clock; the implementation's
field named `monotonic_ns` is actually wall-clock time.

The completed child process's `ru_maxrss` is the lifetime CPU HWM. Stage peaks
and swap/device peaks are sampled, approximate observations, not independently
reset OS high-water marks. Small `/proc` accounting differences of a few MiB
can make a sampled RSS appear slightly above `ru_maxrss`. The raw traces retain
every transition and distinguish anonymous, file-backed, shared, and swap
memory. Profiling-run throughput is not used for the performance comparison.

Raw evidence lives under:

```text
/home/lijin/odml/rt_g3/LiteRT_trt_rtx/litert/vendors/nvidia/results/migration_20260905
```

Each memory directory contains `output.log`, `memory.json`, incremental
`.rss.jsonl`/`.smaps.jsonl`/`.vram.jsonl` journals, and a `summary.txt` containing
every fine-grained interval. Exact commands are retained as `*_command.sh`.
`verify_commits.sh` records the original 29-commit identity checks and is tied
to the September 5 manifest and backup history. The September 6 cleanup changes
both the manifest and author policy, so that historical verifier is not a
validator for the current branch. Historical commands referencing
retired source paths are evidence, not the recommended current workflow;
use the permanent-worktree preset in `gemma4_12b.md` for new runs.
