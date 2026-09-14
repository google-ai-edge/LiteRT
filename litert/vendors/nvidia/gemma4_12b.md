# Gemma 4 12B TensorRT-RTX measurement and optimization

## Result

The 6.41 GiB checkpoint is downloaded and SHA-256 verified. The original
all-signature configuration exceeded this machine's resources. After fixing
compatibility/runtime bugs, the explicit `prefill_1024,decode` configuration
provides the working baseline and all comparisons below (2,048-token context,
1,024 prefill tokens, 256 decode tokens, speculation disabled).

Pruning unused graphs and offloading the vocabulary head raise steady decode
from 25.78 to 65.73–66.03 tokens/sec, with prefill remaining about 3,800
tokens/sec. AOT-hit CPU HWM with a cold runtime cache falls from 19,125.4 to
8,126.5 MiB (57.51%). With a populated runtime cache, the final path peaks at
6,476.5 MiB and reaches about 877 MiB RSS during decode. Device-wide NVIDIA
peak grows by 482 MiB, to 13,716 MiB. Cold compilation still needs about
23 GiB RSS; in-memory JIT also incurs substantial swap and retained storage.

CPU and NVIDIA normal generation agree on `Paris` and the exact sequence
`1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12`. This is a compatibility and targeted
numerical check, not a general quality evaluation. All changes are local;
nothing has been pushed to GitHub.

## Scope and acceptance criteria

Measure text generation on the RTX 5080, starting from LiteRT `e1f885d98`
and LiteRT-LM `ba82499873945908bf8bcfc96e955d0677eb1fa1`. This baseline
already includes the NVIDIA subbyte GEMV kernel, in-memory JIT handles,
shared weights, sharded AOT artifacts, and lazy AOT identity validation.
Do not attribute those existing improvements to this experiment.

Keep the model, quantization, activation mode, context length, prompt, and
measured token counts fixed in paired comparisons. Record any change to
loaded signatures or speculative decoding explicitly. Memory savings are
not accepted at the cost of throughput or correctness.

Checklist:

- [x] Download the specified checkpoint and verify its SHA-256.
- [x] Inspect container sections, signatures, and constant-buffer sizes.
- [x] Build the unchanged baseline and pass plugin/setup tests.
- [x] Obtain a full-model CPU reference response.
- [x] Validate the selected-signature full-model response against CPU.
- [x] Pass the host's protected `benchmark` preflight.
- [x] Measure in-memory JIT, AOT miss, and AOT hit initialization/memory.
- [x] Measure uninstrumented prefill/decode throughput with repeated runs.
- [x] Identify bottlenecks from the baseline; implement one change at a time.
- [x] Check accepted optimizations with tests, paired throughput, and memory;
      retain failed original configurations without inventing their throughput.
- [x] Commit accepted milestones and record final commands/results.

## Checkpoint

Source: [Hugging Face model file](https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/blob/main/gemma-4-12B-it.litertlm).

```text
Local file: /home/lijin/odml/llm/models/gemma-4-12B-it-litert-lm/gemma-4-12B-it.litertlm
Bytes:      6,883,278,368
SHA-256:    58fd31b778ca2c21c80d634fb34fc5a89d11d563a38dfd3cbf1b40dbf252a8b6
Format:     LiteRT-LM 1.6.0
UUID:       a66be780-4cf6-41dd-a4f9-b9d1c7875273
Created:    2026-09-02T16:53:10.794885+00:00
```

Container inventory, from its FlatBuffer metadata:

```text
Section                               Bytes
Main prefill/decode model       6,095,240,176
Embedder                         506,465,360
MTP drafter                      216,123,824
Vision encoder                    50,778,096
Audio encoder                      9,839,872
Tokenizer                          4,689,013
```

The main model has 1,914 subgraphs, including decomposition subgraphs.
Its four exported signatures are `prefill_128` (0), `prefill_1024` (1),
`decode` (2), and `verify` (3). They are not four copies of the model file:
747 nonempty constant buffers total 5,956,315,561 bytes. Decode references
5,953,290,240 bytes of INT4 constants. The largest is the
`[262144, 3840]` vocabulary projection, 503,316,480 packed bytes.
The separate embedder also contains a constant of that shape and size;
content equality has not yet been checked.

## Measurement protocol

- Run the host's `benchmark` command before formal GPU measurements;
  preserve ChatGPT, Codex, remote access, WSL, Explorer, and DWM.
- Finish downloads and builds before measurements. Record GPU idle state,
  driver/SDK versions, source revisions, environment, and raw logs.
- Baseline settings: `PREDEQUANT_MODE=cuda_gemv`, BF16 activations,
  2,048-token context, 1,024-token prefill, 256-token decode. Do not silently
  exclude signatures to make the baseline fit.
- Throughput: eight iterations, report all values and the final six mean
  and median; repeat paired runs with profiling disabled.
- Memory: separate one-iteration runs with the existing phase markers,
  Linux process RSS/HWM, anonymous/file-backed RSS and swap sampling,
  and device-wide CUDA/NVIDIA telemetry. Sampled stage maxima are lower
  bounds, distinct from the kernel-maintained lifetime CPU HWM.
- Distinguish JIT (`LITERT_NVIDIA_TENSORRT_JIT_HANDLE=1`, no AOT), AOT miss,
  AOT hit with a cold runtime cache, and AOT hit with a warm runtime cache.
- If a baseline fails or exceeds resources, retain its failure and last
  measurements; do not report an inferred throughput or successful peak.

The original results are preserved unchanged under
`/home/lijin/odml/rt_g3/LiteRT_trt_rtx/litert/vendors/nvidia/results/previous_runs/gemma4_12b`.
Embedded historical commands still identify their original paths. Fresh
measurements on the latest upstream base are in `trt_rtx_measurements.md`;
use the permanent-worktree preset below for new runs.
Successful selected-signature measurements are recorded below. The original
all-signature configuration did not fit; its failures are retained separately.

## Completed setup checks (2026-09-04)

Both NVIDIA shared libraries and both LiteRT-LM executables built with
Bazel `-c opt`. `ldd` resolves all dispatch dependencies, and
`litert_lm_advanced_main --helpshort` reaches the executable's flag parser.
The help command returns 1 as expected for this Abseil help path; it is
not an inference test.

These Bazel targets pass on the installed TensorRT-RTX 1.5.0 SDK:

```text
//litert/vendors/nvidia:bytecode_test
//litert/vendors/nvidia:profiling_test
//litert/vendors/nvidia/compiler:subbyte_gemv_plugin_test
```

The retained external sampler's CPU-only smoke test observes a 32 MiB
anonymous allocation, its release, the retained process HWM, and NVIDIA-style
stage transitions. It has not yet profiled this model.

The full checkpoint runs successfully on CPU with eight threads, a
2,048-token context, and the prompt
`Answer with only the capital city: What is the capital of France?`.
It returns `Paris` and exits successfully. The cold CPU reference run took
15.85 seconds including initialization, with `/usr/bin/time -v` reporting
7,205,988 KiB maximum RSS. These are single-run compatibility observations,
not a repeated throughput benchmark or TensorRT measurements. Raw evidence:
`cpu_reference/output.log` and `cpu_reference/time.txt` under the
external results directory above. Future AOT output verification should use
`AOT_VERIFY_EXPECTED_OUTPUT=Paris`, not the E2B script default `PARIS`.

The plugin serialization/reference test now covers six cases:

```text
Bits    Rows    Columns   Purpose
2          8        64   Original INT2 regression
4          8        64   Original INT4 regression
4          9      3840   Partial output block at 12B input width
4      15360      3840   12B feed-forward expansion
4       3840     15360   12B feed-forward contraction
4     262144      3840   Full 480 MiB vocabulary projection
```

All cases build and deserialize a TensorRT engine, execute its plugin, and
match every output's BF16 bits to a CPU reference for the deterministic test
inputs. This does not prove full-model NVIDIA accuracy or performance. No
compiler, dispatch, or CUDA kernel implementation was changed by this test
expansion.

The first host preflight selected High Performance and ASUS Profile 1, then
stopped because the protected Windows ChatGPT desktop app was not running.
No protected process was changed. Log:
`/home/lijin/.local/state/benchmark-mode/preflight-20260904-180640.log`.
On 2026-09-04 the user removed the requirement to launch ChatGPT. The
preflight and agent policy now protect it if running but permit a closed app.
The revised preflight passed with Windows High Performance, ASUS Profile 1,
P8/0% GPU utilization, 34 C, and an 874 MiB device-wide idle baseline:
`/home/lijin/.local/state/benchmark-mode/preflight-20260904-190824.log`.
No additional cleanup targets were authorized or added.

## Full-model NVIDIA compatibility failure

A separate, non-benchmark diagnostic attempted the unchanged NVIDIA
implementation with AOT/shared weights enabled and all four signatures kept.
Profiling was disabled. To protect the desktop, only this process ran in a
systemd scope with `MemoryMax=22G` and `MemorySwapMax=8G`. These limits are
not a claim about the machine's absolute model capacity.

The runtime produced 275 TensorRT partitions:

```text
Signature        Partitions
prefill_128               49
prefill_1024             128
decode                    49
verify                    49
Total                    275
```

All engines compiled and all 275 AOT artifacts were persisted, totaling
22,294,793,017 bytes. The compiler reported 16,132,599,914 referenced weight
bytes, including 10,675,121,556 bytes duplicated across shards. Sharding is
per TensorRT partition, not per exported model signature.

Initialization completed contexts 0 through 79, then the process was killed
while creating context 80. Exit status was 137; systemd independently reports
`Result=oom-kill`, a 22.0 GiB scope memory peak, and an 8.0 GiB swap peak.
Scope memory includes charged file cache and is not interchangeable with
process RSS. There was no generated answer and no throughput result. GPU
usage returned to the 893 MiB idle baseline after process termination.

Evidence: `npu_reference/output.log`, `scope_journal.log`, `scope_result.txt`,
and its `aot/` directory under the external results directory. The unit is
`litert-gemma4-12b-compat.scope`; its journal records the OOM termination at
18:42:42 PDT on 2026-09-04.

The logs and source identify these candidates for investigation:

- Native cache-update lowering requires INT8 caches, while this checkpoint
  uses FP16 caches. Inlining its decomposition exposes constant INT64-to-FP16
  casts that `IsCastSupported` currently rejects unconditionally.
- The large prefill path also exceeds the current softmax and batched-matmul
  element caps, causing additional boundaries.
- Both decode and verify vocabulary projections exceed the 256 MiB FC cap.

Do not infer the speedup or memory savings from reducing these boundaries.
Formal baseline measurements and paired validation remain pending. The
diagnostic artifacts are kept separate from future formal-run caches.

## Instrumented AOT-hit baseline (2026-09-04)

After the successful host preflight, `baseline_warm_memory/` retried the
unchanged implementation against the diagnostic's existing AOT artifacts.
The runtime cache directory also existed, but initialization did not finish
in the original diagnostic, so this is not a fully warmed runtime-cache run.
All four signatures, 2,048-token context, and the same Paris prompt were kept.
Memory/dispatch checkpoints were enabled; this is not a throughput run.

The compiler confirmed an AOT hit for all 275 modules. Lookup took 0.224 s
between its begin/hit checkpoints; CPU lifetime HWM at the hit was 664.4 MiB.
Initialization then completed 124 contexts, failing after engine 124 had
deserialized. The scope reported OOM, a 22.0 GiB memory peak, and a 7.0 GiB
swap peak. The last recorded process lifetime HWM was 22,465.0 MiB; the
largest CUDA device-used checkpoint was 11,073.6 MiB. These are partial-run
observations, not successful full-model memory requirements or final peaks.
The guard and swapping also make this unsuitable for timing comparisons.

For representative partition 5, CPU RSS was:

```text
Checkpoint                      MiB
Before runtime creation     3,742.1
After runtime creation      3,742.1
After engine deserialization 3,750.3
After refit                 3,857.0
After context creation      4,390.8
After artifact unmap        4,279.6
```

The context-creation interval added 533.8 MiB; runtime creation itself was
negligible. This is evidence to investigate excessive execution contexts,
not evidence that sharing `IRuntime` objects would solve the memory growth.

Raw checkpoint and systemd logs are retained in `baseline_warm_memory/`.
OOM cleanup terminated the sampler before its final JSON was written, so
there is no complete continuous trace for this attempt. The adapter now
journals RSS, smaps, and GPU samples incrementally for subsequent attempts.

## First targeted change: constant cache casts

The new `tensorrt_graph_builder_test` reproduced rejection of a constant
INT64-to-FP16 cast. Allowing only this additional cast combination makes
the scalar and vector tests pass, including TensorRT compilation,
deserialization, and bit-exact FP16 execution. Other constant cast type
combinations retain the previous support policy.

The full-model compilation in `cast_only_cold/` now partitions as follows:

```text
Signature        Original   Constant-cast support
prefill_128             49                       2
prefill_1024           128                      88
decode                 49                       2
verify                 49                       2
Total                 275                      94
```

This is a measured reduction in graph fragmentation, not yet a demonstrated
end-to-end speedup or memory saving. Large-prefill attention element caps
remain unchanged, as do the FC weight cap, signatures, and runtime settings.
Full-model initialization/output and performance validation remain pending.

The cast-only attempt compiled the first 93 partitions, including a
5,465,459,476-byte self-contained decode engine. Building partition 94 (the
large verifier partition) failed inside TensorRT with `OutOfMemory
(Requested size was 9736686286 bytes.)`. The message does not identify
CPU versus device allocation; do not infer the allocation category from it.
LiteRT fell back to CPU after this error. That fallback was explicitly stopped
after recording the failure; it is not a successful NVIDIA result.
Incremental sample journals and the full log remain in `cast_only_cold/`.

Source inspection identifies an additional compiler-lifetime issue: all
compiled plans and the global shared-weight store remain resident until
every partition has compiled, even though AOT writes independent shards.
Persisting each completed AOT shard before building the next partition is
the next experiment. It must preserve the per-engine artifact contents,
publish the manifest only after all shards succeed, and leave non-AOT
cross-engine deduplication unchanged.

## Second change: stream completed shared-weight AOT shards

The compiler now packs and atomically publishes each completed shard inside
the build loop. Its local deduplicated weights and engine storage are released
before the next build. Only small locators survive. Non-AOT still uses a
global deduplicated weight store and one multi-engine module. The manifest
is published only after every partition succeeds. Compiler cache schema 4
invalidates previous AOT compilation results for these lowering changes.

New tests verify early publication before a later build failure, absence of
a complete manifest on failure, independent shards and warm manifest reuse,
and preserved cross-engine weight sharing in non-AOT mode. The small-model
artifacts retain the before/after content fingerprints. Existing bytecode,
AOT identity/corruption, and constant-cast tests pass.

The full `stream_aot_cold/` attempt still failed while building the large
verifier partition with the same 9,736,686,286-byte allocation request.
Before starting that partition, however, CPU RSS fell from 18,099,089,408
bytes (16.86 GiB, cast-only) to 11,064,320,000 bytes (10.30 GiB, streaming):
a 6.55 GiB reduction at that checkpoint. This is not an end-to-end HWM
reduction. Earlier large-partition build/copy peaks remain, and neither run
completed TensorRT initialization. The CPU fallback after the compiler error
was stopped explicitly and is excluded from NVIDIA results.

All incremental journals and the 93 completed shards are retained outside
the worktree. There is intentionally no complete-model AOT index for that
failed attempt. The next experiment will explicitly select large prefill and
decode and omit unused signatures; it must be labeled as a separate workload
configuration rather than silently changing the all-signature baseline.

## Reproduction environment

### Opt-in two-signature experiment

LiteRT-LM commit `ff61ed89` on local branch
`codex/gemma4-12b-signatures` adds `--selected_signatures`. Empty preserves
the existing all-signature behavior. Its two tests verify settings forwarding
and prefill-runner filtering; the optimized executable builds successfully.

The `selected_cold/` experiment explicitly selects `prefill_1024,decode`
and sets `LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS=prefill_128,verify`.
Both are needed: the compiler's subgraph filter avoids building unused
engines, while selected signatures restrict runtime initialization and
prefill-runner selection. No speculative decoding is enabled.

At the time, this experiment also raised `LITERT_NVIDIA_TENSORRT_MAX_SOFTMAX_ELEMENTS`
and `LITERT_NVIDIA_TENSORRT_MAX_BATCH_MATMUL_OUTPUT_ELEMENTS` to
67,108,864. It produces four partitions: two for large prefill, two for
decode. The default 256 MiB FC weight limit remains unchanged, so the
480 MiB vocabulary projection still runs outside TensorRT.

The softmax and batch-matmul element-count guards have since been removed;
those two environment variables no longer affect partitioning. The historical
results above retain their original configuration. TensorRT now determines
whether the admitted attention operations can be compiled within its resources.

This is a different loaded-signature configuration from the original
all-signature baseline. It does not preserve the short-prefill latency
benefit of `prefill_128`, and is not yet a successful inference result.

All four partitions compiled and persisted successfully (about 10.1 GiB of
AOT artifacts), and all four TensorRT contexts initialized. The process then
aborted when LiteRT-LM queried signature output metadata. `selected_debug/`
and `selected_debug2/` contain debugger backtraces identifying the same
composite-inlining bug at two call sites. Inlining rewrote graph outputs but
did not remap the signatures' separate tensor references before DCE deleted
the old output tensors. The new alias regression test fails on the old
implementation and passes with `418c9cf7e`; the complete algorithm and model
test suites also pass. The temporary LiteRT-LM call-order workaround was
removed after fixing the underlying references.

The failed `selected_cold/` run recorded 22,642.1 MiB lifetime CPU HWM and
11,591 MiB device-wide NVIDIA peak. A failed AOT-hit attempt in
`selected_warm/` recorded 18,939.9 MiB CPU HWM and 12,012 MiB NVIDIA peak.
Neither value is a successful inference peak. Both attempts reached core
dumping after an abort; those dumps were explicitly stopped and their tail
must be excluded from stage-duration analysis. The incremental journals and
final sampler JSON files remain available. Warm-hit CPU memory already
includes substantial reserialization and engine/refit storage; it is not just
the size of the AOT locators.

With the signature fix, `selected_debug3/` reached actual prefill invocation
but crashed in a CPU `Slice<int>` kernel. A mixed CPU/backend host buffer was
registered with the backend and incorrectly marked `kTfLiteNonCpu` with a
null data pointer, despite also having CPU consumers. Commit `11d5091c8`
retains the CPU allocation for such shared host buffers. Its small add-model
regression fails before the fix with XNNPACK's `unexpected null data pointer`
and passes afterward, alongside the full C++ compiled-model test target.
Device-only buffers keep their existing registration path. The full runtime
test target could not be configured because this checkout's `ml_drift`
repository declaration has no source URL; the regression lives in the
independently buildable C++ test target instead.

```bash
cd /home/lijin/odml/rt_g3/LiteRT_trt_rtx/litert/vendors/nvidia
export LITERT_G3_HEAD=/home/lijin/odml/rt_g3/LiteRT_trt_rtx
export LITERT_LM_G3_HEAD=/home/lijin/odml/llm/LiteRT-LM_trt_rtx
export TENSORRT_RTX_ROOT=/home/lijin/opt/tensorrt-rtx-sdk
export CUDA_HOME=/usr/local/cuda
export G4MODEL=/home/lijin/odml/llm/models/gemma-4-12B-it-litert-lm/gemma-4-12B-it.litertlm
export PREDEQUANT_MODE=cuda_gemv
export SELECTED_SIGNATURES=prefill_1024,decode
export SKIP_SUBGRAPHS=prefill_128,verify
# The accepted 12B vocabulary-head offload; use 268435456 for the control.
export LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES=536870912
export LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS=1
export LITERT_NVIDIA_DISPATCH_DUMP_IO=0
export AOT_VERIFY_EXPECTED_OUTPUT=Paris
export MEMORY_PROFILE_DECODE_TOKENS=256

./run_head.sh build
# In the configured interactive zsh, pass `benchmark` before measuring.

# AOT miss, AOT hit with runtime cache cold, and both caches warm.
# Use a new RUN_ROOT; the first pass compiles and needs substantially more RAM.
RUN_ROOT="$PWD/results/12b_aot_measurement" ./run_head.sh memory-profile-aot

# Reuse the already verified artifacts for a normal response or throughput.
export LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR="$PWD/results/12b_aot_measurement/aot_memory_profile/aot_artifacts"
export LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH="$G4MODEL"
export LITERT_NVIDIA_TENSORRT_JIT_HANDLE=0
RUN_ROOT="$PWD/results/12b_numeric" ./run_head.sh numeric
RUN_ROOT="$PWD/results/12b_benchmark" \
  LITERT_NVIDIA_MEMORY_PROFILE=0 LITERT_NVIDIA_DISPATCH_PROFILE=0 \
  ./run_head.sh benchmark

# Fresh JIT is an intentionally high-memory comparison, not the deployment
# recommendation. No reusable AOT artifacts are produced by this command.
RUN_ROOT="$PWD/results/12b_jit_measurement" \
  LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR= \
  LITERT_NVIDIA_TENSORRT_JIT_HANDLE=1 ./run_head.sh memory-profile
```

## CUDA transfer correctness milestone

After the host-allocation fix, the default CUDA-buffer path executed but
generated reserved tokens (`selected_debug4/`, `selected_logits_no_dump/`).
Using host I/O as a control returned `Paris` (`selected_host_io/`), matching
the CPU reference. The dispatch bridge had reversed its staging lock modes:
the CPU-to-device memcpy requested Read, while device-to-CPU requested Write.
NVIDIA's custom-buffer callbacks therefore skipped the required upload or
overwrote a device result with stale staging data. Correcting those two modes
restores `Paris` with the default CUDA buffers (`selected_cuda_fixed/`), without
forcing the large KV buffers through host memory. The four dispatch delegate
tests and both LM signature-selection tests also pass. These short, profiled
runs establish compatibility, not throughput or broad numerical equivalence.

The optional `LITERT_NVIDIA_DISPATCH_DUMP_IO=1` diagnostic itself reads past
the end of small buffers when sampling from their midpoint; its CUDA error
in `selected_logits/` is excluded from model/kernel correctness evidence.
All subsequent runs disable this debug option.

## Candidate to evaluate after the baseline

`tensorrt_graph_builder.cc` defaults
`LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES` to 256 MiB. The 480 MiB
packed INT4 vocabulary projection exceeds that limit. First confirm the
rejection in the model's partitioning log, then compare an explicit 512 MiB
limit while preserving the same workload. This is a source-derived candidate,
not a change to the global default. The measured 512 MiB experiment below
enables this existing capability for the 12B checkpoint.

## Verified signature-pruning milestone

LiteRT now reads explicit runtime signature selection before model conversion.
On NPU compiler-plugin cache misses, it reuses `PruneModelToSignatures` to
retain only selected roots and their reachable decomposition graphs before
partitioning and serialization. The 12B selection retains two signatures and
957 reachable subgraphs, removing two signatures and 957 unused subgraphs.
CPU-only compilation, empty selection, and already compiled models retain
their existing behavior. Unsupported control-flow references skip pruning
without changing the original graph; missing keys remain errors.

This is implemented within LiteRT, respecting the internal graph API's
visibility boundary. The initial attempt to call that API directly from LM
was rejected by Bazel visibility and was removed. No visibility was widened.
LM only exposes/forwards the explicit selection and documents its meaning.

`pruned_256_numeric/` returns `Paris` and hits the exact pre-existing four
engine artifacts and AOT index used by the baseline. No TensorRT engine was
rebuilt for this comparison. Tests pass: model/pruning closure and unsupported
reference cases, runtime options, the full C++ compiled-model target (including
a new CPU signature-selection execution regression), and LM selection tests.

The separate 1024-prefill/256-decode memory passes, each with an empty TensorRT
runtime cache, measured:

```text
Boundary / metric (MiB)                    Baseline       Pruned
Process CPU HWM                           19,125.4       8,534.6
RSS before first artifact mapping          11,939.8       1,427.1
RSS after last artifact unmap              13,915.4       3,325.1
RSS at first decode invocation             14,455.1       3,889.6
Median sampled decode RSS                  14,457.4       3,892.4
NVIDIA device-wide peak                    13,234         13,234
```

HWM saving: 10,590.9 MiB / 55.38%. These are process RSS measurements, not
all anonymous allocations: pruning avoids both an owned serialized copy of
unused weights and faulting their original file-backed pages into process RSS.
The engines, quantization, selected workload, and GPU allocations are unchanged.

Do not compare these two memory runs' startup times directly. Their physical
read totals differ (181 MiB versus 4,645 MiB); the latter refit reads spent
substantially more time on storage. Later throughput runs prime the scoped
AOT artifact files and reuse populated TensorRT runtime caches. They do not
drop global filesystem caches. Even then, original-model page residency can
vary; all raw I/O counts and per-run initialization times are retained.

Uninstrumented eight-iteration runs (mean of final six, tokens/sec):

```text
Run                         Prefill        Decode
selected_perf_baseline_1     3,796.71        25.782
pruned_256_perf_1            3,803.55        26.150
baseline_perf_2             3,796.45        24.707
```

No steady-throughput regression was observed from pruning. Avoid treating
small differences as a precise speedup: the CPU vocabulary projection remains
in these runs, and CPU/I/O/cache state produces observable run-to-run variation.

## Vocabulary-head offload and the combined result

The 512 MiB FC cap selects the 480 MiB INT4 vocabulary projection for the
existing CUDA GEMV plugin (`N=262144, K=3840`). This is a per-run setting, not
a global limit increase or a new kernel. The prefill engine is unchanged.
`fc512_numeric/` returns `Paris` using the frozen, pre-pruning executable,
isolating the head-offload correctness check. Its full-shape plugin reference
test is also among the six passing INT2/INT4 GPU tests.

With pruning and head offload combined, two eight-iteration, uninstrumented
runs produced these means over the final six iterations:

```text
Run                 Prefill tokens/s    Decode tokens/s
final_perf_1              3,794.43              65.732
final_perf_2              3,807.58              66.028
```

This is about 2.55x the original working baseline's decode throughput, with
no observed prefill regression. `final_perf_1` exercised the updated
`run_head.sh benchmark` path, including the signature-selection arguments.
Both runs completed successfully with no execution errors. These checks do
not replace a broad model-quality evaluation.

Matched AOT-hit cases with an empty TensorRT runtime cache, all MiB:

```text
Metric                              Baseline      Prune only       Combined
CPU process HWM                     19,125.4         8,534.6         8,126.5
CPU RSS after final artifact unmap   13,915.4         3,325.1         2,433.9
CPU RSS at first prefill             14,414.9         3,836.9         2,462.8
CPU RSS at first decode              14,455.1         3,889.6         2,516.9
Median sampled decode RSS            14,457.4         3,892.4         2,519.5
NVIDIA device-wide peak              13,234          13,234          13,716
Maximum process swap                     0               0               0
```

Incremental HWM reductions in this order: pruning saves 10,590.9 MiB;
head offload saves a further 408.1 MiB; total saving is 10,998.9 MiB / 57.51%.
These are sequential, configuration-specific contributions, not independent
additive estimates. Artifact I/O differs across runs, so this is a memory
comparison rather than a claim about matched initialization latency.

With both AOT and TensorRT runtime caches populated:

```text
Metric (MiB)                        Prune only       Combined
CPU process HWM                       6,959.6         6,476.5
CPU RSS after final artifact unmap     1,749.1           785.4
CPU RSS at first decode                2,314.4           875.9
Median sampled decode RSS              2,315.7           877.4
Maximum process swap                       0               0
```

The additional head-offload saving in this warm-cache state is about 483 MiB
at HWM and 1,439 MiB during decode. It removes the remaining CPU vocabulary
projection, its serialized/source weights, and CPU execution storage. No
claim is made that all of those bytes are FP32-expanded weights.

Runtime-cache warmth itself matters: the combined configuration's last
sampled inference RSS is about 2,519 MiB with a cold runtime cache versus
877 MiB with a warm one. Roughly 1,605 MiB of that difference is anonymous
memory associated with cold runtime compilation/initialization, rather than
the model file or a larger VRAM allocation. Allocation-level ownership of
that SDK-side difference has not been determined.

The later `baseline_hot_memory/` control paged up to 290.8 MiB and performed
7,734 MiB of physical reads; its 123.67-second executor initialization and
instrumented throughput are not used as the performance-regression reference.
Its RSS HWM also reflects that paging. The earlier unpaged baseline and the
repeated uninstrumented throughput runs are the principal reference.

Storage/GPU trade-off: total AOT module bytes increase from 10,854,387,636 to
11,358,228,140 when the vocabulary head moves into the decode engine, a
503,840,504-byte (480.5 MiB) increase. Device-wide NVIDIA peak rises by 482 MiB.
This fits the RTX 5080 for the selected 2,048-token text workload; it is not
a claim that longer contexts or all four signatures will fit.

## Fresh compilation and initialization breakdown

These are completed one-iteration, instrumented runs of the combined
configuration. They include 1,024 prefill and 256 decode tokens; their
throughput is not substituted for the uninstrumented results above.

```text
Metric                                  Cold AOT process     In-memory JIT
Whole process elapsed, seconds                   272.775           371.687
Compiler + packing/persistence, seconds          240.253           338.256
Executor initialization, seconds                 267.296           362.834
Four context initializations, seconds             24.598            22.882
Process CPU HWM, MiB                          23,517.1          23,523.6
Maximum observed process swap, MiB                 0             6,349.2
CPU RSS at first decode, MiB                   9,240.9          15,353.0
Median sampled decode RSS, MiB                9,241.2          15,359.9
NVIDIA device-wide peak, MiB                  13,716            13,716
```

Compiler time is inside executor initialization, and context initialization
is also inside executor initialization. Do not add those three rows. RSS HWM
and maximum swap are distinct measurements whose peaks need not coincide.
JIT's similar RSS HWM does not imply similar total memory demand: substantial
paging limited residency and affected its time. The cold AOT run did not swap.

Cold AOT compiler checkpoints, MiB and seconds:

```text
Partition  Role                Build s  Pack s  Persist s  RSS after build  Lifetime HWM
0          Small prefill         2.426   0.007      0.210            591.2         685.2
1          Large prefill        94.876   2.936     34.347         12,658.0      21,853.6
2          Small decode          0.181   0.002      0.211          8,695.0      21,853.6
3          Large decode         57.365   2.162     41.083         17,351.6      23,517.1
```

Build includes graph lowering, TensorRT build/serialization, copying the
engine result, and graph-builder cleanup. Pack and persist are measured from
their explicit markers. These columns do not exhaust compiler time: weight
collection, gaps between markers, final cleanup, and index publication also
contribute. Persistence includes file creation/write/sealing; its wall time
depends on storage and page-cache state. Lifetime HWM is the OS-maintained
maximum so far, not a separately reset peak for each partition.

Cold AOT context setup (seconds):

```text
Partition                         0          1          2          3
Initialize total              0.655     13.401      0.415     10.127
```

The raw markers further separate runtime creation, engine deserialization,
refit binding/refit, runtime-cache setup plus execution-context creation,
artifact unmapping, input/output setup, enqueue, synchronization, and release.
The execution-context interval includes runtime-cache hashing/loading before
the SDK context-creation call; it is not exclusively that one SDK call.

In-memory JIT's large retained bundle is intentional in the current handle
implementation: it avoids copying the bytecode into the rewritten LiteRT
FlatBuffer and disables persistent LiteRT JIT model caching, but still
serializes TensorRT engines and packs the engine/refit bundle in RAM. The
bundle survives with the executable handle. Its bundle-pack interval alone
was 78.982 seconds in this paged run. Small TensorRT runtime-cache files are
separate; they do not eliminate model compilation in a new JIT process.

The cold AOT process also remains much larger after compilation than a fresh
AOT-hit process. Its median sampled decode RSS contains about 7,955 MiB
anonymous, 1,161 MiB file-backed, and 126 MiB shared memory, versus about
517 MiB anonymous, 234 MiB file-backed, and 126 MiB shared in the warm-cache
process. The difference is primarily host anonymous memory, not simply a
retained mapping of the original checkpoint. Its allocation-level split
between live SDK/compiler state and allocator-retained free space has not
been established. Run compilation as a separate process
for deployment; reusing its artifacts in a fresh process gives the low-memory
path measured above. No claim is made that first-process JIT has been reduced
to that footprint.

## Reproduction and evidence

The `run_head.sh` preset above exposes signature selection in numeric,
benchmark, AOT verification, and memory commands. Defaults preserve the old
unset compiler-skip variable, including its cache identity; an empty string
would have changed the existing compiler configuration fingerprint. Both
the default/unset and explicit-selection shell argument paths were checked,
and `bash -n` and `git diff --check` pass.

Each new measured run has a `command.sh` under the external results root.
Those scripts retain the actual executable, environment, caches, resource
guard, and workload. The `baseline_runtime/` executable is frozen before
pruning; `pruned_runtime/` is frozen after it. The frozen NVIDIA shared
libraries match the worktree's built libraries by SHA-256. The fresh final
AOT compile reproduces all four previously tested module identities/sizes.

Key result directories:

```text
selected_perf_baseline_1  Unpaged working baseline, eight iterations
pruned_256_perf_1         Memory-only change, eight iterations
baseline_perf_2           Repeat control; storage-sensitive startup
final_perf_1             Combined, via run_head.sh, eight iterations
final_perf_2             Combined repeat, eight iterations
selected_memory_baseline Baseline AOT hit, runtime cache initially empty
pruned_256_memory        Pruning AOT hit, runtime cache initially empty
final_runtime_cold_memory Combined AOT hit, runtime cache initially empty
pruned_hot_memory        Pruning, both caches populated
final_memory             Combined, both caches populated
final_cold_memory        Fresh combined AOT compilation and execution
final_jit_memory         Fresh combined in-memory JIT and execution
count_reference          Exact CPU/NVIDIA counting response comparison
```

Memory directories retain `output.log`, final `memory.json`, and incremental
RSS/smaps/NVIDIA JSONL journals. Sampling: 10 ms RSS, 250 ms smaps, and 100 ms
NVIDIA telemetry. `resource_max_rss_kb` is the child's `getrusage` maximum;
phase-sampled maxima are lower bounds. Hardware telemetry is device-wide,
including the approximately 761 MiB idle Windows baseline, not per-process
VRAM. CUDA device-used and NVIDIA's reported resident usage are different
telemetry counters and are not added together.

Dispatch emits memory checkpoints only for each partition's first invocation.
The first-decode rows therefore refer to that checkpoint, not the 256th token.
The continuous sampler covers the remaining decode calls; its median rows
use the samples after the first large decode call completes and before the
benchmark reports inference completion. It records 382–1,090 such samples in
the principal comparisons, confirming the low steady footprint independently
of the first-invocation checkpoints.

The script-based memory commands produce checkpoint CSVs and HWM summaries.
For the additional continuous samples used here, the retained
`profile_nvidia_stages.py` adapts the pre-existing sampler at
`../webgpu_20260827/profile_webgpu_stages.py`; both must be retained when
copying the external measurement harness to another machine.

No global filesystem caches were dropped. Later paired runs read only their
scoped AOT files to prime them, and preflights retain their diagnostic logs
under `/home/lijin/.local/state/benchmark-mode/`. ChatGPT is no longer required
to be open; an already running instance remains protected. High Performance
and the original temperature-responsive ASUS Profile 1 were used throughout.

## 32K-depth attention guard cleanup and chunk scaling

Commit `25a4b13da` removes the softmax and batch-matmul element-count admission
limits, including the native `odml.runtime_bmm` composite, and their obsolete
SDK fingerprint entries. Dtype, shape, option, workspace and unrelated
operator checks remain. The three new metadata-only regression tests fail
before the cleanup and pass afterward; graph-builder, GEMV and AOT/shared-
weight tests all pass. No CUDA kernel or dispatch implementation changed.

The fresh `prefill_1024` graph at capacity 34818 now has two prefill and two
decode partitions and completes inference. The previous capped graph split
prefill into nine partitions and failed at a CPU-fallback tensor boundary.
Both signatures return `Paris` on the short output check. This establishes
execution, not long-context answer quality or numerical equivalence.

Measurements on 2026-09-08 populate 32768 KV positions before independently
timing 2048 additional prefill tokens or 128 decode tokens. Both backends use
FP16 KV. llama.cpp uses logical batch 2048, FlashAttention, full GPU offload
and the requested microbatch; 512 is not its maximum. Checkpoint weight
encodings, random token streams and sampling work differ between backends.

```text
Backend / prefill chunk           Samples   Prefill tok/s       Decode tok/s
-------------------------------  -------   ----------------   --------------
TensorRT 128, old-plugin control      3     1549.52 +/-  3.76   51.076 +/- .095
TensorRT 128, guards removed          6     1554.03 +/-  5.71   51.268 +/- .257
TensorRT 1024, guards removed         1       33.3169           41.0517
llama.cpp 128                        6     2711.76 +/- 23.32   74.558 +/- 3.911
llama.cpp 512                        6     3289.98 +/- 12.73   74.633 +/- 1.880
llama.cpp 1024                       6     3283.05 +/- 24.55   74.658 +/- .962
```

Plus/minus is sample standard deviation. TensorRT 1024 uses a single sample
per test without an extra in-process warmup, but with AOT/runtime caches and
a complete untimed prefix. Do not infer a precise steady decode regression
from that single sample. The old 128 control retains its necessary 128M
element-limit overrides. The guard-free 128 result shows no meaningful
regression. llama's 128-to-512 prefill gain is 21.32%; 512-to-1024 is flat.

TensorRT's main prefill activation arena grows from 495.52 to 3704.16 MiB,
while main prefill weights change by only 21 KiB. This is one shared grow-only
arena, not four copies. CUDA reports zero free memory in the 1024 case and
Windows GPU shared-memory usage increases by about 1 GiB. These observations
strongly support VRAM oversubscription as the cause of its approximately
46.64x prefill slowdown. They do not isolate the precise paging-time fraction.
The warm 1024 process CPU HWM is 6458.3 MiB, so the bottleneck here is not
the large cold compiler heap. For this 32K workload on the RTX 5080, retain
prefill_128; llama.cpp microbatch 512 is the best tested trade-off. Its 1024
case costs another 690 MiB of sampled device memory without improving speed.

The complete protocol, stage-memory tables, cold/warm initialization results,
frozen binaries, raw traces and reproduction commands are retained in
`results/chunk_scaling_20260908/report.md`. Initial llama wall-clock runs are
excluded: a clock adjustment produced an invalid sample. The final sweep
uses llama.cpp commit `b43095913`, which changes only the benchmark elapsed-
time helper to `steady_clock`; the inference kernels are unchanged. Legacy
NVIDIA memory-marker wall timestamps are not used for throughput conclusions.

## Decode and prefill optimization pass (2026-09-09)

Nine implementation commits on `LiteRT_trt_rtx` improved 32K-depth throughput
on the RTX 5080. The historical headline was 1393 / 48.5 tok/s to 2427 / 74.6
(prefill_128 PP2048 / TG128, driver 596.49), and short-context (2K) decode
60.9 to 91.1 tok/s; see the averaging corrections below. Windows Update
installed driver 596.49 (from 581.95) during the pass. The unchanged baseline
was slower after the reboot (1554 / 51.3 before), but those runs do not isolate
driver causality from other host-state changes. Every comparison below is
same-driver. Historical same-driver llama.cpp: ub128 2632 / 72.2, ub512
3149 / 68.6. The fresh review is in
[optimization_review_20260911.md](optimization_review_20260911.md).

```text
Commit      Change                                   32K PP2048  TG128   2K prefill  decode
---------  ---------------------------------------  ----------  ------  ----------  ------
(baseline)  Codex's guard-free state                     1393     48.5       3503    60.9
ed56ad947   INT4 FC scales as TensorRT block scales      2261     49.4       3416    60.4
6fcffb65d   whole signatures as one partition            2307     51.4       3472    64.4
6a332af46   sibling M=1 sub-byte GEMVs in one launch     2325     51.3       3469    63.3
a2ea45586   KV caches updated in place (KVCacheUpdate)   2355     57.1       3448    67.4
1f65d65cd   per-channel scales for M>512, K>4096           -        -          -       -
b309774a2   persistent auxiliary CUDA streams            2368     57.5       3719    68.1
e44f57984   value caches held as [B, H, S, D]            2323     62.3       3737    73.6
40fabd8ca   runtime_bmm in cache precision, BF16 out     2427     74.6          -       -
f455fcc17   fused decode attention plugin (off)          2400     74.8       3730    91.1
```

This historical table mixes three-sample means with four-sample means that
include the `rep=-1` warmup. Recomputing only `rep>=0` for the two final warm
runs gives 2402.1 / 75.59 and 2424.3 / 73.97 tok/s (PP / TG), respectively;
each has three measured samples. The last row is the committed state
measured after a fresh compile (schema 10). The Paris output check passes at
every step. Full per-run tables,
the anomalous samples and the probe programs are in
`results/opt_20260909/report.md` (untracked, like the earlier result trees).

What each step fixed:

- Block scales: TensorRT-RTX 1.5 never fuses per-channel INT4 dequantization
  into the GEMM; a `__myl_CastMulCast` kernel materialized every weight
  matrix on every prefill invocation (~22 ms of a 70 ms prefill_128 call).
  Block scales along K with a block that divides K fuse into one
  `__myl_ReplCastMulCastFc` kernel. M=1024 down projections (K=15360) are
  slower in fused form, so 1f65d65cd keeps per-channel scales there.
- One partition per signature removes the CPU-resident islands
  (div/floor_mod/minimum admission, Minimum lowering, float operand type
  reconciliation for the soft cap).
- In-place cache updates: the scatter lowering copied every cache on every
  step; `addKVCacheUpdate` writes in place and the dispatch binds the aliased
  output to the input buffer. Decode 51.3 -> 57.1.
- Value cache layout: Gemma stores values as [B, H, D, S] and Myelin
  transposed all 48 value caches on every invocation. Holding them as
  [B, H, S, D] inside the engines (same bytes, the update is transposed
  instead) removes the transposes: decode 57.5 -> 62.3. Holding the keys
  transposed as well measured slower (58.4) because the attention kernels
  want keys as [B, H, S, D].
- runtime_bmm precision: the FP16 matmul -> select -> softmax -> matmul chain
  was fused by Myelin into its `_gemm_mha_v2` kernel, which is 4-5x slower
  than plain FP16 matmul and softmax kernels at the decode shapes (probe:
  local block 73 -> 19 us) and also slower for the prefill shapes. Casting
  the activation operand to the cache type and the result to the activation
  mode type (BF16) keeps the fused kernel out; concatenation and batch
  matmul reconcile the mixed float types this creates in the prefill graphs.
  Decode 62.3 -> 74.6 at 32K and 73.6 -> 91.1 at 2K, prefill 2323 -> 2427.

Remaining decode budget at 32K (about 13.4 ms per token): the GEMV plugins
read the 6 GB of INT4 weights at close to bandwidth (~6.7 ms), the vocab
head 0.5 ms, the eight global attention blocks about 0.28 ms each and the
40 local blocks about 0.02 ms each, ~1000 kernel launches inside the CUDA
graph, plus ~1 ms of LiteRT-LM work and ~1 ms of graph launch/sync latency
outside the engine. Two whole-cache copies per layer remain: every consumer
of a KVCacheUpdate output receives a copy of the cache from Myelin, and
reading the cache input instead makes TensorRT copy the input to protect it.
A plugin consuming the update outputs gets the buffers directly, which is
why f455fcc17 adds a fused decode attention plugin. The first kernel version
measured 40.4 tok/s end to end, slower than the default path. The final kernel
measured 210 us vs Myelin's 285 us for a standalone global block and 24 us vs
19 us for a local block; the old end-to-end result does not measure that final
kernel. It ships off by default behind
`LITERT_NVIDIA_TENSORRT_DECODE_ATTENTION_PLUGIN=1`. A tensor-core
(mma-based) kernel for the 16-row global blocks is the next step there.

Memory (engine labels corrected during the September 11 review): the
**prefill** activation requirement drops from 414,603,520 to 392,871,168 bytes
(395.4 to 374.7 MiB) with the value-cache layout. The final default **decode**
requirement is 81,209,344 bytes (77.4 MiB); the enabled attention experiment
required 5,136,896 bytes (4.9 MiB). These are engine requirements, not additive
resident allocations: dispatch normally shares one activation arena sized for
the largest engine. Reducing decode alone therefore does not remove the
larger retained prefill arena.

The final engines report 5102 MiB of weights for **prefill_128** and 5684 MiB
for **decode**. These are separate engine-owned weight allocations, not a
measurement of exactly how many bytes could be deduplicated across their
different layouts. Sharing weights without sacrificing throughput needs a
compatible W4A16 prefill GEMM: the existing probe found that INT4 weights
supplied as native engine inputs did not fuse (1244 us vs 143 us at M=128).

Measurement notes: nsys/ncu do not observe TensorRT-RTX kernels on this
host; the TensorRT layer profiler has a ~70 us floor per layer on WSL2 and is
only useful for ordering. Standalone probes (`results/opt_20260909/probes/`)
timed with CUDA graph replay reproduce the engine's kernel choices from the
engine inspector names and gave every number above. Warm runs occasionally
fall into a slow mode (prefill ~1700-1830, decode 40-49 tok/s for the whole
process; GPU clocks, temperature and Windows shared GPU memory are normal in
the samples). The historical headline table repeated these runs rather than
including them in its averages, so it should not be treated as an unbiased
average over all processes. The September 11 review retains all runs and
excludes only predeclared warmups and explicitly different workloads/cache
states from each comparison, not samples selected by throughput.
