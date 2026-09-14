# Gemma 4 12B optimization review

This reviews Claude Code session `ef06c228-2554-481a-87e1-dcd2ef46d2dd`
against its actual commands, retained logs and source changes. The reviewed
LiteRT state is `c14114db2`, following the guard-free baseline `65384a7a8`.
LiteRT-LM remains at `d2987579`. Work and measurements use the permanent
`LiteRT_trt_rtx` and `LiteRT-LM_trt_rtx` worktrees.

## What was retained

| Commit | Change | Review assessment |
| --- | --- | --- |
| `ed56ad947` | Express compatible INT4 row scales as block scales | Enables fused weight-only GEMM rather than repeatedly materializing 16-bit weights. |
| `6fcffb65d` | Whole-signature partitioning and supporting operator/type lowering | Removes small CPU boundaries and staging; does not share weights between the resulting engines. |
| `6a332af46` | Fuse sibling GEMV projections | Concatenates compatible q/k/v or gate/up rows into one plugin invocation, then slices the result. |
| `a2ea45586` | In-place KV-cache update | Removes avoidable cache-update work; does not prove every TensorRT-internal cache copy disappears. |
| `1f65d65cd` | Select per-channel rather than block scales for large M/K | Retains the faster measured prefill path for those shapes. |
| `b309774a2` | Retain auxiliary CUDA streams | Reasonable lifecycle cleanup; historical measurements were neutral, not evidence of a major speedup. |
| `e44f57984` | Value cache in sequence-major physical layout | Avoids repeated V transposes; relies on a private layout agreement between NVIDIA signatures. |
| `40fabd8ca` | Cache-precision runtime BMM with BF16 output | Allows better attention fusion while matching the FP16 cache input type; changes arithmetic rounding. |
| `f455fcc17` | Experimental fused decode attention | Retained, disabled by default, as explicitly requested in the original session. |

`c14114db2` is the associated report, not a tenth runtime optimization.
This review checks the aggregate result and focused feature tests; it does not
claim a new isolated ablation of every commit.

## Changes made during this review

`ecfc7510f` consolidates single-projection and grouped subbyte GEMV lowering.
Both paths now use `InspectSubbyteGemvWeights`,
`ValidateSubbyteGemvActivation` and `AddSubbyteGemvPlugin`. Constant/refit names,
layer creation order, plugin ownership and output names are preserved. The
change removes 67 net lines without changing CUDA kernels or feature defaults.
The value-cache comment was moved beside its flag and the reversed FC timing
comparison was corrected.

`73f24816e` fixes a concrete validation gap in the disabled attention plugin.
Its runtime shape/launch boundary now rejects mismatched K/V and output shapes,
unsupported dtypes/formats, invalid or non-narrowable dimensions, and missing
buffers before launching CUDA. Five descriptor tests cover the contract; four
negative tests failed before the fix. This is not a claim that the experimental
kernel is ready for general deployment.

`151344829` extends that regression coverage to nine descriptor tests,
including null/count arguments and missing enqueue buffers. It changes tests
only; the compiler and dispatch binaries used for comparison are unchanged.

The historical report in `gemma4_12b.md` is also corrected:

- Some four-sample means included `rep=-1`, the declared warmup. The final two
  historical runs are 2402.1 / 75.59 and 2424.3 / 73.97 tok/s after excluding
  warmup (PP / TG, three samples each).
- Slow historical whole-process runs were repeated rather than included in the
  headline average. All valid review runs are retained, irrespective of speed.
- Prefill/decode weight and activation labels were reversed. The final prefill
  activation requirement is 392,871,168 bytes; decode requires 81,209,344 bytes.
- The experiment's 40.4 tok/s end-to-end result belongs to an earlier kernel
  version. Its final standalone microbenchmarks are not a final end-to-end test.

## Verification protocol

Host: RTX 5080, driver 596.49, TensorRT-RTX 1.5.0.114, CUDA 12.9.86, WSL.
The host `benchmark` command passed its strict preflight and selected High
Performance with temperature-responsive ASUS Profile 1. Protected applications
and remote-access services were not manipulated. Per-run idle checks and any
reviewed exceptions are recorded separately. GPU workloads run serially.

The same 6,883,278,368-byte INT4 LiteRT-LM checkpoint, BF16 activation setting,
FP16 KV caches, GEMV mode and signature selection are used before and after.
Only predeclared warmups are excluded:

- Long context: populate an untimed 32,768-token prefix for each sample, then
  time PP2048 or TG128, chunk 128, three samples after `rep=-1` warmup. The
  harness checks the populated KV depth and actual generated token count.
- Short context: capacity 2048, PP1024/TG256, chunk 1024, eight iterations,
  average the last six. This is not an ISL=2000 workload.
- Output: a separate expected-output check for the Paris prompt.
- Memory: separate instrumented processes, 20 ms process-RSS polling and
  100 ms smaps/device sampling. Profiling runs are not pooled with throughput
  runs. `/usr/bin/time -v` reports OS lifetime CPU RSS HWM, not a sampled maximum.

AOT hits, runtime-cache outcomes, compilation markers, binary hashes, command
lines and exit codes are retained. AOT-cache warmth does not guarantee equal
OS page-cache residency. No global page-cache drop was performed. The 25 GiB
process-scope memory limit and 8 GiB swap limit are common to the runs.

All 16 pre-existing focused tests pass before and after: 12 graph-builder
tests (numerical and admission checks), one subbyte GEMV CUDA test and three
compiler/AOT tests. The nine descriptor tests bring the after-refactor total
to 25.

Fresh after-refactor compilation reproduces the previous 32K engine plan sizes,
refit counts, activation requirements and runtime FNV-1a content hashes:

| Engine | Plan bytes | Activation bytes | Runtime content hash |
| --- | ---: | ---: | --- |
| prefill_128 | 29,641,980 | 392,871,168 | `e7cf022fbf892ee2` |
| decode | 5,978,820,804 | 81,209,344 | `db04bd4fd89317be` |

These are content-hash checks, not a cryptographic proof. The stripped prefill
plan size excludes its external refit weights. Dispatch shares one activation
arena sized for the largest engine; the two activation requirements are not
additive resident allocations.

Measurements and commands are under `results/review_20260911/`; its README
records the run inventory and the unintended original-code cold-cache run.
That run completed the model workload but failed the wrapper's warm-hit
assertion and is not included in the warm comparison.

## Performance results

All numbers below are tokens/second, with the predeclared warmups excluded.
Each 32K row averages three measured samples. Each short-context row averages
the last six of eight iterations. These are process-level results, not
statistically independent model evaluations.

| 32K populated-prefix run | PP2048, chunk 128 | TG128 |
| --- | ---: | ---: |
| Original guard-free implementation, `original_long_2` | 1485.58 | 51.54 |
| Claude state, `before_long_1` | 2596.84 | 79.69 |
| Claude state, `before_long_2` | 2570.84 | 79.71 |
| Refactor, `after_long_1` | 2573.07 | 78.14 |
| Refactor, `after_long_2` | 2579.48 | 79.48 |

The two-process means are 2583.84 / 79.70 before cleanup and 2576.28 / 78.81
after: -0.29% / -1.11%. Compared with the single original-code warm process,
the after mean is +73.4% prefill and +52.9% decode. This confirms the broad
optimization benefit, not a precise per-commit causal decomposition.

| Short-context run, capacity 2048 | PP1024, chunk 1024 | TG256 |
| --- | ---: | ---: |
| Original, `original_short_1` | 3715.53 | 63.87 |
| Claude state, `before_short_1` | 3980.64 | 97.87 |
| Claude state, `before_short_2` | 3928.78 | 98.11 |
| Refactor, `after_short_1` | 3985.30 | 97.64 |
| Refactor, `after_short_2` | 3280.20 | 94.79 |

The last row is a genuine slower process and is retained. Averaging these
two regular short-context processes gives 3954.71 / 97.99 before and
3632.75 / 96.22 after. It would be misleading to quote only the fast after
run or claim uniformly stable performance from this batch.

### Identical-artifact/cache and mixed-library controls

Both sides next used the exact same AOT directory and separate copies of the
same accepted runtime-cache snapshot. The snapshot was deliberately taken
from the slow run rather than selecting a faster cache. Then each library was
swapped independently. The four final controls also used the same lightweight
500 ms NVML telemetry collection; their measurements are a separate cohort
from the earlier uninstrumented processes.

| Control | Compiler | Dispatch | PP1024 | TG256 | Init, seconds |
| --- | --- | --- | ---: | ---: | ---: |
| `after_control_short` | After | After | 3284.47 | 87.10 | 48.22 |
| `before_control_short` | Before | Before | 3971.63 | 96.77 | 38.84 |
| `control_mix_compiler_after` | After | Before | 3981.68 | 97.51 | 9.86 |
| `control_mix_dispatch_after` | Before | After | 3992.26 | 97.60 | 9.22 |
| `control_after_repeat` | After | After | 3984.80 | 97.65 | 9.69 |
| `control_before_repeat` | Before | Before | 3979.83 | 97.72 | 9.40 |

The final matched pair differs by +0.12% prefill and -0.08% decode. Both mixed
combinations are also fast. Thus no consistent slowdown follows either new
library, but the earlier slow processes have not been causally explained.
The appropriate conclusion is preservation in the controlled repeat, with
an unresolved process-state variability caveat, not an unconditional
performance-stability guarantee. Initialization varied with file residency
and runtime-cache outcomes; it is not a clean compiler-speed comparison.

Additional independent binary checks narrow the possible explanation:

- The before/after CUDA fatbin sections are byte-identical in both libraries.
- After normalizing relocation targets, default dispatch host instructions
  match. Differences are confined to the disabled attention validation path.
- Fast/slow serialized caches contain identical embedded cubin strings and
  kernel names. Differences occur only in small kernel-parameter blobs.

These checks do not prove identical execution scheduling or GPU residency and
are not substitutes for the measured results. No binary-layout tuning or
speculative runtime workaround was made to chase a fast sample.

Both before/after Paris expected-output checks pass. This is a smoke test,
not a full quality evaluation.

### Cold initialization

Fresh after-refactor AOT generation also completed for both shapes:

| Configuration | Total initialization | SDK engine-generation time, summed | CPU lifetime HWM |
| --- | ---: | ---: | ---: |
| 32K prefix / chunk 128 | 476.19 s | 220.39 s | 20,467.6 MiB |
| Capacity 2048 / chunk 1024 | 372.45 s | 144.42 s | 22,577.7 MiB |

The total includes graph preparation, compilation, artifact persistence,
loading/refitting and context setup. The SDK's engine-generation messages
measure a narrower part of that interval. These runs use memory profiling and
are not pooled with unprofiled throughput or used to claim a refactor-induced
compilation-memory reduction.

### Warm memory, separated by runtime-cache outcome

The fair cache-hit comparison is `before_memory` versus
`after_memory_cache_hit`. Both use the same AOT engine files and successfully
load the runtime cache. Values are MiB:

| Measurement | Before cleanup | After cleanup |
| --- | ---: | ---: |
| CPU lifetime RSS HWM, OS-reported | 6429.7 | 6417.2 |
| CPU RSS after prefill artifact unmaps | 695.4 | 693.3 |
| CPU RSS after decode artifact unmaps | 730.4 | 717.2 |
| CPU RSS at first prefill invocation end | 750.4 | 737.3 |
| CPU RSS at first decode invocation end | 1205.4 | 1197.0 |
| NVML device-wide baseline | 910 | 910 |
| NVML device-wide sampled peak | 13559 | 13559 |
| Sampled device usage above baseline | 12649 | 12649 |
| Process swap, sampled maximum | 0 | 0 |

The small RSS differences do not represent a new memory optimization; this is
a structural refactor. Invocation rows are checkpoint snapshots, not stage
high watermarks. OS lifetime HWM is not reset at stage boundaries. NVML values
are sampled device-wide usage, not an exact per-process allocation inventory
or the same counter as `cudaMemGetInfo` total-minus-free.

The earlier `after_memory` run rejected its prefill runtime cache and reached
6881.8 MiB CPU HWM, with 1665.2 MiB RSS at first decode invocation end. That
roughly 450 MiB increase is consistent with runtime-cache regeneration: the
unmodified build also reaches about 6878 MiB HWM when its cache is rejected.
Comparing a before hit with an after regeneration would incorrectly attribute
the difference to the refactor. The accepted-cache repeat above avoids that
confound. Memory-sampled initialization times varied widely (70.8 and 110.6 s
for the two accepted-cache profiles) and are not throughput measurements.

## Remaining correctness and deployment limits

There is also a runtime-cache reload problem already present in Claude's
reviewed state, distinct from an AOT miss. Several before/after processes
report Myelin deserialization errors, ignore the runtime cache and continue.
The refactor does not change runtime-cache code.

A standalone test now reproduces this inside TensorRT-RTX 1.5.0.114. It uses
no LiteRT code, creates no execution context, runs no inference, retains the
input buffers and performs no cache-file writes. The SDK accepts the input,
serializes it, then immediately rejects its own output in a fresh cache:

| Cache input | Accepted input bytes | Re-serialized bytes | Fresh deserialize |
| --- | ---: | ---: | --- |
| Short-context prefill | 1,533,940 | 1,530,104 | Fails |
| 32K prefill | 1,667,894 | 1,662,778 | Fails |
| Decode control | 696,598 | 696,598 | Passes twice |

Previously rejected prefill files are also rejected by this standalone test.
This isolates a persistence defect in the installed SDK, not an input-buffer
lifetime or filesystem-write failure in LiteRT. The observed missing
block-scale-dequantization metadata is a lead for NVIDIA, not a complete
decoding of the proprietary cache format. Source, build instructions, accepted
and rejected snapshots, and logs are retained under
`results/review_20260911/runtime_cache_roundtrip*` and `sdk_roundtrip_*.log`.

Cache rejection explains regeneration, but does not alone establish the cause
of a slower steady-inference process. All such runs remain in the report;
their startup times must not be presented as uniform runtime-cache hits.
No speculative production cache workaround or SDK replacement was added.

The numerical tests and Paris check are useful regression checks, not a
perplexity evaluation or proof of quality equivalence over arbitrary prompts.
Precision/fusion changes warrant a longer teacher-forced/logit comparison
before a quality-equivalence claim.

`OnlyTransposedMatmulReaders` checks local graph uses, but the optimized V-cache
layout also depends on matching prefill/decode consumers. A mixed CPU/native
consumer or separately compiled signature needs an explicit layout contract
before this becomes a generic lowering. The tested Gemma signature pair uses
matching layouts; this review does not disable its optimization.

The experimental attention launcher's per-instantiation `static bool
attribute_set` is not CUDA-context/device-aware and is not synchronized for
concurrent host callers. Keep the experiment off until its lifecycle and
concurrency contract, full numerical coverage and end-to-end benefit are
validated. Descriptor checks are not exhaustive arbitrary-shape safety: the
existing CUDA split-count expression can overflow near INT32_MAX sequence
length, far beyond the measured allocations. Its current rows<=16 scope
cannot address large-chunk prefill.

## Applying NVIDIA's Edge-LLM guidance

The public Edge-LLM stack is a separate TensorRT Enterprise deployment, not a
mode of the installed RTX SDK. Its public documentation lists Gemma 4 12B and
x86-64 Linux as a developer platform. That does not establish WSL release
qualification or public native-Windows support. No Edge-LLM installation,
conversion or benchmark was performed here.
[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html),
[support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html),
[models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html).

Source inspection is pinned to NVIDIA/TensorRT-Edge-LLM
`e8b29522938901f6df19ebeedd4b69bc8edbcd97`.

### 1. Fused prefill attention, followed by decode attention

`CuteDslFMHAV2Runner` explicitly admits SM120/121 and supports FP16 causal/local
attention with head dimensions 256 and 512. The other Blackwell FMHA runner is
not the correct blanket choice: its SM100/101/110 path excludes the 5080.
`DecoderXQARunner` also selects a D512 two-CTA variant for SM120/121.
[FMHA-v2](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp),
[XQA](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/decodeAttentionKernels/decoderXQARunner.cpp).

Prototype their raw-pointer launchers for our actual shapes, then integrate
validated cases through RTX `IPluginV3`. Fuse QK, masking, softmax and PV so the
full attention-score matrix is not materialized. Validate BF16/FP16 conversion,
query-head folding, scaling, local/causal masks, valid cache lengths and stream
ownership. Our matched tensors have already undergone query/key processing
and KV update: importing the entire Edge attention plugin could apply RoPE or
cache updates twice. Benchmark both 128 and 1024 chunks at 32K before changing
defaults. Source-level support is not a demonstrated speedup here.

The adapter needs more than matching pointer types. The supplied FMHA-v2
wrapper constructs contiguous `[B,S,H,D]` descriptors, unlike our physical
cache layouts. Establish compatible strides before introducing full-cache
transposes that could erase the benefit. XQA's actual compiler rejects BF16
Q/output; D256 accepts GQA ratios 2/4/6/8 and D512 accepts 2/4/8/16. Its
two-CTA variant also checks device capabilities. Recognized causal/local masks
are not interchangeable with arbitrary Boolean masks. Paged attention needs
its own cache/page-table ABI and is a separate integration milestone.
[FMHA descriptors](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp#L285-L348),
[XQA restrictions](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp#L250-L295).

The shipped FMHA generation uses an approximate near-one online-softmax
rescale. Numerical evaluation should compare exact-rescale and shipped
configurations, not attribute all differences to FP16 rounding. XQA's NVRTC
compilation occurs during engine building and its cubins are serialized for
runtime loading; it need not add inference-time NVRTC compilation.
[FMHA generation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/build_cutedsl.py#L666-L710),
[rescale approximation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/fmha_v2_cutedsl/fmha.py#L2272-L2284),
[XQA lifecycle](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/xqa/README.md).

The pinned artifact instructions explicitly list an x86-64 SM120 CUDA12
package. CUDA 12.9 is therefore not by itself a source-level blocker or a
reason to replace the host toolkit with CUDA13. Actual kernel/RTX build,
linkage and loading still require verification in an isolated probe.
[Kernel artifacts](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/README.md#L86-L99).

### 2. One GPU weight layout shared by prefill and decode

Edge-LLM's `ExternalWeightManager` manages final-layout GPU weights and binds
their addresses to engine inputs. Its INT4 V2 plugin dispatches small M to
GEMV and larger M to GEMM with the same fragment-packed weights and scales.
This solves the consumer-layout problem that host refit deduplication alone
cannot solve.
[Weight manager](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/runtime/state/externalWeightManager.h),
[INT4 plugin](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/plugins/int4GroupwiseGemmPluginV2/int4GroupwiseGemmPluginV2.cpp),
[shared-layout GEMV](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/plugins/int4GroupwiseGemmPluginV2/cuteDslInt4Gemv.h).

Our installed RTX headers expose `IPluginRegistry::acquirePluginResource` /
`releasePluginResource` and `IExecutionContext::setInputTensorAddress`; those
are usable ownership/binding building blocks, not automatic native-layer
weight sharing. First establish performant compatible GEMV/GEMM consumers,
then use device/shape/dtype/quantization-aware ownership and lifetimes. The
existing native-input INT4 GEMM probe lost fusion (1244 us versus 143 us at
M=128), so simply externalizing native weights is not acceptable. Edge's
FP16 fragment/group-scale contract differs from our BF16 row-packed GEMV.
Sharing weights would still leave KV caches, activation arenas and contexts.

Specifically, the supplied GEMV/GEMM artifacts bake group size 128 and require
their fragment packing, FP16 activation/scale/output representation and K
alignment. Per-channel scales need explicit conversion or new variants. The
small-M crossover and reduced tactic set also need measurement on the 5080.
[INT4 variants](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/build_cutedsl.py#L1728-L1797).

Edge itself uses one base executor with prefill/decode optimization profiles;
its convenience weight-manager binding rejects a second bind. It is not an
out-of-box cross-engine deduper. Reusing the design in our two separately
compiled LiteRT signatures requires integration-owned shared allocation
lifetime and per-context binding, or a larger unified-engine redesign.
[Profile selection](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/runtime/llmRankRuntime.cpp#L64-L68),
[Binding contract](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/runtime/state/externalWeightManager.cpp#L935-L946).

### 3. A separate, matched deployment benchmark

NVIDIA's claimed INT4-AWQ/llama.cpp parity at ISL=2000 is useful motivation, not
proof of 32K or prefill/memory parity. Match batch, ISL/OSL, KV type, chunk,
tokenization, warmup, greedy/speculative mode and host state; report full-prefix
latency separately from depth-benchmark continuation speed. AWQ INT4, GGUF
Q4_0 QAT, the LiteRT per-channel INT4 checkpoint and NVFP4 differ in scales,
packing and model quality. NVFP4 needs a new quantized conversion, not a switch
that reinterprets existing INT4 bytes.

Use an isolated Enterprise SDK if pursuing that benchmark, preserving the RTX
SDK and verified artifacts. Kernel availability, SDK build success, correct
execution, model quality and measured end-to-end improvement are separate
verification milestones.
