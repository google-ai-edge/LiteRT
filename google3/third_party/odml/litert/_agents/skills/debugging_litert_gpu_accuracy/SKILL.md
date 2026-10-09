---
name: debugging-litert-gpu-accuracy
description: >-
  Diagnoses and fixes on-device GPU numerical corruption, garbage generation,
  zeroed KV caches, and ML Drift delegate lowering bugs for LiteRT and
  LiteRT-LM models on Android (OpenCL and WebGPU). Use when a model produces
  correct output on CPU but generates garbage, NaNs, or wrong outputs on GPU;
  when comparing CPU vs GPU execution op-by-op or state-buffer-by-state-buffer;
  or when debugging composite ops (cache_update, runtime_bmm, sdpa), ring-buffer
  sliding-window KV caches, or TensorStorageType (BUFFER vs TEXTURE_2D)
  mismatches in ml_drift. Don't use for NPU/QNN dispatch issues, CPU-only model
  conversion failures, or adding standard E2E device test targets (use
  adding-litert-lm-e2e-tests).
---

# Debugging LiteRT / LiteRT-LM On-Device GPU Accuracy & ML Drift Bugs

Use this 6-phase funnel to isolate why a LiteRT-LM model produces correct
output on CPU (`--backend=cpu`) but generates garbage, NaNs, or empty/corrupted
state on Android GPU (`--backend=gpu`, OpenCL or WebGPU).

```
Phase 1: Unpack & diff model metadata (.litertlm) against a GPU-good model
   │
Phase 2: On-device repro matrix (CPU vs OpenCL vs WebGPU + prompt sweep)
   │
Phase 3: Op-by-op accuracy_debugger on extracted TFLite subgraph
   │     ├─ Op diverges in isolation ──► Fix op kernel / precision
   │     └─ All ops pass (maxdiff≈0) ──► Composite / graph-lowering / state bug
   ▼
Phase 4: Host-side CPU vs GPU state-buffer dump after prefill_128 & decode
   │     (Pinpoints exact corrupted KV cache layers: e.g. local ring vs global)
   ▼
Phase 5: ML Drift descriptor dump & in-shader sentinel readback probes
   │     (Bisects runtime param chain & TensorStorageType mismatches)
   ▼
Phase 6: Fix in ml_drift / delegate, unit-test GpuModel invariants, & validate
```

--------------------------------------------------------------------------------

## Critical Environment Gotchas (Read First)

1.  **Delete the on-device ML Drift program cache before every GPU run after
    editing any kernel, parser, or selector code.** Otherwise the device
    silently reuses the cached compiled OpenCL/WebGPU program and your C++ edits
    never execute:

    ```bash
    adb shell "rm -f /data/local/tmp/{model_dir}/*mldrift_program_cache.bin"
    ```

    Keep `*_mldrift_weight_cache.bin` and `*.xnnpack_cache` so weight conversion
    stays fast.

2.  **Injecting device environment variables:** `run_litert_lm.sh` does not
    forward custom host env vars or provide a build-only flag. Run
    `run_litert_lm.sh --skip_model_push` once to build and push
    `libLiteRtOpenClAccelerator.so` and `litert_lm_advanced_main`, then invoke
    `adb shell` directly with your probe env vars:

    ```bash
    OUT=/data/local/tmp/{model_dir}
    adb shell "rm -f ${OUT}/*mldrift_program_cache.bin"
    adb shell "ADSP_LIBRARY_PATH=${OUT} LD_LIBRARY_PATH=${OUT} \
      LITERT_LM_DEBUG_DUMP_STATE=1 MLDRIFT_CACHE_PROBE=1 \
      ${OUT}/litert_lm_advanced_main \
      --model_path=${OUT}/model.litertlm --backend=gpu \
      --async=false --clear_kv_cache_before_prefill=true \
      --convert_weights_on_gpu=true --minloglevel=0 \
      --input_prompt=\"hi\""
    ```

3.  **Layering check in `third_party/ml_drift` and `delegate/composite`:**
    Adding `#include "third_party/absl/log/absl_log.h"` to kernel files fails
    Bazel/Blaze `layering_check`. Use `#include <cstdio>` + `fprintf(stderr,
    ...)` and `#include <cstdlib>` + `std::getenv(...)` for temporary probes.

--------------------------------------------------------------------------------

## Phase 1: Unpack & Diff Model Metadata Against a Known-Good Model

Unpack the failing `.litertlm` bundle and a known-good GPU model to compare
their `ExecutorMetadataProto`, `LlmMetadataProto`, and embedded `.tflite`
subgraphs:

```bash
/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents run \
  //third_party/odml/litert_lm/runtime/util:litertlm_builder_cli -- \
  unpack --input_file=/tmp/dbg/failing.litertlm --output_dir=/tmp/dbg/unpacked
```

Check `ExecutorMetadataProto.pbtext` and `LlmMetadataProto.pbtext` for:

-   **Hybrid KV cache topology**: Does the model mix `TYPE_LOCAL_*_CACHE`
    (sliding-window / ring buffer, e.g. `minimum_sequence_length: 1024`,
    `maximum_sequence_length: 1280`) with global caches
    (`maximum_sequence_length: 32771`)?
-   **`attention_mask_settings`**: Presence of `sliding_window_size` triggers
    the `gpu_optimized_single_buffer_cache_` ring-buffer path in LiteRT-LM and
    `ring_buffer_sdpa` in the exported graph.

--------------------------------------------------------------------------------

## Phase 2: Reproduce & Narrow the Failure Surface on Device

Run the repro prompt across backends and configurations to eliminate whole
subsystems up front:

1.  **CPU vs OpenCL GPU vs WebGPU:**

    ```bash
    # OpenCL GPU
    ./third_party/odml/litert_lm/runtime/engine/run_litert_lm.sh \
      --models={model_name} --target_os=android --backend=gpu \
      --skip_model_push --input_prompt="{prompt}"

    # WebGPU (--use_webgpu)
    ./third_party/odml/litert_lm/runtime/engine/run_litert_lm.sh \
      --models={model_name} --target_os=android --backend=gpu --use_webgpu \
      --skip_model_push --input_prompt="{prompt}"
    ```

    -   If **both OpenCL and WebGPU fail identically**, the bug is in
        backend-agnostic code: the exported graph, LiteRT-LM host tensor
        filling, or shared `ml_drift/common` / `delegate/composite` graph
        lowering (such as `LiteRtOpSelector`), **not** an OpenCL-specific driver
        or shader bug.

2.  **Prompt-length sweep:** Test a minimal prompt (`"hi"` — note that system
    prompt wrapping still yields ~14 tokens) vs >128 tokens (multi-chunk
    prefill) vs >`sliding_window_size`. If even a 14-token prompt fails on the
    very first generated token, wrap-around and multi-chunk boundary math are
    ruled out.

--------------------------------------------------------------------------------

## Phase 3: Run Op-by-Op `accuracy_debugger` (and Know Its Blind Spots)

Run `accuracy_debugger` on the unpacked
`Section3_TFLiteModel_tf_lite_prefill_decode.tflite` (`prefill_128` subgraph) to
compare GPU vs CPU reference op-by-op:

```bash
./litert/tools/accuracy_debugger/google/debug_accuracy_qc.sh \
  --model_path=/tmp/dbg/unpacked/Section3_TFLiteModel_tf_lite_prefill_decode.tflite \
  --signature_name=prefill_128 \
  --num_ops=400
```

Inspect the generated CSV (`Index, Op Code, Tensor Name, Max Diff, MSE, Cos Sim,
SNR, PSNR, NaN, Min Ref, Max Ref, Min Accel, Max Accel, Status`):

-   **If a standard op shows low `Cos Sim` / high `Max Diff` / `NaN`**: You have
    an isolated op numerical/precision bug.
-   **Blind spot to remember**: `accuracy_debugger` verifies each op in
    isolation and reports `ACCEL_COMPILE_FAILED` on
    `shlo.composite(odml.runtime_bmm)` and `shlo.composite(odml.cache_update)`.
    Consequently, an entire parameter computation chain (e.g. `SLICE` →
    `FLOOR_MOD` → `CONCAT`) can pass with `maxdiff = 0` in `accuracy_debugger`
    while still producing garbage in the full compiled graph due to **cross-op
    storage-type / producer-rewiring bugs** triggered only when composite ops
    are lowered alongside standard ops.

--------------------------------------------------------------------------------

## Phase 4: Diff CPU vs GPU State Buffers After `prefill_128` & `decode`

When `accuracy_debugger` shows standard ops are healthy, instrument
`google3/third_party/odml/litert_lm/runtime/executor/llm_litert_compiled_model_executor.cc`
(see [references/mldrift_shader_probes.md](references/mldrift_shader_probes.md)
for the drop-in snippet) to dump `n`, `sum`, `sumabs`, `min`, `max`, `nan`, and
the first 6 elements (`head=[...]`) of every input and output buffer in
`BindTensorsAndRunPrefill` and `BindTensorsAndRunDecode` under `--async=false`.

Run once with `--backend=cpu` and once with `--backend=gpu` on `"hi"` and diff:

-   **Check `param_tensor` (`prefill_in`)**: Verify the 7-channel `int32`
    runtime parameter tensor `[start, end, end, ...]` matches on host.
-   **Check `kv_cache_k_{layer}` / `kv_cache_v_{layer}` (`prefill_out`)**:
    -   If **global layers** (e.g. layers 3, 7, 11, ...) match CPU
        (`sumabs > 0`) while **local/sliding-window layers** (e.g. layers 0, 1,
        2, ...) have `sumabs = 0` (`min=0, max=0`), the ring-buffer branch of
        `add_values_to_cache_kernel.cc` is early-exiting without writing.

--------------------------------------------------------------------------------

## Phase 5: Bisect with ML Drift Descriptor & Shader Readback Probes

Understand the parameter contract before probing:

| Slot | Meaning | Consumer |
| :--- | :--- | :--- |
| `0` | `token_index_offset` (`start` or `start % S`) | `add_values_to_cache` (both branches) |
| `1` | `active_tokens` (`end` or `min(end, S)`) | `add_values_to_cache` non-ring bound |
| `2` | `kActiveTokensAlignedIndex` | `runtime_batched_matmul` (`src_end_ch_index` / `dst_end_ch_index`) & `sdpa_transposed` |
| `3` | Ring `update_length` (`end - start`) | `add_values_to_cache` **ring branch only** (`if (X >= update_length) return;`) |

Use two targeted probes (full code in
[references/mldrift_shader_probes.md](references/mldrift_shader_probes.md)):

1.  **In-shader param readback (`MLDRIFT_CACHE_PROBE`)**: In
    `ml_drift/delegate/composite/add_values_to_cache_kernel.cc`,
    bypass the `X >= update_length` gate and overwrite `final_value_k` with
    `float4(params.Read(0..3))` and `final_value_v` with
    `float4(params.Read(4..7))`. The host `[DBG]` dump of
    `kv_cache_k_0 head=[...]` and `kv_cache_v_0 head=[...]` then directly prints
    the exact `int32` values the GPU shader read from `args.params`!
2.  **Host-side `OperationDef` / `TensorStorageType` dump
    (`MLDRIFT_SLICE_PROBE`)**: In
    `google3/third_party/ml_drift/common/kernels/strided_slice.cc` (inside
    `GetStridedSliceCode`), print `starts.c`, `ends.c`, `src.Channels()`, and
    `src.GetStorageType()` (`1 = BUFFER`, `3 = TEXTURE_2D`) for small int32
    tensors (`c <= 16`).

### Classic Root-Cause Signature (`b/565413009`)

If `MLDRIFT_CACHE_PROBE` shows `params[0..2]` are valid (`0, 14, 14`) while
`params[3..6]` are garbage (`±inf` in fp16), and `[SLICEDBG]` shows:

```text
starts.c=0 ends.c=3 | src c=7 storage=1 (BUFFER)      -> valid
starts.c=3 ends.c=7 | src c=7 storage=3 (TEXTURE_2D)  -> GARBAGE (unproduced tensor!)
```

Look at `LiteRtOpSelector::ParamTensorToBuffer` in
`ml_drift/delegate/composite/litert_op_selector.cc`:

-   Composite ops (`add_values_to_cache`, `runtime_batched_matmul`,
    `sdpa_transposed`) declare `params` via `AddSrcBuffer` (`BUFFER` storage),
    whereas standard ops like `StridedSlice` (`rest = param'[3:7]`) consume the
    original tensor via `AddSrcTensor` (`TEXTURE_2D`) without consulting
    `replaced_tensors_`.
-   Calling
    `model_builder->UpdateOutputTensor(param_tensor, new_param_tensor.id)`
    **re-points the producer node's output** from `param_id` to
    `new_param_tensor.id`, leaving `param_id` with **no producer in the
    `GpuModel`** so any non-buffer consumer reads uninitialized GPU memory.
-   **Correct fix**: Emit `model_builder->Copy(param_tensor, new_param_tensor)`
    instead of `UpdateOutputTensor`. When `param_tensor` has no other consumers,
    `LinkNodes()` in `google3/third_party/ml_drift/common/merge_nodes.cc`
    automatically fuses the single-consumer copy back into the producer.

--------------------------------------------------------------------------------

## Phase 6: Unit-Test the `GpuModel` Invariant & Validate End-to-End

1.  **Revert all temporary probes** (`hg revert` on kernel/executor files)
    before running final validation.
2.  **Write a hermetic `cc_test`** (see
    `ml_drift/delegate/composite/litert_op_selector_test.cc`)
    that builds a `GpuModel` via `GpuModelBuilder` + `LiteRtOpSelector` and
    asserts that **every `in_id` of every `GpuNode` in `gpu_model.nodes` is
    present in `defined_tensors`** (graph inputs ∪ const tensors ∪ node
    outputs):

    ```bash
    /google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test \
      //ml_drift/delegate/composite:litert_op_selector_test
    ```

3.  **Validate on device** across:
    -   Failing model on OpenCL (`--backend=gpu`)
    -   Failing model on WebGPU (`--backend=gpu --use_webgpu`)
    -   A baseline non-sliding-window model
        (`--models=kanana_i4_107_kv_1024 --backend=gpu`)

--------------------------------------------------------------------------------

## References

-   **Drop-in C++ probes (Executor state dumper, Shader param readback, Slice
    storage dumper)**:
    [mldrift_shader_probes.md](references/mldrift_shader_probes.md)

## Contributions

To contribute or modify this skill or its references, follow the
[contribution guidelines](references/contributing.md) before making changes.
