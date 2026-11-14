<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Arm Vulkan ML developer tool

`litert_arm_vulkan_ml_tool` is a C++ diagnostic CLI for the direct Arm Vulkan ML
backend. It uses the production operator analysis, partitioner, lowering and
Vulkan dispatch. Dataset preparation, image preprocessing and end-to-end vision
model evaluation live in the shared
[`litert_vision_eval`](../../../../tools/vision_eval/README.md) runner.

| Command | Purpose | Vulkan device needed |
| --- | --- | --- |
| `inspect` | Operator coverage, rejection reasons, tensor types/shapes/quantization and selected partitions | No |
| `compile` | Lower selected partitions, validate SPIR-V, export assembly, constants, operator/result-ID mappings and optional VGF | No |
| `capabilities` | Query extensions/revisions, feature bits, tensor limits/formats and TOSA engine profiles/extensions | Yes |
| `validate` | Require complete coverage, validate SPIR-V and compare GPU inference against CPU reference | Yes, with ML support |
| `trace` | Find the first intermediate output outside tolerance using CPU reference | Yes, with ML support |
| `coverage` | Show registered TFLite operator lowering subsets | No |

## Build

From the repository root:

```sh
mkdir -p .bazelisk-cache .cache .bazel-output
XDG_CACHE_HOME="$PWD/.cache" BAZELISK_HOME="$PWD/.bazelisk-cache" \
  bazelisk --output_base="$PWD/.bazel-output" build -c opt \
  --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
  //litert/vendors/arm_vulkan_ml/tools:litert_arm_vulkan_ml_tool \
  //litert/vendors/arm_vulkan_ml/compiler:compiler_plugin_so \
  //litert/vendors/arm_vulkan_ml/dispatch:dispatch_api_so
export TOOL="$PWD/bazel-bin/litert/vendors/arm_vulkan_ml/tools/litert_arm_vulkan_ml_tool"
```

SPIRV-Tools is pinned as a tool-only dependency. The production compiler and
dispatch libraries do not depend on it.

For Android arm64, add the following flags, using your installed SDK/NDK paths
and a separate Bazel output base:

```sh
--config=android_arm64 \
--repo_env=ANDROID_SDK_HOME=/path/to/android_sdk \
--repo_env=ANDROID_NDK_HOME=/path/to/android_sdk/ndk/30.0.16248370 \
--repo_env=ANDROID_NDK_VERSION=30 \
--repo_env=ANDROID_SDK_API_LEVEL=35 \
--repo_env=ANDROID_NDK_API_LEVEL=35 \
--repo_env=ANDROID_BUILD_TOOLS_VERSION=35.0.0
```

Use adb to place the executable, model and inputs on the device. For `validate`,
also deploy `libLiteRtCompilerPlugin_ArmVulkanML.so` and
`libLiteRtDispatch_ArmVulkanML.so` and pass their directories below. The executable
uses Android's `libvulkan.so`; Linux uses `libvulkan.so.1`. `inspect` and `compile`
work on Android devices without Arm ML extensions.

## Inspect and compile

```sh
"$TOOL" inspect --model model.tflite --report_txtpb inspect.txtpb
"$TOOL" compile --model model.tflite --output_dir compiled \
  --report_txtpb compile.txtpb
```

The TextFormat report includes original zero-based subgraph, operator and tensor
indices. Partial coverage remains explicitly marked `coverage_status: "partial"`;
a successful `inspect` or `compile` does not imply full-model execution support.
Unsupported operators include the production rejection reason. Pass
`--graph_view raw|effective` (default `effective`) to inspect the pre- or
post-legalization graph, and `--softmax_lut_type auto|int8|int16` (`auto` uses
`int8`) or `--rescale_multiplier_bits 16|32` (default `32`) to test lowering
variants.

Each selected partition exports `.spv`, `.spvasm`, a TextFormat manifest and binary
constant buffers. The lowering map associates newly created SPIR-V result IDs
with original operators. Shared IDs belong to their first creating operator;
module scaffolding and final output wiring have no operator attribution. These
are diagnostic artifacts, not a serialized executable accepted by dispatch.
SPIR-V validation uses the universal 1.6 environment and is not proof of Vulkan
pipeline acceptance or hardware conformance.

## Dump lowered partitions as VGF

```sh
"$TOOL" compile --model model.tflite --output_dir compiled \
  --dump_vgf --report_txtpb compiled/compile.txtpb
```

This additionally creates `subgraph_<N>_partition_<M>.vgf` for every accelerated
partition. Each container embeds its graph SPIR-V, exact constant payloads,
entry point, descriptor bindings, physical tensor formats/shapes and IO ordering.
IO names are `input_0`, `input_1`, etc. and `output_0`, `output_1`, etc.; the TextFormat
manifest preserves the original LiteRT operator/tensor mapping. The manifest also
records `vgf_file`, `vgf_sha256` and `vgf_vulkan_header_version` (matching the pinned
`VK_HEADER_VERSION`).
Keep the manifest alongside the VGF for profile/extension requirements.

VGF export runs offline on Linux or Android. The pinned Arm encoder writes VGF
format 0.4.0; its output was also decoded using Arm's SDK 0.11.0 `vgf_dump`.
Disable export with `--dump_vgf=0`, which is also the default. Each file contains
one accelerated partition. Consult `complete_coverage`/`coverage_status` before
using a dump as a complete model: CPU fallback and LiteRT orchestration are
outside the container. The production compiler/dispatch libraries do not depend
on the VGF encoder and continue using their JIT execution path.

For example, inspect the embedded SPIR-V with Arm's VGF inspection utility:

```sh
vgf_dump --input compiled/subgraph_0_partition_0.vgf \
  --dump-spirv 0 --output graph.spv
```

## Query the device

```sh
"$TOOL" capabilities --report_txtpb device.txtpb
"$TOOL" capabilities --model model.tflite --report_txtpb compatibility.txtpb
```

Without a model this inventories available Vulkan devices, including ordinary
GPUs lacking ML extensions. With a model it requires a single fully lowerable
subgraph and matches its requirements against each engine. Format flags describe
linear tensor storage. TOSA conformant profiles/extensions, extension revisions,
feature bits and actual model tensor limits are checked using dispatch's logic.
The compiler derives TOSA level requirements from the emitted graph, including
intermediate tensor byte sizes, effective dilated kernels, strides, resize
ratios and generated tensor lists. Dispatch checks these against the queried
engine level; the report identifies checks made against the emitted graph.
Pipeline acceptance remains `not_tested`. An operator's presence in the specification alone never proves
support for its types or shape on the device.

On commercially available MediaTek Dimensity 9600 retail phones purchased
through standard retail channels (`Arm Mali-G2-Ultra-NX`), `capabilities`
surfaces several hardware properties that differ from abstract `TOSA.001000.1`
bounds and host emulation: `PRO-INT` and `EXT-INT16` report
`ACCELERATED | CONFORMANT` (`VK_DATA_GRAPH_TOSA_LEVEL_8K_ARM`, `shaderInt16`
and `shaderInt64` enabled) whereas `PRO-FP` reports `CONFORMANT` without
`ACCELERATED` (shader fallback), `EXT-DOUBLEROUND` is unadvertised,
`VkPhysicalDeviceTensorPropertiesARM::maxTensorDimensionCount` is `4`, linear
`R8_SINT`/`R16_SINT`/`R32_SINT`/`R64_SINT`/`R32_SFLOAT` formats advertise
`VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM`, and `capabilities --model`
reports `importable = true` (`dedicated_only = false`) for linear
`AHardwareBuffer` and `DMA-BUF` IO tensors.

For Arm's Linux emulator, install the Arm ML emulation-layer package and set
`VK_DRIVER_FILES` to the software Vulkan ICD, `VK_LAYER_PATH` to the package's
`deploy/share/vulkan/explicit_layer.d`, and `LD_LIBRARY_PATH` to `deploy/lib`.
Set:

```sh
export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
```

Pass `--allow_emulation=1` explicitly to each execution/capability command. This
permits the emulator's CPU `Emulation Layer` profile advertisement for PRO-INT
and PRO-FP. Missing TOSA extensions are never waived. Emulator results establish
functional behavior only; they do not qualify native Arm GPU performance or
conformance.

## Vision dataset evaluation

End-to-end dataset evaluation and image preprocessing (ImageNet classification,
COCO/Pascal VOC/ADE20K segmentation, and NYUv2 depth/normals) are handled by
[`litert_vision_eval`](../../../../tools/vision_eval/README.md). To inspect or
trace a specific dataset sample in `litert_arm_vulkan_ml_tool`, run
`litert_vision_eval` with `--dump_inputs=true` to export raw input bytes
(`inputs/input_<INDEX>.raw`) and reference them in a `--tensor_inputs` manifest
below.

## Raw tensor validation

`validate` and `trace` accept `--tensor_inputs inputs.txtpb` (`TensorInputsManifest` protobuf TextFormat):

```textproto
cases {
  id: "broadcast"
  inputs { index: 0 file: "lhs.f32" }
  inputs { index: 1 file: "rhs.f32" }
}
```

`index` is the zero-based model input position, not the tensor ID. Supply every
input exactly once. Files contain contiguous little-endian raw bytes matching
each input tensor's dtype (`FP32`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT32`,
`BOOL` encoded 0/1 per byte, or `COMPLEX64`) and static model byte count. Paths
resolve relative to the TextFormat manifest. Linux and Android arm64 builds use
little-endian hosts. Constant INT32 control operands belong in the model.
Dynamic and scalar input interfaces remain unsupported.

```sh
"$TOOL" validate --model broadcast.tflite --tensor_inputs inputs.txtpb \
  --max_samples 3 \
  --compiler_plugin_library_dir bazel-bin/litert/vendors/arm_vulkan_ml/compiler \
  --dispatch_library_dir bazel-bin/litert/vendors/arm_vulkan_ml/dispatch \
  --report_txtpb comparison.txtpb
"$TOOL" trace --model broadcast.tflite --tensor_inputs inputs.txtpb \
  --mode isolated --report_txtpb isolated.txtpb
```

Validation checks every model output against TFLite CPU reference and runs
SPIR-V validation before pipeline creation. Defaults are up to three cases,
FP32 `--atol=1e-4 --rtol=1e-3`, INT8 `--max_code_error=4 --mean_code_error=0.5`,
and `--require_full_acceleration=1`. Use `--start_sample` to pick the starting
case index and `--max_samples` for batch size (1–10 cases per run), and
`--reference=tflite|fp64|xnnpack` (`fp64`/`xnnpack` support raw single-op `INT8`
`SOFTMAX`, and `fp64` also supports single-op `SQUARE` and `CONV_3D`). INT8 gates
apply in quantized code space; reports also retain dequantized errors, case IDs,
input hashes, per-output shapes/errors and partial results on failure.
`--runtime_library_dir` is available when built for dynamic LiteRT runtime
loading.

## Trace a mismatch

```sh
"$TOOL" trace --model model.tflite \
  --tensor_inputs inputs.txtpb --start_sample 0 \
  --mode accumulated --capture_budget_mb 256 --report_txtpb trace.txtpb
```

Trace processes one case (also the only accepted `--max_samples` value):

- `accumulated` executes successive original-model prefixes and compares each
  operator output with the CPU prefix. It stops at the first output outside the
  selected tolerance, even if later operations would hide that error.
- `isolated` feeds each Vulkan operator the corresponding CPU reference inputs.
  This distinguishes a local lowering error from error propagated by earlier ops.

Each pass exposes one operator's outputs. Captured input/output storage is
bounded by `--capture_budget_mb` (default 256 MiB); model weights and execution
workspace are outside that budget. Exposing outputs can change fusion and
optimization, so this is an instrumented diagnostic, not an unmodified driver
trace. Recompiling prefixes can be slow on large models. Unsupported operators,
resource exhaustion or pipeline failures report `incomplete` with the last
attempted operator, never a passing result. A numerical mismatch reports
`first_divergence` and exits unsuccessfully. Use `--max_code_error=0 --mean_code_error=0` for exact INT8 comparisons. Optional `--dump_tensors=1 --output_dir=trace-dumps` saves the failing CPU/Vulkan tensors as raw binary.

All commands print protobuf TextFormat and optionally save it with `--report_txtpb`. Reports
carry schema version, command, model/binary hashes when available and a status.
Exit codes are 0 for success, 1 for execution/report failures, and 2 for malformed
arguments. Detailed driver/compiler diagnostics go to stderr. Numeric Vulkan
format, quality and version fields retain the API values without reinterpretation.

## Vulkan validation diagnostics

Use `--vulkan_validation=core` or `--vulkan_validation=sync` with `capabilities`,
`validate` or `trace`. The default is `off`. Offline `inspect`/`compile` commands
reject a validation request because they do not create a Vulkan instance.
`sync` enables synchronization validation in addition to core checks.

The requested Khronos layer and required extensions must be available.
Layer binaries and build directories remain local development artifacts.

Reports separate `execution_status`, `numerical_status` and `vulkan_validation`.
The latter includes loaded layer version, baseline match, diagnostics, error and
warning counts. Every validation error fails the overall check; there is no
known-error allowlist. Missing layers, collection failures, truncated event
streams or missing teardown records fail with validation `incomplete`. Warnings
are retained without failing. Numerical status is `not_run` when execution
failed before comparison, and may be `incomplete` for partial tracing.

Capture is bounded to 4096 messages per instance. Each instance writes its own
JSON event stream, including creation and device teardown. The CLI collects
streams after runtime/plugin cleanup and removes its temporary directory.
Public LiteRT C APIs are unchanged. Direct backend diagnostic setup also uses
`LITERT_ARM_VULKAN_ML_VALIDATION` and
`LITERT_ARM_VULKAN_ML_VALIDATION_LOG_DIR`; the CLI scopes and restores these.

## Operator implementation coverage

Run `litert_arm_vulkan_ml_tool coverage` to inspect the registered TFLite
operator lowering subsets from `op_registry`, or add `--txtpb` and
`--report_txtpb FILE` for machine-readable output. No Vulkan loader is required.

Integer and Boolean outputs compare exactly in validation and both trace modes;
existing FP32 and quantized comparison rules still apply.

## External tensor memory

With `capabilities --model FILE`, each device includes `external_tensors` entries
for model inputs/outputs, with DMA-BUF/AHardwareBuffer importability and rejection
reasons. These describe the tensor configuration; attachment must still validate
the actual allocation, memory type, size, alignment and dedicated-allocation rules.

Direct execution operator records include `buffer_io.input_copy_bytes`,
`output_copy_bytes` and `imported_slots`. Counts describe explicit backend CPU
staging payloads for that execution. Raw tensor validation and trace inputs use
host memory; supplying native allocations currently goes through LiteRT dispatch's
buffer registration API. These diagnostics do not assert that a driver performs
no internal data movement.
