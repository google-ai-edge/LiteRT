# Arm Vulkan ML developer tool

`litert_arm_vulkan_ml_tool` is a C++ diagnostic CLI for the direct Arm Vulkan ML
backend. It uses the production operator analysis, partitioner, lowering and
Vulkan dispatch. Image preprocessing and end-to-end evaluation use the shared
LiteRT ImageNet evaluator, which continues to support other backends.

| Command | Purpose | Vulkan device needed |
| --- | --- | --- |
| `inspect` | Operator coverage, rejection reasons, tensor types/shapes/quantization and selected partitions | No |
| `compile` | Lower selected partitions, validate SPIR-V, export assembly, constants, operator/result-ID mappings and optional VGF | No |
| `capabilities` | Query extensions/revisions, feature bits, tensor limits/formats and TOSA engine profiles/extensions | Yes |
| `validate` | Require complete coverage, validate SPIR-V and compare GPU inference against CPU reference | Yes, with ML support |
| `trace` | Find the first intermediate output outside tolerance using CPU reference | Yes, with ML support |
| `tosa-query` | Filter specification type rows and profile/extension requirements | No |
| `coverage` | Show the reviewed operator implementation inventory | No |

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
dispatch libraries do not depend on it. The generated TOSA capability database
is optional diagnostic input: only `tosa-query` reads it. Inference compiler
and dispatch libraries never load it.

For Android arm64, add the following flags, using your installed SDK/NDK paths
and a separate Bazel output base:

```sh
--config=android_arm64 \
--repo_env=ANDROID_SDK_HOME=/data/bt/os/android_sdk \
--repo_env=ANDROID_NDK_HOME=/data/bt/os/android_sdk/ndk/30.0.16248370 \
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
"$TOOL" inspect --model model.tflite --report_json inspect.json
"$TOOL" compile --model model.tflite --output_dir compiled \
  --report_json compile.json
```

The JSON report includes original zero-based subgraph, operator and tensor
indices. Partial coverage remains explicitly marked `coverage_status: partial`;
a successful `inspect` or `compile` does not imply full-model execution support.
Unsupported operators include the production rejection reason.

Each selected partition exports `.spv`, `.spvasm`, a JSON manifest and binary
constant buffers. The lowering map associates newly created SPIR-V result IDs
with original operators. Shared IDs belong to their first creating operator;
module scaffolding and final output wiring have no operator attribution. These
are diagnostic artifacts, not a serialized executable accepted by dispatch.
SPIR-V validation uses the universal 1.6 environment and is not proof of Vulkan
pipeline acceptance or hardware conformance.

## Dump lowered partitions as VGF

```sh
"$TOOL" compile --model model.tflite --output_dir compiled \
  --dump_vgf --report_json compiled/compile.json
```

This additionally creates `subgraph_<N>_partition_<M>.vgf` for every accelerated
partition. Each container embeds its graph SPIR-V, exact constant payloads,
entry point, descriptor bindings, physical tensor formats/shapes and IO ordering.
IO names are `input_0`, `input_1`, etc. and `output_0`, `output_1`, etc.; the JSON
manifest preserves the original LiteRT operator/tensor mapping. The manifest also
records `vgf_file`, `vgf_sha256` and `vgf_vulkan_header_version` (currently 349).
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
"$TOOL" capabilities --report_json device.json
"$TOOL" capabilities --model model.tflite --report_json compatibility.json
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

## Validate ImageNet models

Execution initially supports single-subgraph classification models with one
static batch-one RGB FP32/INT8 input and one `[1, classes]` output. The input
layout must be unambiguously NCHW or NHWC. Supply a tab-separated ImageNet manifest
with `id`, zero-based `label`, and image `path` columns. Relative paths resolve
under `--image_root`, which defaults to the manifest's directory. See the shared
[ImageNet evaluator guide](../../../../../g3doc/instructions/IMAGENET_ANDROID_EVAL.md)
for dataset preparation and the evaluator's other backends.

Download a pinned example model:

```sh
uvx --from huggingface_hub hf download litert-community/MobileNet-v2 \
  mobilenet_v2.tflite \
  --revision 48e55de4765732640b9702c520999e4bf1a1f50f \
  --local-dir models

"$TOOL" validate --model models/mobilenet_v2.tflite \
  --manifest /path/to/manifest.tsv --start_sample 2000 --max_samples 3 \
  --preprocessing_config litert/vendors/arm_vulkan_ml/tools/devtool/preprocessing_232.json \
  --compiler_plugin_library_dir bazel-bin/litert/vendors/arm_vulkan_ml/compiler \
  --dispatch_library_dir bazel-bin/litert/vendors/arm_vulkan_ml/dispatch \
  --report_json validation.json
```

Defaults are three images, FP32 `--atol=1e-4 --rtol=1e-3`, INT8
`--max_code_error=4 --mean_code_error=0.5`, full acceleration and top-1 agreement.
INT8 gates apply in quantized code space; reports also retain dequantized errors.
Sample counts are restricted to 1–10. This is a developer smoke test, not an
ImageNet accuracy measurement. Each image's errors/predictions and the exact
preprocessed-input hash are retained. CPU baseline uses reference kernels.
`--runtime_library_dir` is available when built for dynamic LiteRT runtime loading.

`suite.example.json` pins seven model artifacts (FP32/INT8 MobileNetV2, INT8
ResNet18, EfficientNet-B0, DenseNet121, MobileNetV3 Large and Inception V3).
Inception uses resize 342 and crop 299; download its pinned static INT8 artifact:

```sh
uvx --from huggingface_hub hf download litert-community/inception_v3 \
  inception_v3_int8_channelwise.tflite \
  --revision 1b7f421b5ecf8082620670e381c11aa284687237 --local-dir models
```

Copy the suite and the four
preprocessing JSON files to a local working directory, download models into its
`models/` directory, then run:

```sh
"$TOOL" validate --suite /path/to/suite.example.json \
  --manifest /path/to/manifest.tsv --start_sample 2000 \
  --compiler_plugin_library_dir /path/to/compiler \
  --dispatch_library_dir /path/to/dispatch --report_json suite-results.json
```

Paths specified by suite entries resolve relative to the suite file. CLI paths
resolve from the working directory. Model SHA-256 must match before execution.
`known_failure` is an annotation only. Native TOSA rescaling uses SINGLE_ROUND
and can differ from the double-rounding reference. Per-model numerical errors
and DenseNet121's known prediction difference remain failures under the default
gates; [the example suite](suite.example.json) records the known failures.
Do not commit downloaded models or generated reports/artifacts.

## Trace a mismatch

```sh
"$TOOL" trace --model model.tflite \
  --manifest /path/to/manifest.tsv --start_sample 2000 \
  --preprocessing_config preprocessing.json --mode accumulated \
  --capture_budget_mb 256 --report_json trace.json
```

Trace processes one image (also the only accepted `--max_samples` value):

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

All commands print JSON and optionally save it with `--report_json`. Reports
carry schema version, command, model/binary hashes when available and a status.
Exit codes are 0 for success, 1 for execution/report failures, and 2 for malformed
arguments. Detailed driver/compiler diagnostics go to stderr. Numeric Vulkan
format, quality and version fields retain the API values without reinterpretation.

## Arbitrary FP32 tensor fixtures

`validate` and `trace` also accept `--tensor_inputs inputs.json`, mutually
exclusive with the ImageNet manifest and preprocessing flags:

```json
{"cases": [{"id": "broadcast", "inputs": [
  {"index": 0, "file": "lhs.f32"},
  {"index": 1, "file": "rhs.f32"}
]}]}
```

`index` is the zero-based model input position, not the tensor ID. Supply every
input exactly once. Files contain contiguous little-endian FP32 values with
exactly the static model shape's byte count. Paths resolve relative to the JSON
manifest. Linux and Android arm64 builds use little-endian hosts. Constant INT32
control operands belong in the model. Dynamic and scalar input interfaces remain
unsupported.

```sh
"$TOOL" validate --model broadcast.tflite --tensor_inputs inputs.json \
  --max_samples 3 --report_json comparison.json
"$TOOL" trace --model broadcast.tflite --tensor_inputs inputs.json \
  --mode isolated --report_json isolated.json
```

Validation checks every model output against TFLite CPU reference and runs
SPIR-V validation before pipeline creation. It uses the same bounded execution
machinery as tracing; tensor cases do not apply classification prediction gates.
Validation defaults to three cases and trace to one, with `--start_sample` for
selection. Reports retain case IDs, input hashes, per-output shapes/errors and
partial results on failure. Suites may specify `tensor_inputs` per model.

[The regression driver](../testing/README.md) creates deterministic operator fixtures and exercises
inspect, compile, capabilities, validate and both trace modes. It requires
`numpy`, `flatbuffers` and generated TFLite Python bindings with object APIs.
Generate bindings from `tflite/schema/schema.fbs` with FlatBuffers `flatc --python
--gen-object-api`, place their parent directory on `PYTHONPATH`, and configure the
Vulkan environment as above. Then run:

```sh
python litert/vendors/arm_vulkan_ml/tools/testing/qualify_portable_ops.py \
  --tool "$TOOL" --output_dir .arm-vulkan-preflight/portable-qualification \
  --allow_emulation
```

The driver uses four synthetic tensor cases per supported fixture, including
negative values, zeros and ties where applicable. These are not ImageNet images.
It exits unsuccessfully for unexpected coverage, pipeline or numerical failures.

## Vulkan validation diagnostics

Use `--vulkan_validation=core` or `--vulkan_validation=sync` with `capabilities`,
`validate` or `trace`. The default is `off`. Offline `inspect`/`compile` commands
reject a validation request because they do not create a Vulkan instance.
`sync` enables synchronization validation in addition to core checks.

The requested Khronos layer and required extensions must be available. Point
`VK_LAYER_PATH` at both the validation and emulator manifest directories when
using emulation; `VK_ADD_LAYER_PATH` is ignored by loaders when `VK_LAYER_PATH`
is already set. Use a validation layer built from tag `v1.4.349`, commit
`8506dcd07d3fbd78f6381c75cab8fa62a91d1b64`, to match the pinned header baseline.
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

The fixture driver accepts `--vulkan_validation=sync` and uses that mode for its
capability, validation and trace commands. A matching baseline and a passing
fixture run are execution evidence; they do not establish an exhaustive VUID
audit or native GPU qualification. See the [audit notes](../tosa_capabilities/README.md).

## Query TOSA specification candidates on Linux or Android

`tosa-query` ports the Python capability query engine to C++. It reads an explicit
schema-version-1 `TOSA.001000.1` database generated on the host. Copy the database
JSON alongside the executable on Android; Python, Git, registry checkouts and
Vulkan ML extensions are not needed for querying it.

```sh
"$TOOL" tosa-query --database tosa_001000_1.json
"$TOOL" tosa-query --database tosa_001000_1.json \
  --op CONV2D --type input=fp16_t --enum acc_type=FP32 --capability PRO-FP
"$TOOL" tosa-query --database tosa_001000_1.json \
  --op CAST --type input=bf16_t --type output=fp8e4m3_t \
  --capability PRO-FP --capability EXT-BF16 --capability EXT-FP8E4M3 \
  --json --report_json cast-query.json
```

`--type`, `--enum` and `--capability` may be repeated. Type filters name registry
operand columns and use TOSA names such as `i8_t` and `fp32_t`. Operator names are
case-insensitive; enum values, operand names and capability names are exact.
Repeated assignments for the same operand/enum use the last value, matching the
Python reference. Omit `--op` for database counts; filters require an operator.

Without capability flags, the query returns all matching specification rows.
Supplying capability flags filters profile/extension OR/AND requirements and
enum-specific requirements against exactly that supplied set. Zero candidates
is a successful query, not a supported operation. Unknown names, malformed input
and incompatible database versions fail. JSON reports preserve the complete
operator record with filtered `variants`, a candidate count and database hash.

This command does not query the phone or evaluate shape equations, TOSA-level
limits, lowering availability or hardware support. Reports explicitly mark those
checks as unassessed/not evaluated/not queried. Use `capabilities --model` and
actual model validation for device evidence. `--vulkan_validation` must be off
for this offline command. The inference compiler/dispatch still do not load the
database; only this diagnostic subcommand does.

Host parity regression:

```sh
python litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/tosa_query_parity_test.py \
  --tool "$TOOL" --database tosa_001000_1.json
```

The test compares C++ output with the Python reference for all operators, all
350 type rows, profile filtering, enum-extension requirements and invalid names.
Host extraction and source-audit scripts remain Python tools.

## Operator implementation coverage

Run `litert_arm_vulkan_ml_tool coverage` for an offline inventory, or add
`--json` and `--report_json FILE` for machine-readable output. The compact
reviewed manifest is compiled into the executable; no specification database,
Vulkan loader, or Python installation is required on the target.

Coverage currently means 60 of 66 TOSA instructions have production lowering
routes from existing TFLite builtins. It does not mean all types/shapes work.
Each record separates instruction emission, SPIR-V validation, emulator
qualification and hardware qualification. `pending` means no qualification claim.
`compile` reports each partition's actual `tosa_opcodes`; fixture qualification
asserts that its intended instruction was emitted.

Raw tensor manifests also accept INT8, INT32, UINT32 and BOOL model inputs. Files use
little-endian 32-bit integers or one byte per INT8/Boolean (Boolean values 0/1); the model supplies
the type and shape. Integer/Boolean outputs compare exactly in validation and
both trace modes. Existing FP32 and quantized comparison rules still apply.

The `qualify_portable_ops.py --quantized_only` group exercises the quantized
subsets described in the backend README. It overrides the normal INT8 tolerance
with `--max_code_error=0 --mean_code_error=0`; all numerical comparisons must be
exact. It checks intended TABLE, RESIZE and TRANSPOSE_CONV2D opcode emission as
well as full acceleration, validation, CPU comparison and both trace modes.

## External tensor memory

With `capabilities --model FILE`, each device includes `external_tensors` entries
for model inputs/outputs, with DMA-BUF/AHardwareBuffer importability and rejection
reasons. These describe the tensor configuration; attachment must still validate
the actual allocation, memory type, size, alignment and dedicated-allocation rules.

Direct execution operator records include `buffer_io.input_copy_bytes`,
`output_copy_bytes` and `imported_slots`. Counts describe explicit backend CPU
staging payloads for that execution. Existing raw/image trace inputs use host
memory; supplying native allocations currently goes through LiteRT dispatch's
buffer registration API. These diagnostics do not assert that a driver performs
no internal data movement.
