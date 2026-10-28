# Arm Vulkan ML developer tool

`litert_arm_vulkan_ml_tool` is a C++ diagnostic CLI for the direct Arm Vulkan ML
backend. It uses the production operator analysis, partitioner, lowering and
Vulkan dispatch. Image preprocessing and end-to-end evaluation use the shared
LiteRT ImageNet evaluator, which continues to support other backends.

| Command | Purpose | Vulkan device needed |
| --- | --- | --- |
| `inspect` | Operator coverage, rejection reasons, tensor types/shapes/quantization and selected partitions | No |
| `compile` | Lower selected partitions, validate SPIR-V, export assembly, constants and operator/result-ID mappings | No |
| `capabilities` | Query extensions/revisions, feature bits, tensor limits/formats and TOSA engine profiles/extensions | Yes |
| `validate` | Require complete coverage, validate SPIR-V and compare GPU inference against CPU reference | Yes, with ML support |
| `trace` | Find the first intermediate output outside tolerance using CPU reference | Yes, with ML support |

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
is development reference material; this executable does not load it at runtime.

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
TOSA level constraints remain explicitly `unchecked`; pipeline acceptance is
`not_tested`. An operator's presence in the specification alone never proves
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

`suite.example.json` pins six model artifacts (FP32/INT8 MobileNetV2, INT8
ResNet18, EfficientNet-B0, DenseNet121 and MobileNetV3 Large). Copy it and the three
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
`known_failure` is an annotation only: DenseNet121's known prediction difference
and MobileNetV3 Large's pooling error remain failures under the default gates.
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

`qualify_portable_ops.py` creates deterministic operator fixtures and exercises
inspect, compile, capabilities, validate and both trace modes. It requires
`numpy`, `flatbuffers` and generated TFLite Python bindings with object APIs.
Generate bindings from `tflite/schema/schema.fbs` with FlatBuffers `flatc --python
--gen-object-api`, place their parent directory on `PYTHONPATH`, and configure the
Vulkan environment as above. Then run:

```sh
python litert/vendors/arm_vulkan_ml/tools/devtool/qualify_portable_ops.py \
  --tool "$TOOL" --output_dir .arm-vulkan-preflight/portable-qualification \
  --allow_emulation
```

The driver uses four synthetic tensor cases per supported fixture, including
negative values, zeros and ties where applicable. These are not ImageNet images.
It exits unsuccessfully for unexpected coverage, pipeline or numerical failures.
