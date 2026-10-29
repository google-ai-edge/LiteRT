# Arm Vulkan ML regression testing

The supported developer commands are the [Arm diagnostic CLI](../devtool/README.md)
and the [shared ImageNet evaluator](../../../../tools/imagenet_eval_runner.md).
This directory contains host regression drivers and a focused integration test.
Downloaded models, generated fixtures and reports belong in ignored output folders.

## Host setup

Install Mesa Lavapipe and Vulkan validation layers matching the pinned Vulkan
headers (1.4.349). Create an isolated Python environment from the repository root:

```sh
uv venv --python 3.13 .arm-vulkan-preflight/venv
uv pip install --python .arm-vulkan-preflight/venv/bin/python \
  -r litert/vendors/arm_vulkan_ml/tools/testing/requirements.txt \
  -r litert/vendors/arm_vulkan_ml/tools/tosa_capabilities/requirements.txt
```

The Arm emulation wheel supplies the graph/tensor layers and native libraries.
Configure child processes using `emulation_environment.layer_environment(Path(ICD))`,
where `ICD` is the installed Lavapipe manifest. The helper sets `VK_DRIVER_FILES`,
`VK_LAYER_PATH`, `VK_INSTANCE_LAYERS` and `LD_LIBRARY_PATH`; it does not alter the
parent process. Add the Khronos layer manifest directory to that child's
`VK_LAYER_PATH` when requesting `--vulkan_validation=sync`.

For interactive shell use, the equivalent settings are:

```sh
export VK_DRIVER_FILES=/path/to/lvp_icd.x86_64.json
export VK_LAYER_PATH=/path/to/emulation_layer/deploy/share/vulkan/explicit_layer.d:/path/to/validation/layers
export VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation
export LD_LIBRARY_PATH=/path/to/emulation_layer/deploy/lib
```

Emulation must be explicitly allowed by the CLI's `--allow_emulation=1`, or by
`LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1` for the public LiteRT API. Unadvertised
TOSA extensions remain rejected. These tests establish functional behavior;
they do not establish native GPU performance or exhaustive conformance.
Scenario Runner and Python VGF packages are not required.

## Synthetic operator cases

Generate the specification database using the [extraction guide](../tosa_capabilities/README.md).
Build `//litert/vendors/arm_vulkan_ml/tools:litert_arm_vulkan_ml_tool` and generate
Python object bindings with this checkout's FlatBuffers compiler:

```sh
bazelisk run @flatbuffers//:flatc -- --python --gen-object-api \
  -o .arm-vulkan-preflight/schema tflite/schema/schema.fbs
PYTHONPATH=.arm-vulkan-preflight/schema \
.arm-vulkan-preflight/venv/bin/python \
  litert/vendors/arm_vulkan_ml/tools/testing/qualify_portable_ops.py \
  --tool bazel-bin/litert/vendors/arm_vulkan_ml/tools/litert_arm_vulkan_ml_tool \
  --output_dir .arm-vulkan-preflight/operator-tests \
  --allow_emulation --vulkan_validation=sync
```

Use the repository's per-worktree Bazel cache/output flags for these builds.
The driver checks 59 FP32/typed positive fixtures and eight expected rejections.
Repeat with `--quantized_only` and a separate output directory for 32 INT8
positive fixtures and six rejections. Four synthetic cases are generated per
positive fixture. Each checks intended opcode emission, SPIR-V, capabilities,
full validation, and accumulated/isolated tracing. INT8 comparisons are exact.
These cases consume no ImageNet images.

## Real LiteRT model lifecycle

`//litert/vendors/arm_vulkan_ml/tools/testing:lifecycle_test` generates tiny FP32
and INT8 ADD models and uses the real compiler plugin, dispatch and public
LiteRT compiled-model API. It checks complete acceleration, exact expected sums,
repeated invocation, interleaved live instances and survival after another model
is destroyed. No downloaded model is needed.

Build the test with Bazel. With the Vulkan environment above configured, opt in:

```sh
bazelisk test -c opt \
  --test_env=LITERT_ARM_VULKAN_ML_RUN_INTEGRATION_TESTS=1 \
  --test_env=LITERT_ARM_VULKAN_ML_ALLOW_EMULATION=1 \
  --test_env=VK_DRIVER_FILES --test_env=VK_LAYER_PATH \
  --test_env=VK_INSTANCE_LAYERS --test_env=LD_LIBRARY_PATH \
  //litert/vendors/arm_vulkan_ml/tools/testing:lifecycle_test
```

The target supplies plugin locations through runfiles. Without explicit opt-in
it skips. Production capability queries and Vulkan initialization must succeed
when enabled; failures are not converted into skips.

## Independent convolution oracle

Build `//litert/vendors/arm_vulkan_ml/compiler:conv2d_fixture` and
`@arm_vulkan_ml_tool_spirv_tools//:spirv-val`. The fixture uses the production
analyzer/emitter and direct Vulkan invocation, exposing its accumulator for the
independent NumPy INT64 oracle:

```sh
.arm-vulkan-preflight/venv/bin/python \
  litert/vendors/arm_vulkan_ml/tools/testing/conv2d_oracle_test.py \
  --fixture bazel-bin/litert/vendors/arm_vulkan_ml/compiler/conv2d_fixture \
  --model litert/test/testdata/arm_vulkan_ml_conv2d_int8.tflite --op-index 0 \
  --spirv-val "$PWD/bazel-bin/external/arm_vulkan_ml_tool_spirv_tools/spirv-val" \
  --output-dir .arm-vulkan-preflight/convolution-oracle
```

Four inputs exercise zero points, negative values, saturation and layout.
Accumulator and rescale comparisons are exact; the separate TFLite reference
comparison allows one code difference. `generate_conv_fixture.py OUTPUT 2`
creates a grouped-convolution variant using the generated schema bindings.

For full-model smoke checks use the public example suite and preprocessing
configurations documented by the Arm CLI. Use three images per model. Model
conversion/calibration belongs to the model-preparation project, outside LiteRT.
