# Accelerator Test Suite (ATS) for LiteRT

The Accelerator Test Suite (ATS) is a verification tool used to test the
functionality and performance of LiteRT operations across different hardware
backends (CPU, GPU, and vendor NPUs like Qualcomm and MediaTek).

It generates combinations of operations and executes them to verify correct
execution and measure latency.

## Basic Usage

ATS is built as a test binary that can be run directly or through
pre-configured `bazel` targets.

### Running ATS on Device

The `BUILD` file defines several suites for different backends targeting
connected local devices:

```bash
# Run CPU tests
bazel run //litert/ats:cpu_ats -- [flags]

# Run CPU tests on the TFLite built-in kernels (no delegate)
bazel run //litert/ats:builtin_cpu_ats -- [flags]

# Run GPU tests
bazel run //litert/ats:gpu_ats -- [flags]

# Run Qualcomm NPU tests
bazel run //litert/ats:qualcomm_ats -- [flags]
```

### Running ATS on Host Directly

To run ATS on your local workstation host, you can run the base `:ats` binary:

```bash
bazel run //litert/ats:ats -- [flags]
```

### Running ATS on macOS (CPU, Metal GPU, and WebGPU)

To execute ATS on a local Apple Silicon Mac (`--config=darwin_arm64`):

```bash
# Run CPU (XNNPACK) ATS locally
bazel test //litert/ats:cpu_macos_ats \
  --config=darwin_arm64 \
  --test_output=streamed

# Run Metal GPU ATS locally
bazel test //litert/ats:metal_macos_ats \
  --config=darwin_arm64 \
  --//third_party/bazel_rules/rules_apple/apple/build_settings:signing_certificate_name="-" \
  --test_output=streamed

# Run WebGPU (Dawn-over-Metal) ATS locally
bazel test //litert/ats:webgpu_macos_ats \
  --config=darwin_arm64 \
  --//third_party/bazel_rules/rules_apple/apple/build_settings:signing_certificate_name="-" \
  --test_output=streamed
```

### Common Flags

*   `--backend=<backend>`: Specify the execution backend (e.g., `cpu`, `gpu`,
    `npu`).
*   `--soc_manufacturer=<manufacturer>`: Optional. Specify the NPU
    manufacturer when `--backend=npu` (e.g., `qualcomm`, `mediatek`).
*   `--cpu_kernel_mode=<mode>`: Optional. CPU kernels to test when
    `--backend=cpu`: `delegate` (default, XNNPACK), `builtin` (TFLite built-in
    kernels without a delegate), or `reference` (TFLite reference kernels).
*   `--compile_mode`: Run in compilation-only mode (useful for testing AOT
    compilation for NPUs).
*   `--do_register=<pattern>`: Positive inclusion filter defining the suite's
    test scope. Only tests whose names match at least one `--do_register` regex
    are registered in GTest; non-matching tests are omitted from registration
    entirely. Can be specified multiple times.
*   `--dont_register=<pattern>`: Negative exclusion filter for backend-specific
    unsupported tests. Matching tests are still registered in GTest and marked
    `SKIPPED` (`GTEST_SKIP()`) so that coverage denominators (`subtests_passed`
    and `capture.csv`) remain complete across backends. Takes precedence over
    `--do_register`. Can be specified multiple times.
*   `--extra_models=<path>`: Optional directory or `.tflite` model file path(s)
    to register as `ExtraModel` tests.
*   `--models_out=<path>`: Optional directory path where ATS will serialize and
    export generated `.tflite` model artifacts during test teardown.
*   `--quiet`: Suppress printing the report summary to standard output.

> [!NOTE]
> Using `--do_register` or `--dont_register` on the command line *appends* to
> the patterns already specified in the `BUILD` file targets, rather than
> overriding them.

> [!TIP]
> **Test Name Matching**: Registration filters match against the space-joined
> string `<suite> <test> <desc>`, where `<suite>` is
> `<prefix>_<fixture>_<logic>` (e.g., `CoreSingleOp_inference_Unary`,
> `CompositeOp_inference_SdpaTransposed`, or `inference_ExtraModel`) and
> `<desc>` contains the full op signature with concrete tensor shapes (or the
> `.tflite` model filename).

*   `--gtest_filter=<filter>`: Standard GoogleTest filter to select specific
    tests.

#### Examples

Run tests with custom registration filters (appends to patterns already
specified in `BUILD`):
```bash
bazel run //litert/ats:cpu_ats -- --do_register="SingleOp.*tfl\.relu"
```

Debug a specific test case using a GTest filter and suppress the summary
report:
```bash
bazel run //litert/ats:cpu_ats -- --gtest_filter="*ats_42*" --quiet
```

Export generated `TransformerLayer` `.tflite` models to a directory on the host machine without printing the summary report:
```bash
bazel run //litert/ats:ats -- --models_out=/tmp/ats_transformer_models --do_register=TransformerLayer --quiet=true
```

## Output

After execution completes, ATS generates a CSV report summarizing the results
for each operation combination, including execution success and latency.
