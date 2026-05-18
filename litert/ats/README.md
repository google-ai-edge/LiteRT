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

### Common Flags

*   `--backend=<backend>`: Specify the execution backend (e.g., `cpu`, `gpu`,
    `npu`).
*   `--soc_manufacturer=<manufacturer>`: Optional. Specify the NPU
    manufacturer when `--backend=npu` (e.g., `qualcomm`, `mediatek`).
*   `--compile_mode`: Run in compilation-only mode (useful for testing AOT
    compilation for NPUs).
*   `--do_register=<pattern>`: Only run tests whose names match the given regex
    pattern. Can be specified multiple times.
*   `--dont_register=<pattern>`: Skip tests whose names match the given regex
    pattern. Can be specified multiple times.
*   `--quiet`: Suppress printing the report summary to standard output.

> [!NOTE]
> Using `--do_register` or `--dont_register` on the command line *appends* to
> the patterns already specified in the `BUILD` file targets, rather than
> overriding them.

> [!TIP]
> **Test Name Matching**: Test names are constructed mapping to the template
> `ats_<test_id>_<family>_<logic>`. The registration filters use regular
> expressions to perform search matches against these names.

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

## Output

After execution completes, ATS generates a CSV report summarizing the results
for each operation combination, including execution success and latency.
