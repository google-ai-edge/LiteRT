# ⚡️ LiteRt Accelerator Test Suite (ats) Documentation

The **LiteRt Accelerator Test Suite (ats)** is a comprehensive tool used to
validate the functional correctness and measure the performance of custom
accelerator implementations integrated with the LiteRt framework.

-----

## Overview and Core Functionality

The primary function of `ats` is to execute pre-defined machine learning models
against a target accelerator and compare the results against the LiteRt
**standard CPU backend**.

*   **Validation:** The suite performs **numeric validation** by comparing the
    output tensors (activations) produced by the accelerator against those
    produced by the known-good CPU backend. This ensures the accelerator
    implementation maintains the required precision and correctness.
*   **Performance Metrics:** It automatically captures and records critical
    performance details, including **latency** and other relevant metrics, which
    are made available to the user.
*   **Execution:** The tests are typically executed on a target device (e.g., an
    Android phone) and are managed by a **shell script** wrapper that handles
    file transfers and setup via the `adb` (Android Debug Bridge) tool.

-----

## Test Data (Models)

The `ats` suite uses a collection of widely-used **`.tflite` models** as its
test data.

### Currently Included Models

The following models are automatically included and downloaded for testing
(subject to change):

*   `hf_all_minilm_l6_v2`
*   `hf_mobilevit_small`
*   `qai_hub_midas`
*   `qai_hub_real_esrgan_x4plus`
*   `torchvision_mobilenet_v2`
*   `torchvision_resnet18`
*   `torchvision_squeezenet1_1`
*   `u2net_lite`
*   `whisper_tiny_decoder`
*   `whisper_tiny_encoder`
*   `yamnet`
*   `yolo11n`

### Future Expansion

The suite is planned to be expanded to include dedicated tests for **single
operations (ops)**, in addition to full models.

### Manual Model Retrieval

While models are automatically downloaded during a `bazel run`, you can manually
retrieve the entire model set using `wget`:

```bash
wget -p -O <target_file> https://storage.googleapis.com/litert/ats_models.tar.gz
```

-----

## Compilation Mode (AOT)

For accelerators that support an **Ahead-of-Time (AOT) compilation** step, `ats`
can be executed in a dedicated **"compile mode"**.

*   **Purpose:** This mode is designed to be run on a **workstation** (host
    machine), not the target device. It compiles the models for the specified
    target hardware without executing them.
*   **Output:** All compiled models are output to a designated directory on the
    workstation.
*   **Activation:** This mode is activated using the `--compile_mode` flag.

-----

## Defining an `ats` Suite with Bazel

Users leverage the `litert_define_ats` Bazel macro to configure and define an
`ats` testing target specific to their accelerator.

The macro automatically creates **two** runnable targets:

1.  The standard **on-device JIT test** (for execution and validation).
2.  A dedicated **AOT "compile only" mode test** (for host compilation).

### Example `litert_define_ats` Usage

The example below defines an `ats` suite named `example_npu_ats` for an
accelerator with the backend name `example`:

```bazel
litert_define_ats(
    name = "example_npu_ats",
    backend = "example",
    compile_only_suffix = "_aot",
    do_register = [
        "*mobilenet*",
    ],
    extra_flags = ["--limit=1"],
    jit_suffix = "",
)
```

### Execution

To execute the standard on-device test (which handles all `adb` push/pull
operations):

```bash
bazel run -c opt --config=android_arm64 :example_npu_ats
```

To execute the AOT compilation test:

```bash
bazel run :example_npu_ats_aot
```

-----

## Command-Line Flags

The `ats` executable accepts several flags for granular control over testing and
reporting.

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--backend` | `std::string` | **Required.** Which LiteRt backend to use as the accelerator under test (the "actual"). Options are `cpu`, `npu`, or `gpu`. |
| `--compile_mode` | `bool` | If true, runs the AOT compilation step on the workstation instead of on-device execution. |
| `--models_out` | `std::string` | The directory path where side-effect serialized (compiled) models are saved. Only relevant for AOT or JIT compilation. |
| `--dispatch_dir` | `std::string` | Path to the directory containing the accelerator's dispatch library (relevant for NPU). |
| `--plugin_dir` | `std::string` | Path to the directory containing the accelerator's compiler plugin library (relevant for NPU). |
| `--soc_manufacturer` | `std::string` | The SOC manufacturer to target for AOT compilation (relevant for NPU compilation). |
| `--soc_model` | `std::string` | The SOC model to target for AOT compilation (relevant for NPU compilation). |
| `--iters_per_test` | `size_t` | Number of iterations to run per test, each with different randomized tensor data. |
| `--max_ms_per_test` | `int64_t` | Maximum time in milliseconds to run each test before a timeout. |
| `--fail_on_timeout` | `bool` | Whether the test should fail if the execution times out. |
| `--csv` | `std::string` | File path to save the detailed report in CSV format. |
| `--dump_report` | `bool` | Whether to dump the entire report details directly to the user's console output. |
| `--data_seed` | std::optional&lt;int&gt; | A single seed for global data generation. |
| `--do_register` | std::vector&lt;std::string&gt; | Regex(es) for explicitly including specific tests (e.g., `*mobilenet*`). |
| `--dont_register` | std::vector&lt;std::string&gt; | Regex(es) to exclude specific tests. |
| `--extra_models` | std::vector&lt;std::string&gt; | Optional list of directories or model files to add to the test suite. |
| `--limit` | `int32_t` | Limit the total number of tests registered and run. |
| `--quiet` | `bool` | Minimize logging output during the test run. |

-----

## Integrating New Accelerators with `litert_device_script`

The `ats` suite, like other on-device LiteRt tests, relies on the
**`litert_device_script`** tooling to manage the execution environment. This
tooling automatically creates **shell script wrappers** around standard
`cc_test` or `cc_binary` targets, handling the complex steps of pushing
necessary libraries and data files to a remote device via `adb`.

### Backend Registration

To enable a new accelerator for use with `litert_device_script` (and therefore
`ats`), its required libraries must be registered in the
**`litert_device_common.bzl`** Bazel file. Registration is based on a unique
**"backend" name** which maps to a set of buildable or pre-compiled libraries
needed for LiteRt to operate with that accelerator.

#### Registration Steps

1.  **Define a `BackendSpec` function:** Create a function that returns a
    dictionary containing your new accelerator's specification.

2.  **Specify Libraries (`libs`):** This is a list of tuples detailing the Bazel
    target path for the shared library and the environment variable
    (`LD_LIBRARY_PATH`) required for the device linker to find it.

    *   **Dispatch Library:** Required for runtime execution.
    *   **Compiler Plugin Library:** Required for AOT compilation mode.

3.  **Specify Library Names (`plugin`, `dispatch`):** Provide the simple file
    names of the plugin and dispatch libraries.

4.  **Register the Spec:** Merge your new spec function into the main `_Specs`
    function to make it available by its unique backend ID.

### Example Registration (`_ExampleSpec`)

The following code from `litert_device_common.bzl` illustrates how the "example"
accelerator is registered:

```python
def _ExampleSpec():
    return {
        # The unique backend ID
        "example": BackendSpec(
            id = "example",
            libs = [
                # Dispatch Library and how to find it on device
                ("//third_party/odml/litert/litert/vendors/examples:libLiteRtDispatch_Example.so", "LD_LIBRARY_PATH"),
                # Compiler Plugin Library
                ("//third_party/odml/litert/litert/vendors/examples:libLiteRtCompilerPlugin_Example.so", "LD_LIBRARY_PATH"),
            ],
            plugin = "libLiteRtCompilerPlugin_Example.so",
            dispatch = "libLiteRtDispatch_Example.so",
        ),
    }

# ... (Other specs are defined here)

def _Specs(name):
    # Your new spec function must be included here
    return (_QualcommSpec() | _GoogleTensorSpec() | _MediatekSpec() | _CpuSpec() | _GpuSpec() | _ExampleSpec())[name]
```

### Leveraging Registration with `litert_device_exec`

Once registered, users can leverage the **`litert_device_exec`** and related
macros with the new **`backend_id`**. This macro automatically bundles the
required libraries and any specified data files with the target binary.

```bazel
cc_binary(
	name = "example_bin",
	srcs = ["example_bin.cc"],
)

litert_device_exec(
    name = "example_bin_device",
    backend_id = "example",  # Uses the libraries registered under "example"
    data = [
        "//third_party/odml/litert/litert/test:testdata/constant_output_tensor.tflite",
    ],
    target = ":example_bin",
)
```

Running this target (`bazel run ... :example_bin_device`) will:

1.  Build the `example_bin` C++ binary.
2.  Push the binary, `libLiteRtDispatch_Example.so`,
    `libLiteRtCompilerPlugin_Example.so`, and the `.tflite` file to the device.
3.  Execute the binary using `adb shell`.

> **Note on Device Paths:** The canonical location for files on the device
> mirrors Bazel's runfile tree, specifically
> `/data/local/tmp/runfiles/runfiles_relative_path`. The device script
> automatically handles setting the appropriate paths for the dynamic linker.

-----
