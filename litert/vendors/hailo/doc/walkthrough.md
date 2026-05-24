# Walkthrough: Hailo NPU Support in LiteRT

This document provides a walkthrough of the changes implemented to integrate support for the Hailo NPU hardware (Raspberry Pi AI HAT) into LiteRT.

---

## Changes Implemented

We have successfully created the following directories and files to support the Hailo NPU backend in LiteRT:

### 1. Workspace Configuration (External SDK)
*   **[hailo.bazel](file:///home/dkp/LiteRT/third_party/hailo/hailo.bazel)**: Defines the external Bazel library target `@hailo_sdk//:hailort`, pointing to the HailoRT C/C++ SDK headers and shared library (`libhailort.so`).
*   **[hailo.bzl](file:///home/dkp/LiteRT/third_party/hailo/hailo.bzl)**: Defines the workspace rule `hailo_configure()` using the environment variable `HAILO_RT_DIR`.
*   **[WORKSPACE](file:///home/dkp/LiteRT/WORKSPACE)**: Modified to import and load `hailo_configure()`.

### 2. Hailo Compiler Plugin (AOT Wrapper)
*   **[hailo_compiler_plugin.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/compiler/hailo_compiler_plugin.cc)**: Implements the LiteRT Compiler Plugin C API.
    *   **Partitioning**: Groups all operations in the target subgraph into a single partition index (0) to map to a single compiled NPU graph.
    *   **Compilation**: Reads the path to a pre-compiled `.hef` file from the environment variable `LITERT_HAILO_HEF_PATH`, reads the raw bytecode, and wraps it directly inside the `LiteRtCompiledResult` without invoking any Python toolchain or compiler subprocesses in-process.
*   **[hailo_compiler_plugin_test.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/compiler/hailo_compiler_plugin_test.cc)**: GTest-based unit tests verifying the compiler plugin's partitioning logic, SoC config mapping, and pre-compiled HEF file wrapping.
*   **[BUILD](file:///home/dkp/LiteRT/litert/vendors/hailo/compiler/BUILD)**: Bazel target file defining compilation rules for the compiler plugin dynamic library and test suite.

### 3. Hailo Dispatch API (Runtime Execution)
*   **[device_context.h](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/device_context.h)** & **[device_context.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/device_context.cc)**: Manages the `hailort::VDevice` virtual device handle and registers CPU host memory buffers mapped to user tensors.
*   **[invocation_context.h](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/invocation_context.h)** & **[invocation_context.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/invocation_context.cc)**: Loads the embedded HEF bytecode using `hailort::Hef::create_from_buffer` and configures input/output streams (`hailort::InputVStream` and `hailort::OutputVStream`). Handles inference execution by feeding the streams during `Invoke()`.
*   **[dispatch_api.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/dispatch_api.cc)**: Exposes the Dispatch C API entry points (`LiteRtDispatchGetApi`) mapped to the device and invocation contexts.
*   **[BUILD](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/BUILD)**: Bazel target file defining compilation rules for the Dispatch API shared library `libLiteRtDispatch_Hailo.so`.

### 4. Build Configuration
*   **[CMakeLists.txt](file:///home/dkp/LiteRT/litert/vendors/CMakeLists.txt)**: Configures CMake to check for `<hailo/hailort.hpp>` and compile the Hailo Dispatch shared library if found.
*   **[BUILD](file:///home/dkp/LiteRT/litert/vendors/hailo/BUILD)**: Root vendor Bazel package declaration file.

---

## How to Build and Run

### 1. Build using Bazel
Set `HAILO_RT_DIR` to the location of your HailoRT SDK installation (containing `include/hailo/hailort.hpp` and `lib/libhailort.so`):
```bash
export HAILO_RT_DIR=/path/to/hailort/sdk
bazel build //litert/vendors/hailo/compiler:compiler_plugin
bazel build //litert/vendors/hailo/dispatch:dispatch_api
```

To run the compiler plugin unit tests:
```bash
bazel test //litert/vendors/hailo/compiler:hailo_compiler_plugin_test
```

### 2. Package a Model (AOT Wrap)
Compile your TFLite model to a `.hef` file offline using the Hailo Dataflow Compiler (DFC) on your workstation.
Then, package it into a LiteRT model:
```bash
export LITERT_HAILO_HEF_PATH=/path/to/compiled_model.hef
bazel run //litert/tools:apply_plugin_main -- \
  --model=/path/to/input_model.tflite \
  --soc_manufacturer=Hailo \
  --soc_model=Hailo-8 \
  --libs=bazel-bin/litert/vendors/hailo/compiler \
  --o=/path/to/output_model_with_embedded_hef.tflite
```

### 3. Execute on Target Hardware (Raspberry Pi)
Deploy `libLiteRtDispatch_Hailo.so` and the compiled `output_model_with_embedded_hef.tflite` to the Raspberry Pi. Execute inference using the standard LiteRT runtime interpreter; it will load the Dispatch API and accelerate execution on the Hailo NPU.
