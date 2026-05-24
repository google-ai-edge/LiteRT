# Support for Hailo NPU in LiteRT: Implementation Plan

This document proposes a plan to add support for the Hailo NPU (specifically targeting the Hailo-8 and Hailo-8L NPUs used on the Raspberry Pi AI HAT board, as well as Hailo-10 and Hailo-15) in LiteRT.

---

## License & Open-Source Strategy Analysis

1. **HailoRT Runtime Library (`hailort`)**: The user-space runtime library repository is licensed under the **MIT License**. The MIT License is highly permissive and fully compatible with the Apache License 2.0.
2. **Hailo Drivers (`hailort-drivers`)**: The kernel PCIe drivers are licensed under the **GPL v2** (typical for Linux kernel modules). This code runs at the OS level and does not affect user-space integration libraries.
3. **Hailo Dataflow Compiler (DFC)**: The model compiler toolchain is a proprietary Python-based SDK (distributed by Hailo). 
4. **Conclusion**: You **can** contribute the bridge code (the Compiler Plugin and the Dispatch API implementation) back to the LiteRT repository. The code will compile against the open-source HailoRT headers (under MIT) and dynamically load `libhailort.so` at runtime. The proprietary compiler (Hailo DFC) and drivers/firmware are runtime/build-time prerequisites that are installed separately by the user and do not need to be distributed inside the LiteRT repository itself. This matches how Qualcomm (QAIRT/QNN) and Samsung (LiteCore) integrations are structured in LiteRT.

---

## Architecture Overview

To integrate Hailo hardware, we need to implement two core components:
1. **Hailo Compiler Plugin (AOT)**: Integrates with the LiteRT serialization layer to wrap a pre-compiled Hailo Executable Format (`.hef`) file directly into the output `.tflite` model.
2. **Hailo Dispatch API (Runtime)**: Integrates with the LiteRT runtime layer on the Raspberry Pi to load the `.hef` module, initialize the Hailo device, bind input/output buffers, and invoke hardware-accelerated inference.

```mermaid
graph TD
    subgraph AOT Compilation Phase (Workstation / Host)
        A[Original TFLite Model] --> B[LiteRT Compiler Plugin]
        B -->|1. Partitioning| C[Identify Supported Subgraphs]
        H[User Pre-compiled .hef file] -->|2. Wrapping| B
        B -->|Embed HEF| F[Serialized LiteRT Model with embedded HEF]
    end

    subgraph Runtime Execution Phase (Raspberry Pi)
        F --> G[LiteRT Interpreter]
        G --> H2[LiteRT Dispatch Delegate]
        H2 -->|LiteRtDispatchInitialize| I[Dynamically load libhailort.so]
        H2 -->|InvocationContextCreate| J[Load HEF with Hef::create_from_buffer]
        H2 -->|Register/Attach Buffers| K[Configure Input/Output Streams]
        H2 -->|DispatchInvoke| L[Inference on Hailo NPU]
    end
```

---

## Proposed Changes

We will introduce a new vendor integration under `litert/vendors/hailo`.

### 1. Workspace Configuration

We need to add workspace rules to find/download HailoRT headers and library files (e.g. from the system path `/usr/include/hailort` / `/usr/lib` or a custom directory).

#### [NEW] [hailo.bazel](file:///home/dkp/LiteRT/third_party/hailo/hailo.bazel)
Build rule to expose HailoRT headers and library targets to Bazel.
```python
cc_library(
    name = "hailort",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    srcs = ["lib/libhailort.so"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
```

#### [NEW] [hailo.bzl](file:///home/dkp/LiteRT/third_party/hailo/hailo.bzl)
Workspace configuration rules for Hailo.
```python
load("//litert/sdk_util:repo.bzl", "configurable_repo")

def hailo_configure():
    configurable_repo(
        name = "hailo_sdk",
        build_file = Label("@//third_party/hailo:hailo.bazel"),
        local_path_env = "HAILO_RT_DIR",
    )
```

#### [MODIFY] [WORKSPACE](file:///home/dkp/LiteRT/WORKSPACE)
Load and initialize `hailo_configure()`.

---

### 2. Hailo Compiler Plugin

Located in `litert/vendors/hailo/compiler/`.

#### [NEW] [hailo_compiler_plugin.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/compiler/hailo_compiler_plugin.cc)
Implements the Compiler Plugin C API:
- `LiteRtGetCompilerPluginSupportedHardware`: returns `kLiteRtHwAcceleratorNpu`.
- `LiteRtCompilerPluginPartition`: Iterates through the subgraph ops, checking if they are supported by Hailo NPU. Selects supported operations.
- `LiteRtCompilerPluginCompile`: 
  1. Checks for the path of a pre-compiled `.hef` file (supplied via custom LiteRtOptions or the `LITERT_HAILO_HEF_PATH` environment variable).
  2. Reads the `.hef` file directly from the filesystem.
  3. Packages the HEF byte array into the `LiteRtCompiledResult` bytecode array, which LiteRT then automatically wraps into the output `.tflite` model.

---

### 3. Hailo Dispatch API

Located in `litert/vendors/hailo/dispatch/`.

#### [NEW] [dispatch_api.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/dispatch_api.cc)
Exposes the entry point `LiteRtDispatchGetApi` returning `LiteRtDispatchInterface` implemented using the HailoRT C++ API.

#### [NEW] [device_context.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/device_context.cc) & [device_context.h](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/device_context.h)
Manages the lifetime of a Hailo device:
- `DispatchDeviceContextCreate`: Initializes a Hailo `hailo_vdevice` or `hailo_device` using `VDevice::create()`.
- Implements buffer registration APIs to map LiteRT tensor buffers.

#### [NEW] [invocation_context.cc](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/invocation_context.cc) & [invocation_context.h](file:///home/dkp/LiteRT/litert/vendors/hailo/dispatch/invocation_context.h)
Manages model execution state:
- `DispatchInvocationContextCreate`: Receives the `.hef` bytecode, loads it using `Hef::create_from_buffer`, and configures the `ConfiguredNetworkGroup`.
- Configures virtual input/output streams (`VStream`).
- `DispatchAttachInput` / `DispatchAttachOutput`: Binds user-provided tensor buffers to the virtual streams.
- `DispatchInvoke`: Activates the network group, writes inputs to virtual streams, executes inference, reads outputs, and deactivates the group.

---

### 4. Build Configuration

#### [MODIFY] [CMakeLists.txt](file:///home/dkp/LiteRT/litert/vendors/CMakeLists.txt)
Conditionally includes the `hailo` subdirectory if `hailo/hailort.hpp` is found:
```cmake
check_include_file_cxx("hailo/hailort.hpp" HAVE_HAILO_HDR)
if(HAVE_HAILO_HDR)
  _litert_add_dispatch_so(Hailo "hailo/dispatch" "Hailo")
else()
  message(STATUS "Skipping Hailo dispatch: hailo/hailort.hpp not found")
endif()
```
