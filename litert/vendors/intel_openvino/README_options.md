# Intel OpenVINO Options for LiteRT

This directory contains the Intel OpenVINO-specific options implementation for
LiteRT, allowing fine-grained control over Intel OpenVINO inference parameters.

## Overview

The Intel OpenVINO options provide a way to configure various aspects of Intel
OpenVINO inference, including device selection and performance tuning.

## Files Structure

```
litert/
├── c/options/
│   ├── litert_intel_openvino_options.h    # C API header
│   └── litert_intel_openvino_options.cc   # C API implementation
├── cc/options/
│   ├── litert_intel_openvino_options.h    # C++ wrapper header
│   └── litert_intel_openvino_options.cc   # C++ wrapper implementation
└── vendors/intel_openvino/
    ├── intel_openvino_options_example.cc  # Usage example
    └── README_options.md                  # This file
```

## Available Options

### Device Type

-   **CPU**: Run inference on Intel CPU
-   **GPU**: Run inference on Intel GPU
-   **NPU**: Run inference on Intel NPU
-   **AUTO**: Let OpenVINO automatically select the best device

### Performance Mode

-   **Latency**: Optimize for low latency (single inference)
-   **Throughput**: Optimize for high throughput (batch processing)
-   **Cumulative Throughput**: Optimize for cumulative throughput across
    multiple requests

### Configuration Map

Allows setting arbitrary OpenVINO configuration properties as key-value pairs.
These properties are applied directly to the OpenVINO Core during model
compilation.

**Common Use Cases:** - Configure inference precision hints (e.g.,
`INFERENCE_PRECISION_HINT`) - Set NPU compilation parameters (e.g.,
`NPU_COMPILATION_MODE_PARAMS`) - Set up model caching (e.g., `CACHE_DIR`) -
Control number of inference streams (e.g., `NUM_STREAMS`) - Configure
device-specific optimization parameters

### Plugin-Specific Options

In addition to OpenVINO Core properties, the following keys are recognized by
the LiteRT Intel OpenVINO compiler plugin and are consumed internally rather
than forwarded to the OpenVINO Core:

-   **`optimize_fq_after_matmul`** (`"true"` | `"false"`): When set to `"true"`,
    enables an NPU-only model optimization pass that eliminates `FakeQuantize`
    operations placed immediately after `MatMul` nodes. This can improve NPU
    performance for quantized models. Has no effect on non-NPU devices.
    Defaults to `false`.

    ```cpp
    options.SetConfigsMapOption("optimize_fq_after_matmul", "true");
    ```

#### Example: passing `optimize_fq_after_matmul` through `apply_plugin_main`

The flag is forwarded via the generic `--intel_openvino_configs_map` flag,
which takes a comma-separated list of `KEY=VALUE` pairs:

```bash
apply_plugin_main \
    --cmd=apply \
    --model=/path/to/model.tflite \
    --soc_manufacturer=IntelOpenVINO \
    --soc_model=PTL \
    --libs=/path/to/plugin/dir \
    --o=/path/to/output.tflite \
    --intel_openvino_device_type=npu \
    --intel_openvino_performance_mode=latency \
    --intel_openvino_configs_map="optimize_fq_after_matmul=true,INFERENCE_PRECISION_HINT=f16"
```

The plugin recognizes `optimize_fq_after_matmul` as an internal key and
consumes it directly; the remaining entries are forwarded to the OpenVINO
Core as configuration properties.

## Usage Example

### C++ API

```cpp
#include "litert/cc/options/litert_intel_openvino_options.h"

using litert::intel_openvino::IntelOpenVinoOptions;

// Create options
auto options = IntelOpenVinoOptions::Create().Value();

// Configure device and performance
options.SetDeviceType(kLiteRtIntelOpenVinoDeviceTypeNPU);
options.SetPerformanceMode(kLiteRtIntelOpenVinoPerformanceModeLatency);

// Set custom OpenVINO configuration properties
options.SetConfigsMapOption("INFERENCE_PRECISION_HINT", "f16");
options.SetConfigsMapOption("NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sigmoid");
options.SetConfigsMapOption("CACHE_DIR", "/tmp/ov_cache");
```

### C API

```c
#include "litert/c/options/litert_intel_openvino_options.h"

LrtIntelOpenVinoOptions options;
LrtIntelOpenVinoOptionsCreate(&options);

// Configure options
LrtIntelOpenVinoOptionsSetDeviceType(options, kLiteRtIntelOpenVinoDeviceTypeNPU);
LrtIntelOpenVinoOptionsSetPerformanceMode(options, kLiteRtIntelOpenVinoPerformanceModeLatency);

// Set custom configuration properties
LrtIntelOpenVinoOptionsSetConfigsMapOption(options, "INFERENCE_PRECISION_HINT", "f16");
LrtIntelOpenVinoOptionsSetConfigsMapOption(options, "CACHE_DIR", "/tmp/ov_cache");

// Extract opaque payloads manually for passing
const char* identifier;
void* payload;
void (*payload_deleter)(void*);
LrtGetOpaqueIntelOpenVinoOptionsData(options, &identifier, &payload, &payload_deleter);

// Cleanup
LrtDestroyIntelOpenVinoOptions(options);
payload_deleter(payload);
```

### Parsing from TOML

Intel OpenVINO options can also be parsed directly from a TOML-formatted string payload using the C API. This is the mechanism used by the runtime when loading external configurations dynamically.

```c
#include "litert/c/options/litert_intel_openvino_options.h"

const char* toml_payload =
    "device_type = 2\n"  // NPU
    "performance_mode = 0\n" // Latency
    "configs_map.INFERENCE_PRECISION_HINT = \"f16\"\n";

LrtIntelOpenVinoOptions options = NULL;
LiteRtStatus status = LrtCreateIntelOpenVinoOptionsFromToml(toml_payload, &options);

if (status == kLiteRtStatusOk) {
  // Options successfully instantiated from string payload

  LrtDestroyIntelOpenVinoOptions(options);
}
```

## Integration with Intel OpenVINO Compiler Plugin

These options are designed to be used with the Intel OpenVINO compiler plugin
located in `litert/vendors/intel_openvino/compiler/`. The compiler plugin can
read these options to configure the OpenVINO Core and compile models with the
specified settings.

Example integration in the compiler plugin:

```cpp
// In openvino_compiler_plugin.cc
void ConfigureOpenVinoFromOptions(ov::Core& core, const IntelOpenVinoOptions& options) {
  // Set device
  std::string device = DeviceTypeToString(options.GetDeviceType());

  // Configure performance hints
  if (options.GetPerformanceMode() == kLiteRtIntelOpenVinoPerformanceModeLatency) {
    core.set_property(device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }
}
```

## Default Values

-   Device Type: NPU
-   Performance Mode: Latency

## Artificial Intelligence

These contents may have been developed with support from one or more
Intel-operated generative artificial intelligence solutions.
