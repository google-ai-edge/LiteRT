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

LiteRtOpaqueOptions opaque_options;
LiteRtIntelOpenVinoOptionsCreate(&opaque_options);

LiteRtIntelOpenVinoOptions options;
LiteRtIntelOpenVinoOptionsGet(opaque_options, &options);

// Configure options
LiteRtIntelOpenVinoOptionsSetDeviceType(options, kLiteRtIntelOpenVinoDeviceTypeNPU);
LiteRtIntelOpenVinoOptionsSetPerformanceMode(options, kLiteRtIntelOpenVinoPerformanceModeLatency);

// Set custom configuration properties
LiteRtIntelOpenVinoOptionsSetConfigsMapOption(options, "INFERENCE_PRECISION_HINT", "f16");
LiteRtIntelOpenVinoOptionsSetConfigsMapOption(options, "CACHE_DIR", "/tmp/ov_cache");
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
