# Intel OpenVINO Flags for LiteRT

## Overview

The Intel OpenVINO flags allow users to configure Intel OpenVINO options via
command-line arguments, making it easy to tune inference parameters without
recompiling code.

## Files Structure

```
litert/tools/flags/vendors/
├── intel_openvino_flags.h              # Header with flag declarations
├── intel_openvino_flags.cc             # Implementation with flag definitions
├── intel_openvino_flags_test.cc        # Unit tests
├── intel_openvino_flags_example.cc     # Usage example
└── README_intel_openvino_flags.md      # This file
```

## Available Flags

### Device Configuration

| Flag                           | Type | Default | Description           |
| ------------------------------ | ---- | ------- | --------------------- |
| `--intel_openvino_device_type` | enum | `npu`   | Target device (`cpu`, |
:                                :      :         : `gpu`, `npu`, `auto`) :

### Performance Configuration

Flag                                | Type | Default   | Description
----------------------------------- | ---- | --------- | -----------
`--intel_openvino_performance_mode` | enum | `latency` | Performance mode (`latency`, `throughput`, `cumulative_throughput`)

### Advanced Configuration

Flag                           | Type   | Default | Description
------------------------------ | ------ | ------- | -----------
`--intel_openvino_configs_map` | string | `""`    | Comma-separated key=value pairs for OpenVINO configuration properties (e.g., `INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/cache`)

## Usage Examples

### Command Line Usage

```bash
# Basic NPU inference
./your_binary --intel_openvino_device_type=npu

# High throughput NPU inference
./your_binary \
  --intel_openvino_device_type=npu \
  --intel_openvino_performance_mode=throughput

# Advanced configuration with custom OpenVINO properties
./your_binary \
  --intel_openvino_device_type=npu \
  --intel_openvino_configs_map="NPU_COMPILATION_MODE_PARAMS=compute-layers-with-higher-precision=Sigmoid,CACHE_DIR=/tmp/ov_cache"

# GPU with low precision inference
./your_binary \
  --intel_openvino_device_type=gpu \
  --intel_openvino_configs_map="INFERENCE_PRECISION_HINT=f16"
```

### Programmatic Usage

```cpp
#include "litert/tools/flags/vendors/intel_openvino_flags.h"
#include "absl/flags/parse.h"

int main(int argc, char** argv) {
  // Parse command line flags
  absl::ParseCommandLine(argc, argv);

  // Create options from parsed flags
  auto options_result = litert::intel_openvino::IntelOpenVinoOptionsFromFlags();
  if (options_result) {
    auto options = std::move(options_result.Value());

    // Use options to configure Intel OpenVINO
    if (options.GetDeviceType() == kLiteRtIntelOpenVinoDeviceTypeNPU) {
      // Configure NPU-specific settings
    }
  }

  return 0;
}
```

## Flag Value Details

### Device Types

-   **`cpu`**: Execute on Intel CPU (best compatibility)
-   **`gpu`**: Execute on Intel GPU (good for parallel workloads)
-   **`npu`**: Execute on Intel NPU (best efficiency for AI workloads)
-   **`auto`**: Let OpenVINO choose the best available device

### Performance Modes

-   **`latency`**: Optimize for single inference latency
-   **`throughput`**: Optimize for batch processing throughput
-   **`cumulative_throughput`**: Optimize for multiple concurrent requests

### Configuration Map

The `--intel_openvino_configs_map` flag allows you to pass arbitrary OpenVINO
configuration properties as comma-separated key=value pairs. These properties
are passed directly to the OpenVINO Core's `set_property()` method.

**Common Configuration Properties:** - `INFERENCE_PRECISION_HINT=f16`: Suggest
inference precision (f16, f32, bf16) - `NPU_COMPILATION_MODE_PARAMS=<params>`:
NPU-specific compilation parameters - `CACHE_DIR=/path/to/cache`: Directory for
model caching - `PERFORMANCE_HINT=LATENCY`: Performance hint (alternative to
performance_mode flag) - `NUM_STREAMS=4`: Number of parallel inference streams -
`ENABLE_PROFILING=YES`: Enable performance profiling

**Example:** `bash ./your_binary
--intel_openvino_configs_map="INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/ov_cache,NUM_STREAMS=2"`

## Flag Parsing and Validation

The flags support automatic parsing and validation:

```cpp
// Valid device type strings
"cpu", "gpu", "npu", "auto"

// Valid performance mode strings
"latency", "throughput", "cumulative_throughput"
```

Invalid flag values will result in parsing errors with descriptive error
messages.

## Build Integration

To use Intel OpenVINO flags in your target:

```bazel
cc_binary(
    name = "your_binary",
    srcs = ["your_binary.cc"],
    deps = [
        "//litert/tools/flags/vendors:intel_openvino_flags",
        # other dependencies
    ],
)
```

With dynamic runtime support:

```bazel
cc_binary(
    name = "your_binary",
    srcs = ["your_binary.cc"],
    deps = [
        "//litert/tools/flags/vendors:intel_openvino_flags_with_dynamic_runtime",
        # other dependencies
    ],
)
```

## Testing

Run the flag tests to verify functionality:

```bash
bazel test //litert/tools/flags/vendors:intel_openvino_flags_test
```

The tests verify: - Flag parsing for all enum types - Default value handling -
Option creation from flags - Error handling for invalid values - Flag reset
functionality

## Artificial Intelligence

These contents may have been developed with support from one or more
Intel-operated generative artificial intelligence solutions.
