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

### Performance Configuration

Flag                                | Type | Default   | Description
----------------------------------- | ---- | --------- | -----------
`--intel_openvino_performance_mode` | enum | `latency` | Performance mode (`latency`, `throughput`, `cumulative_throughput`)

### Advanced Configuration

Flag                           | Type   | Default | Description
------------------------------ | ------ | ------- | -----------
`--intel_openvino_configs_map` | string | `""`    | Comma-separated key=value pairs for OpenVINO configuration properties (e.g., `INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/cache`)

### Per-Graph (Per-Partition) Overrides

For multi-subgraph models (e.g. distinct `prefill` / `decode` signatures), each
partition can be compiled for a different OpenVINO device and given partition-
specific OpenVINO properties. The compiler plugin records the chosen device in
a self-describing header on each partition's bytecode, so the dispatcher
imports every partition on the device it was compiled for automatically.

Flag                                  | Type   | Default | Description
------------------------------------- | ------ | ------- | -----------
`--intel_openvino_graph_backends`     | string | `""`    | Either a bare backend name (`cpu`, `gpu`, `npu`) applied to all partitions, or semicolon-separated `GRAPH_INDEX:BACKEND` entries (e.g. `0:npu;1:cpu`) for per-partition selection. Partitions without an entry default to NPU.
`--intel_openvino_graph_configs_map`  | string | `""`    | Semicolon-separated `GRAPH_INDEX:KEY=VALUE` entries (e.g. `1:INFERENCE_PRECISION_HINT=f32`). Merged on top of `--intel_openvino_configs_map` for the indicated graph.

`GRAPH_INDEX` is the partition index produced by the LiteRT partitioner. To map
a TFLite signature key to a partition index, resolve it via
`LiteRtGetSignatureSubgraph` (or the equivalent C++/Python API) before passing
the integer to these flags.

## Usage Examples

### Command Line Usage

```bash
# Basic NPU inference (apply NPU to every partition)
./your_binary --intel_openvino_graph_backends=npu

# High throughput NPU inference (all partitions on NPU)
./your_binary \
  --intel_openvino_graph_backends=npu \
  --intel_openvino_performance_mode=throughput

# Advanced configuration with custom OpenVINO properties
./your_binary \
  --intel_openvino_graph_backends=npu \
  --intel_openvino_configs_map="NPU_COMPILATION_MODE_PARAMS=compute-layers-with-higher-precision=Sigmoid,CACHE_DIR=/tmp/ov_cache"

# GPU with low precision inference (all partitions on GPU)
./your_binary \
  --intel_openvino_graph_backends=gpu \
  --intel_openvino_configs_map="INFERENCE_PRECISION_HINT=f16"

# AOT-compile a 2-signature model: partition 0 on NPU, partition 1 on CPU
# with a per-partition precision hint
apply_plugin_main \
  --cmd=apply \
  --model=/path/to/model.tflite \
  --soc_manufacturer=IntelOpenVINO \
  --soc_model=PTL \
  --libs=/path/to/plugin/dir \
  --o=/path/to/output.tflite \
  --intel_openvino_graph_backends="0:npu;1:cpu" \
  --intel_openvino_graph_configs_map="1:INFERENCE_PRECISION_HINT=f32"
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
    auto backend = options.GetGraphBackend(/*graph_index=*/0);
    if (backend && *backend == kLiteRtIntelOpenVinoGraphBackendNPU) {
      // Configure NPU-specific settings
    }
  }

  return 0;
}
```

## Flag Value Details

### Graph Backends

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
// Valid graph backend strings
"cpu", "gpu", "npu"

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
