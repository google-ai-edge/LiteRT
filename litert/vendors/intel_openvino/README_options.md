# Intel OpenVINO Options for LiteRT

This directory contains the Intel OpenVINO-specific options implementation for
LiteRT, allowing fine-grained control over Intel OpenVINO inference parameters.

## Overview

The Intel OpenVINO options provide a way to configure various aspects of Intel
OpenVINO inference, including per-partition device selection and performance
tuning.

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

### Graph Backend (per partition)

The target OpenVINO device is selected per partition / signature. Supported
backends:

-   **CPU**: Run partition on Intel CPU
-   **GPU**: Run partition on Intel GPU
-   **NPU**: Run partition on Intel NPU (default when no override is set)

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
    performance for quantized models. Has no effect on non-NPU devices. Defaults
    to `false`.

    ```cpp
    options.SetConfigsMapOption("optimize_fq_after_matmul", "true");
    ```

#### Example: passing `optimize_fq_after_matmul` through `apply_plugin_main`

The flag is forwarded via the generic `--intel_openvino_configs_map` flag, which
takes a comma-separated list of `KEY=VALUE` pairs:

```bash
apply_plugin_main \
    --cmd=apply \
    --model=/path/to/model.tflite \
    --soc_manufacturer=IntelOpenVINO \
    --soc_model=PTL \
    --libs=/path/to/plugin/dir \
    --o=/path/to/output.tflite \
    --intel_openvino_graph_backends=0:npu \
    --intel_openvino_performance_mode=latency \
    --intel_openvino_configs_map="optimize_fq_after_matmul=true,INFERENCE_PRECISION_HINT=f16"
```

The plugin recognizes `optimize_fq_after_matmul` as an internal key and consumes
it directly; the remaining entries are forwarded to the OpenVINO Core as
configuration properties.

## Usage Example

### C++ API

```cpp
#include "litert/cc/options/litert_intel_openvino_options.h"

using litert::intel_openvino::IntelOpenVinoOptions;

// Create options
auto options = IntelOpenVinoOptions::Create().Value();

// Configure backend for partition 0 and performance mode
options.SetGraphBackend(/*graph_index=*/0, kLiteRtIntelOpenVinoGraphBackendNPU);
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
LrtIntelOpenVinoOptionsSetGraphBackend(options, /*graph_index=*/0,
                                       kLiteRtIntelOpenVinoGraphBackendNPU);
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
    "performance_mode = 0\n" // Latency
    "configs_map.INFERENCE_PRECISION_HINT = \"f16\"\n"
    "graph.0.graph_backend = 2\n";  // NPU for partition 0

LrtIntelOpenVinoOptions options = NULL;
LiteRtStatus status = LrtCreateIntelOpenVinoOptionsFromToml(toml_payload, &options);

if (status == kLiteRtStatusOk) {
  // Options successfully instantiated from string payload

  LrtDestroyIntelOpenVinoOptions(options);
}
```

## Per-Graph Backend Selection

When a model contains multiple subgraphs (for example, distinct signatures for
`prefill` and `decode`), each partition can be compiled for and dispatched to a
different OpenVINO device. The compiler plugin records the chosen device in a
self-describing header prepended to each partition's bytecode, so the dispatcher
imports every partition on the device it was compiled for &mdash; no additional
plumbing is required between the compile and dispatch steps.

There is no model-wide device default; partitions without an explicit graph
backend fall back to NPU.

The override key is the **graph (partition) index** assigned by the LiteRT
partitioner. Applications that work in terms of TFLite signatures should map the
signature key to its subgraph index (e.g. via `LiteRtGetSignatureSubgraph`)
before calling these APIs.

### C++ API

```cpp
auto options = IntelOpenVinoOptions::Create().Value();

// Compile and dispatch partition 0 on NPU.
options.SetGraphBackend(/*graph_index=*/0,
                        kLiteRtIntelOpenVinoGraphBackendNPU);

// Compile and dispatch partition 1 on CPU, with a per-graph precision hint
// that takes precedence over the model-wide configs_map.
options.SetGraphBackend(/*graph_index=*/1,
                        kLiteRtIntelOpenVinoGraphBackendCPU);
options.SetGraphConfigsMapOption(/*graph_index=*/1,
                                 "INFERENCE_PRECISION_HINT", "f32");
```

### TOML payload

Per-graph overrides are also expressible directly in the TOML payload:

```toml
graph.0.graph_backend = 2                      # NPU
graph.1.graph_backend = 0                      # CPU
graph.1.configs_map.INFERENCE_PRECISION_HINT = "f32"
```

## Integration with Intel OpenVINO Compiler Plugin

These options are designed to be used with the Intel OpenVINO compiler plugin
located in `litert/vendors/intel_openvino/compiler/`. The compiler plugin can
read these options to configure the OpenVINO Core and compile models with the
specified settings.

Example integration in the compiler plugin:

```cpp
// In openvino_compiler_plugin.cc
void ConfigureOpenVinoFromOptions(ov::Core& core, const IntelOpenVinoOptions& options,
                                  int graph_index) {
  // Resolve the partition's backend (defaults to NPU when no override exists).
  auto backend_or = options.GetGraphBackend(graph_index);
  LiteRtIntelOpenVinoGraphBackend backend =
      backend_or ? *backend_or : kLiteRtIntelOpenVinoGraphBackendNPU;
  std::string device = GraphBackendToString(backend);

  // Configure performance hints
  if (options.GetPerformanceMode() == kLiteRtIntelOpenVinoPerformanceModeLatency) {
    core.set_property(device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }
}
```

## Default Values

-   Graph Backend (per partition, when no override is set): NPU
-   Performance Mode: Latency

## Artificial Intelligence

These contents may have been developed with support from one or more
Intel-operated generative artificial intelligence solutions.
