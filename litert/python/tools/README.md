<!-- Copyright (C) 2026 Intel Corporation
     SPDX-License-Identifier: Apache-2.0

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License. -->

# LiteRT Python Tools

This directory contains Python command-line tools for working with LiteRT
models.

## `benchmark_litert_model`

Benchmark the inference performance of a LiteRT model using the Python API.
This is the Python counterpart to the C++ `litert/tools/benchmark_model`,
useful for validating pip wheel installations on target devices without
compiling C++ code.

Installed as the `litert-benchmark` console entry point in the pip wheel.

### Basic Usage

```bash
# Run from pip wheel
litert-benchmark --model=model.tflite

# Run as module
python -m litert.python.tools.benchmark_litert_model --model=model.tflite
```

### Hardware Acceleration Options

- `--use_cpu`: Use CPU acceleration (default: true)
- `--use_gpu`: Use GPU acceleration
- `--use_npu`: Use NPU acceleration (requires dispatch library)
- `--no_cpu`: Disable CPU fallback
- `--require_full_delegation`: Fail if model is not fully accelerated

```bash
# CPU only (default)
litert-benchmark --model=model.tflite

# GPU with CPU fallback
litert-benchmark --model=model.tflite --use_gpu

# NPU with CPU fallback
litert-benchmark --model=model.tflite --use_npu \
    --dispatch_library_path=/path/to/libLiteRtDispatch.so

# NPU only, require full acceleration
litert-benchmark --model=model.tflite --use_npu \
    --dispatch_library_path=/path/to/dispatch.so --require_full_delegation
```

### Benchmark Parameters

- `--num_runs` (default: 50): Number of benchmark iterations
- `--warmup_runs` (default: 5): Number of warmup iterations
- `--num_threads` (default: 1): Number of CPU threads
- `--signature` (default: first): Signature key to run

### Environment Parameters

- `--dispatch_library_path`: Path to vendor dispatch library
- `--compiler_plugin_path`: Path to compiler plugin library directory
- `--runtime_path`: Path to LiteRT runtime library directory

### Output Options

- `--result_json`: Path to save results as JSON (for CI integration)
- `--verbose`: Enable verbose output with per-iteration timing

### Output Format

```
========================================
 BENCHMARK RESULTS
========================================
Model initialization:  45.32 ms
Warmup (first):        120.45 ms
Warmup (avg):          98.76 ms (5 runs)
Inference (avg):       85.23 ms (50 runs)
Inference (min):       82.10 ms
Inference (max):       92.45 ms
Inference (median):    84.50 ms
Inference (std):       3.21 ms
Inference (p5):        82.50 ms
Inference (p95):       91.00 ms
Throughput:            12.45 MB/s
Fully accelerated:     True
========================================
```

### JSON Output

Use `--result_json` to save structured results:

```bash
litert-benchmark --model=model.tflite --result_json=results.json
```

Output includes model metadata, accelerator configuration, and full latency
statistics (avg, min, max, median, std, p5, p95).

### Intel OpenVINO Options

When using `--use_npu` with the Intel OpenVINO dispatch library, these
additional options control device selection and performance tuning:

- `--intel_openvino_device_type` (default: `npu`): Target device —
  `cpu`, `gpu`, `npu`, or `auto`
- `--intel_openvino_performance_mode` (default: `latency`): Performance hint —
  `latency`, `throughput`, or `cumulative_throughput`
- `--intel_openvino_configs_map`: Comma-separated `KEY=VALUE` pairs for custom
  OpenVINO properties

The dispatch library is auto-discovered from the pip wheel when installed with
`pip install ai-edge-litert[npu-intel]`. To specify it manually, use
`--dispatch_library_path`.

```bash
# NPU inference (dispatch auto-discovered from wheel)
litert-benchmark --model=model.tflite --use_npu

# OpenVINO CPU inference
litert-benchmark --model=model.tflite --use_npu \
    --intel_openvino_device_type=cpu

# NPU with throughput optimization
litert-benchmark --model=model.tflite --use_npu \
    --intel_openvino_performance_mode=throughput

# Custom OpenVINO properties
litert-benchmark --model=model.tflite --use_npu \
    --intel_openvino_configs_map="NUM_STREAMS=2,INFERENCE_PRECISION_HINT=f16"
```
