# LiteRT Tools Overview

This directory contains a suite of command-line tools to help you develop,
debug, and benchmark your LiteRT models.

## `run_model`

A handy tool for executing LiteRT models using the CompiledModel API. It
supports running models on different hardware accelerators (CPU, GPU, NPU) with
various debugging and benchmarking options.

### Basic Usage

```bash
run_model --graph=<model_path>
```

### Complete Parameter Reference

#### Required Parameters

-   **`--graph`** (string): Path to the LiteRT model file (.tflite)

    ```bash
    run_model --graph=/path/to/model.tflite
    ```

#### Hardware Acceleration Parameters

-   **`--accelerator`** (string, default: "cpu"): Specifies which hardware
    backend to use

    -   Valid values: `cpu`, `gpu`, `npu`

        ```bash
        run_model --graph=model.tflite --accelerator=gpu
        ```

-   **`--dispatch_library_dir`** (string): Path to dispatch library directory
    (required for NPU)

    -   Contains libLiteRtDispatch_xxx.so files

        ```bash
        run_model
        --graph=model.tflite --accelerator=npu
        --dispatch_library_dir=/path/to/dispatch/libs
        ```

-   **`--compiler_plugin_library_dir`** (string): Path to compiler plugin
    directory (for JIT compilation)

    ```bash
    run_model --graph=model.tflite --compiler_plugin_library_dir=/path/to/compiler/plugins
    ```

#### Execution Parameters

-   **`--iterations`** (size_t, default: 1): Number of times to execute the
    model

    -   Reports timing statistics: first run, slowest, fastest, and average

        ```bash
        run_model --graph=model.tflite --iterations=100
        ```

-   **`--signature_index`** (size_t, default: 0): Index of model signature to
    run (for multi-signature models)

    ```bash
    run_model --graph=model.tflite --signature_index=1
    ```

#### Debug and Analysis Parameters

-   **`--print_tensors`** (bool, default: false): Print tensor values after
    execution

    ```bash
    run_model --graph=model.tflite --print_tensors=true
    ```

-   **`--compare_numerical`** (bool, default: false): Fill inputs with test
    pattern for numerical comparison

    -   Fills inputs with rotating values (0.0, 0.1, 0.2, ..., 0.9)
    -   Also enables tensor statistics printing

        ```bash
        run_model --graph=model.tflite --compare_numerical=true
        ```

-   **`--sample_size`** (size_t, default: 5): Number of sample elements to print
    from tensors

    -   Prints samples from beginning, middle, and end of tensors
    -   Only used with `--print_tensors` or `--compare_numerical`

        ```bash
        run_model --graph=model.tflite --print_tensors=true --sample_size=10
        ```

### Usage Examples

#### CPU Execution (default)

```bash
run_model --graph=model.tflite
```

#### GPU Execution

```bash
run_model --graph=model.tflite --accelerator=gpu
```

#### NPU Execution via Dispatch API

```bash
run_model --graph=model.tflite --accelerator=npu --dispatch_library_dir=/path/to/dispatch
```

#### Performance Benchmarking

```bash
# Run 50 iterations and report timing statistics
run_model --graph=model.tflite --accelerator=gpu --iterations=50
```

#### Debugging with Tensor Inspection

```bash
# Print all tensor values with 10 samples each
run_model --graph=model.tflite --print_tensors=true --sample_size=10

# Run with test pattern and compare numerical results
run_model --graph=model.tflite --compare_numerical=true
```

#### Multi-Signature Model

```bash
# Run the second signature (index 1) of a model
run_model --graph=model.tflite --signature_index=1 --print_tensors=true
```

#### Complete Example with All Debug Options

```bash
run_model \
  --graph=model.tflite \
  --accelerator=npu \
  --dispatch_library_dir=/path/to/dispatch \
  --iterations=10 \
  --print_tensors=true \
  --compare_numerical=true \
  --sample_size=20
```

### Vendor-Specific Flags

The tool also supports vendor-specific flags when compiled with appropriate
options: - Qualcomm: Additional logging and runtime configuration flags -
MediaTek: Hardware-specific optimization flags - Google Tensor: Custom runtime
parameters

Note: These flags are only available when the tool is built with the
corresponding vendor support enabled.

## `analyze_model_main`

Inspect the structure of a LiteRT model. It can provide a summary or a detailed
view of the model's subgraphs and operators.

### Model Summary

```bash
analyze_model_main --model_path=<model_path> --only_summarize
```

### Full Analysis

```bash
analyze_model_main --model_path=<model_path>
```

## `benchmark_model`

Benchmark the performance of a LiteRT model on different hardware with improved
readability and control over output verbosity.

### Basic Usage

```bash
benchmark_model --graph=<model_path> --use_cpu
```

### Hardware Acceleration Options

-   `--use_cpu`: Use CPU acceleration (default: true)
-   `--use_gpu`: Use GPU acceleration
-   `--use_npu`: Use NPU acceleration

### Examples

#### Standard benchmark with clean output

```bash
benchmark_model --graph=model.tflite --use_gpu
```

### Benchmark Parameters

-   `--num_runs` (default: 50): Target number of benchmark iterations
-   `--min_secs` (default: 1.0): Minimum seconds to run, may increase actual
    runs
-   `--max_secs` (default: 150.0): Maximum seconds to run, may limit actual runs
-   `--warmup_runs` (default: 1): Number of warmup iterations before
    benchmarking
-   `--warmup_min_secs` (default: 0.5): Minimum warmup duration

### Output Format

**Standard Output:**

```
========== BENCHMARK RESULTS ==========
Model initialization: 45.32 ms              # One-time model loading cost
First inference:      120.45 ms             # Very first run (often slower)
Warmup (avg):         98.76 ms (1 runs)     # Warmup to prime caches
Inference (avg):      85.23 ms (50 runs)    # Main benchmark result
Inference (min):      82.10 ms              # Fastest single run
Inference (max):      92.45 ms              # Slowest single run
Inference (std):      3.21 ms               # Standard deviation

Throughput:           12.45 MB/s            # Input data processed per second

Memory Usage:
Init footprint:       25.60 MB              # Memory after model load
Overall footprint:    32.40 MB              # Total memory used
Peak memory:          35.20 MB              # Maximum memory spike
======================================
```

### Understanding the Metrics

-   **Warmup runs**: Initial runs to "warm up" the system (fill caches, trigger
    JIT compilation). These are excluded from final statistics.
-   **Inference runs**: The actual benchmark iterations used for performance
    measurement.
-   **Throughput**: Calculated as (input_size_bytes × runs_per_second) / MB,
    showing data processing rate.
-   **First/curr**: First shows the first benchmark run time, curr shows the
    most recent run time.

## `apply_plugin`

Apply a hardware-specific compiler plugin to a model. This is used to
pre-compile a model for a specific hardware target, which can improve
performance and reduce on-device compilation time.

### Basic Usage

```bash
apply_plugin --model=<input_model_path> --soc_manufacturer=<manufacturer> --soc_model=<soc_model> --libs=<path_to_plugin> --o=<output_model_path>
```

## `npu_numerics_check`

Compare the output of a model running on an NPU with the output from a CPU to
check for numerical differences. This is useful for verifying the correctness of
the NPU implementation.

### Basic Usage

```bash
npu_numerics_check --cpu_model=<cpu_model_path> --npu_model=<npu_model_path> --dispatch_library_dir=<path_to_dispatch_lib>
```

## `culprit_finder`

A powerful debugging tool to identify the specific operator ("culprit") in a
model that causes issues (e.g., crashes, numerical errors) when using hardware
acceleration. It performs a binary search over the model's operators to isolate
the problematic one. Please see the subdirectory for more detailed information.
