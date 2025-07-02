# LiteRT Tools Overview

This directory contains a suite of command-line tools to help you develop,
debug, and benchmark your LiteRT models.

## `run_model`

This is a simple tool to run a model with the CompiledModel API.

```
run_model --graph=<model_path>
```

### Use NPU via Dispatch API

If you're using the Dispatch API, you need to pass the Dispatch library
(libLiteRtDispatch_xxx.so) location via `--dispatch_library_dir`

```
run_model --graph=<model_path> --dispatch_library_dir=<dispatch_library_dir>
```

### Use GPU

If you run a model with GPU accelerator, use `--use_gpu` flag.

```
run_model --graph=<model_path> --use_gpu
```

## `analyze_model_main`

Inspect the structure of a LiteRT model. It can provide a summary or a
detailed view of the model's subgraphs and operators.

### Model Summary
```bash
analyze_model_main --model_path=<model_path> --only_summarize
```

### Full Analysis
```bash
analyze_model_main --model_path=<model_path>
```

## `benchmark_model`

Benchmark the performance of a LiteRT model on different hardware.

### Basic Usage
```bash
benchmark_model --graph=<model_path> --use_cpu
```
Can also use `--use_gpu` and `--use_npu`

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
