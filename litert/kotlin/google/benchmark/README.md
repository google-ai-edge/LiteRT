# LiteRT Kotlin API Benchmark Tool App

The LiteRT Kotlin Benchmark App provides model performance benchmarking on
Android using the LiteRT Kotlin API (`com.google.ai.edge.litert`).

It offers feature parity for CPU and GPU acceleration with the C++ benchmark
tool (`//litert/tools:benchmark_model`).

## Build and Install

```bash
blaze --blazerc=java/com/google/android/gmscore/blaze/blazerc build --config=gmscore_arm64 \
  litert/kotlin/google/benchmark:benchmark \
  && adb install -r blaze-bin/litert/kotlin/google/benchmark/benchmark.apk
```

## Running Benchmarks

### Device Storage Permissions Note

On Android 11+ (API 30+), Scoped Storage restricts apps from accessing arbitrary
root `/sdcard/` paths without special file permissions. You can use either of
the following approaches:

**Approach A (Recommended - No Permission Required)**: Push the model to the
app-specific external storage directory:

```bash
adb shell mkdir -p /sdcard/Android/data/com.google.ai.edge.litert.benchmark/files
adb push model.tflite /sdcard/Android/data/com.google.ai.edge.litert.benchmark/files/model.tflite
```

**Approach B (Grant All Files Access via ADB)**:

```bash
adb shell appops set com.google.ai.edge.litert.benchmark MANAGE_EXTERNAL_STORAGE allow
adb push model.tflite /sdcard/model.tflite
```

### Option 1: Individual Intent Extras (`--es`, `--ei`, `--ez`, `--ef`)

```bash
adb shell am start \
    -n com.google.ai.edge.litert.benchmark/.BenchmarkActivity \
    --es "graph" "/sdcard/Android/data/com.google.ai.edge.litert.benchmark/files/model.tflite" \
    --ei "num_runs" 50 \
    --ei "warmup_runs" 5 \
    --ez "use_gpu" true \
    --es "gpu_precision" "fp16"
```

### Option 2: Command-Line Flags String (`--es args`)

```bash
adb shell am start \
    -n com.google.ai.edge.litert.benchmark/.BenchmarkActivity \
    --es args "--graph=/sdcard/Android/data/com.google.ai.edge.litert.benchmark/files/model.tflite --use_gpu=true --gpu_precision=fp16 --num_runs=50 --warmup_runs=5"
```

## Supported Parameters

Parameter Flag                      | Type        | Default       | Description
:---------------------------------- | :---------- | :------------ | :----------
`graph`, `model_path`               | String      | *Required*    | Absolute path to `.tflite` model file on device.
`signature`                         | String      | `""`          | Target model signature key to execute.
`use_gmscore`                       | Boolean     | `false`       | Whether to use GMSCore LiteRT runtime instead of bundled runtime (GPU acceleration not supported yet).
`use_cpu`                           | Boolean     | `true`        | Whether to enable CPU acceleration.
`use_gpu`                           | Boolean     | `false`       | Whether to enable GPU acceleration.
`num_threads`, `num_threads_cpu`    | Int         | System        | Number of CPU threads to allocate.
`gpu_backend`                       | String      | `"automatic"` | GPU backend (`automatic`, `opencl`, `opengl`).
`gpu_precision`                       | String      | `"default"`   | GPU precision mode (`default`, `fp16`, `fp32`, `fp16_with_fp32_accum`).
`gpu_low_priority`                  | Boolean     | `false`       | Run GPU execution at low priority.
`enable_weight_sharing`             | Boolean     | `false`       | Enable constant tensor weight sharing on GPU.
`convert_weights_on_gpu`            | Boolean     | `false`       | Convert weights directly on GPU.
`xnnpack_weight_cache_file_path`    | String      | `null`        | Path to XNNPACK weight cache file.
`num_runs`                          | Int         | `50`          | Number of measured benchmark iterations.
`warmup_runs`                       | Int         | `1`           | Number of initial warmup runs before measurement.
`warmup_min_secs`                   | Double      | `0.5`         | Minimum total duration (in seconds) spent in warmup.
`min_secs`                          | Double      | `1.0`         | Minimum total duration (in seconds) spent in benchmark.
`run_delay`                         | Double/Long | `0`           | Delay between benchmark runs (in ms or seconds).
`input_layer_value_range`           | String      | `""`          | Range for input layer values (e.g. `input1,0,1:input2,0,255`).
`finish_on_completion`, `finish`    | Boolean     | `false`       | Automatically close activity after benchmark completes.

## Benchmark Methodology

The LiteRT Kotlin Benchmark App measures model inference performance,
initialization latency, and memory footprint across the following phases:

1.  **Initialization Phase**:

    -   Executes forced Garbage Collection (`System.gc()`) to establish a clean
        JVM memory baseline prior to initialization.
    -   Creates the LiteRT runtime `Environment`.
    -   Configures `CompiledModel.Options` for the specified accelerator (CPU or
        GPU) with parameters such as thread count, GPU backend
        (`OpenCL`/`OpenGL`), precision (`FP16`/`FP32`), and weight sharing
        options.
    -   Compiles the `.tflite` model and measures wall-clock initialization time
        (`initMs`).
    -   Allocates input and output `TensorBuffer` instances and populates input
        buffers with pseudo-random values matched to the input tensor's data
        type, element count, and configured value range.

2.  **Warmup Phase**:

    -   Executes unmeasured warmup runs (`compiledModel.run`) to eliminate
        cold-start overheads (driver initialization, memory allocation, GPU
        shader compilation, and thermal state stabilization).
    -   Runs until **both** `warmup_runs` count and `warmup_min_secs` duration
        criteria are satisfied. Warmup timings are tracked separately.

3.  **Inference Benchmark Phase**:

    -   Executes measured inference runs continuously until **both** `num_runs`
        count and `min_secs` duration criteria are satisfied.
    -   Supports optional inter-run delay (`run_delay`) between inferences to
        mitigate CPU/GPU thermal throttling.

4.  **Statistical Calculation**:

    -   Computes execution metrics across all measured inference iterations:
        -   **Average (Mean) Latency**: Average time per inference run.
        -   **Min / Max Latency**: Fast and slow latency extremes.
        -   **Standard Deviation**: Variance in run times across iterations.
        -   **Percentiles**: 5th percentile ($P_5$), Median ($P_{50}$), and 95th
            percentile ($P_{95}$) computed via linear interpolation on sorted
            sample values.

5.  **Memory Usage Sampling**:

    -   Tracks process Proportional Set Size (PSS) and Native Heap Allocation
        via `android.os.Debug`.
    -   Records baseline memory before model loading, post-compilation memory,
        and peak memory reached during inference execution to report total
        memory footprint (`Peak - Baseline`).

6.  **Cleanup**:

    -   Closes all native buffer handles, compiled models, and environment
        resources.

## Viewing Benchmark Results

Results are displayed on screen and logged to Android Logcat (`Log.i`).

To filter results in Logcat via ADB:

```bash
adb logcat -s LiteRtBenchmark
```
