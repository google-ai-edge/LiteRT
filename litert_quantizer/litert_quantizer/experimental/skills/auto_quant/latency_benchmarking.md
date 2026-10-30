# Latency Benchmarking (Optional Third Pareto Axis)

File size is a proxy. What users deploy against is on-device latency and peak
memory, and quantization decisions can move latency in BOTH directions:
INT8/INT4 kernels are usually faster, but scattered precision boundaries
insert quantize/dequantize ops that can make a "smaller" model slower (this is
the runtime motivation behind the block-isolation rules in
`regex_targeting.md`).

Latency measurement is OPTIONAL — only perform it when the user states a
latency requirement or asks for on-device numbers. Never block the size/quality
search on it.

## 1. What to Measure

For the baseline float model and the final Top 3 recommended profiles (not
every intermediate iteration):

*   **Average inference latency (ms)** over ≥ 50 warm runs, after ≥ 5 warmup
    runs.
*   **Peak RSS memory (MB)** during inference.
*   For LLMs: report `prefill` latency (per prompt) and `decode` latency (per
    token) SEPARATELY — a single blended number is meaningless.

## 2. Host-Machine Benchmarking (Default)

The LiteRT benchmark tool provides consistent numbers without a device:

```bash
# Install once: pip install ai-edge-litert
# The benchmark_model binary is also distributed with LiteRT releases.
benchmark_model \
  --graph=model/quantized/dynamic_int8_selective_ver2.tflite \
  --num_threads=4 \
  --num_runs=50 \
  --warmup_runs=5 \
  --use_xnnpack=true \
  --report_peak_memory_footprint=true
```

Pin `--num_threads` to the same value across ALL configurations; thread count
variation swamps quantization effects.

## 3. On-Device Benchmarking (When Hardware Is Available)

Host CPU numbers do not transfer to mobile NPUs/DSPs. If the user has a device
attached (e.g. via `adb`):

```bash
adb push model/quantized/dynamic_int8_selective_ver2.tflite /data/local/tmp/
adb push benchmark_model /data/local/tmp/ && adb shell chmod +x /data/local/tmp/benchmark_model
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/dynamic_int8_selective_ver2.tflite \
  --num_threads=4 --num_runs=50 --warmup_runs=5 \
  --use_gpu=false  # toggle delegates explicitly, one at a time
```

Benchmark each delegate (CPU/XNNPACK, GPU, NNAPI/QNN) as a SEPARATE
measurement — delegates differ wildly in which quantization layouts they
accelerate.

## 4. Interpreting Results

*   **Quantize/dequantize boundary cost**: If a selective model with float
    skips is SLOWER than the baseline INT8 model despite being smaller,
    scattered precision transitions are the likely culprit. Revisit
    `regex_targeting.md` block isolation before blaming the recipe.
*   **Reporting**: Add a `Latency (ms)` column to the Pareto comparison table
    in the final report, and render the optional latency Pareto figure per
    `pareto_curve_plotting.md` §5. Always state the exact hardware, delegate,
    and thread count next to any latency number.
