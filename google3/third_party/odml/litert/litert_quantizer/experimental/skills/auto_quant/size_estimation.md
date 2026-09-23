# Model Compression Size Estimation

Unlike in-memory frameworks where you might calculate theoretical sizes using
PyTorch `nn.Module` object trees, the AI Edge Quantizer (AEQ) operates directly
on and outputs TFLite FlatBuffer files (`.tflite`). Therefore, the most accurate
and recommended way to measure compression size is to serialize the model to
disk and read its actual file size.

This document describes how to measure sizes in standard AEQ workflows and
provides context on the theoretical formulas.

## Quick usage (Actual File Size)

```python
import os
from ai_edge_quantizer import quantizer, recipe

# 1) Get the baseline size (using the float32 target)
float_model_path = "/path/to/isnet_float.tflite"
baseline_size_mb = os.path.getsize(float_model_path) / (1024 * 1024)
print(f"Baseline (FP32) Size: {baseline_size_mb:.2f} MB")

# 2) Quantize and export
recipe_mode = recipe.dynamic_wi8_afp32()
qt = quantizer.Quantizer(float_model=float_model_path)
qt.load_quantization_recipe(recipe_mode)
quantization_result = qt.quantize()

out_model_path = "/path/to/isnet_quantized.tflite"
quantization_result.export_model(out_model_path, overwrite=True)

# 3) Get the precise quantized size
size_mb = os.path.getsize(out_model_path) / (1024 * 1024)
ratio = baseline_size_mb / size_mb if size_mb > 0 else 0.0

print(f"Quantized Size: {size_mb:.2f} MB")
print(f"Compression Ratio: {ratio:.2f}x")
```

## Theoretical Size Formulas

If you wish to calculate theoretical sizes during model conversion or
architecture exploration, you can estimate them based on the weights. TFLite
stores models using the FlatBuffer schema, adding some metadata overhead, but
the raw binary payloads of the tensors strictly dominate the total file size.

By default, an unquantized source model typically maps network weights stored as
**fp32** (4 bytes per element).

### Quantized weight

```text
weight_bytes = numel * (n_bits / 8)
```

-   For **INT8 (8-bit)**: `numel * 1` byte.
-   For **INT4 (4-bit)**: `numel * 0.5` bytes (2 values closely packed).

### Scale and zero-point overhead

The number of scale and zero-point parameters depends on your configured
`granularity`:

-   **Per-tensor (`TENSORWISE`)**: 1 group (1 scale/ZP sequence across the
    entire weight tensor).
-   **Per-channel (`CHANNELWISE`)**: `weight.shape[O]` groups (1 scale/ZP
    sequence explicitly per output channel/filter).

```text
scale_bytes      = n_groups * 4            # Typically stored as 32-bit floats
zero_point_bytes = n_groups * 4            # Typically stored as 32-bit integers
```

For symmetric quantization, the zero-point is nominally zero but may still
explicitly exist in the underlying LiteRT serialization array configurations.

### Uncompressed parameters

Any tensors corresponding to biases, large FP32 embedding tables, and any
operator explicitly skipped via selective overrides (e.g.
`algorithm_key='no_quantize'`) remain stored in their original data type, which
is usually **fp32**.

```text
uncompressed_bytes = numel * 4
```

### Total Theoretical Bytes

```text
total_bytes = sum(weight_bytes)
            + sum(scale_bytes)
            + sum(zero_point_bytes)
            + sum(uncompressed_bytes)
            + flatbuffer_overhead        # Generally ~10-100s of KBs
total_mb    = total_bytes / (1024 * 1024)
```

## Reporting Sizes in Sweeps

When reporting your experimental comparisons across alternative quantization
methodologies, always report the final evaluated parameters:

-   **Absolute file size** in MB (`.tflite` payload disk metric).
-   **Compression ratio** vs the baseline float32.

Example Output:

```text
Config                           | Quality (Primary Metric) | Size (MB) | Ratio   | Notes
-------------------------------------------------------------------------------------------
fp32 baseline                    |                   0.0000 |     167.3 |    1.0x |
dynamic_int8                     |                   0.0005 |      42.1 |    4.0x |
dynamic_int4                     |                   0.0120 |      21.5 |    7.8x | Heavy precision loss
dynamic_int8_selective           |                   0.0001 |      48.6 |    3.4x | Skipped final output blocks
```

