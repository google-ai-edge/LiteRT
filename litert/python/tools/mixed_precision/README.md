# LiteRT Mixed Precision Tool

The LiteRT Mixed Precision tool allows you to perform mixed-precision conversions on LiteRT/TFLite models. Specifically, it enables you to convert a model's weights and activations to float16 (FP16) while selectively keeping specific operations or layers in float32 (FP32). This is useful for preserving model accuracy when certain operations are sensitive to precision loss in FP16.

## Features

- **FP16 Conversion**: Convert the entire model (inputs, outputs, weights, and operations) to FP16.
- **Selective FP32 Operations**: Specify specific operation types (e.g., `tfl.AddOp`, `tfl.Conv2DOp`) to remain in FP32.
- **Selective FP32 Layers/Names**: Specify name patterns or substrings (matching operation locations or StableHLO composite names) to keep matching layers in FP32.
- **Automatic Boundary Casting**: Inserts necessary `tfl.CastOp` operations at the boundaries between FP16 and FP32 operations to ensure type consistency.

## Usage

You can run the mixed precision tool from the command line using `mixed_precision_main`.

### Command Line Flags

- `--input_file`: Path to the input TFLite model. (Required)
- `--output_file`: Path to the output TFLite model. (Required)
- `--convert_to_fp16`: If true, converts the model to FP16.
- `--fp32_ops`: Comma-separated list of dialect-prefixed op types to keep in FP32 (e.g. `tfl.AddOp,tfl.CumsumOp`).
- `--fp32_names`: Comma-separated list of location/layer name substrings or composite name patterns to keep matching operations in FP32 (e.g., `layer_norm_1,my_custom_conv`).
- `--clamp_add_ops_after_rms_norm`: If true, clamps add operations after RMS norm to prevent overflow.

### Example Commands

#### 1. Standard FP16 Conversion
Converts the entire model to FP16, with default FP32 exceptions (e.g. RMS Norm):
```bash
mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_fp16.tflite \
  --convert_to_fp16
```

#### 2. Keep Specific Op Types in FP32
Converts the model to FP16 but keeps all `Add` and `Cumsum` operations in FP32:
```bash
mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_mixed.tflite \
  --convert_to_fp16 \
  --fp32_ops="tfl.AddOp,tfl.CumsumOp"
```

#### 3. Keep Specific Layers/Names in FP32
Converts the model to FP16 but keeps operations belonging to layers matching `"attention"` or `"custom_conv"` in FP32:
```bash
mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_mixed.tflite \
  --convert_to_fp16 \
  --fp32_names="attention,custom_conv"
```
