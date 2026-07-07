# LiteRT Mixed Precision Tool

The LiteRT Mixed Precision tool allows you to perform mixed-precision
conversions on LiteRT/TFLite models. Currently, the tool specifically supports
converting a model's weights and activations from float32 to float16 (FP16),
while selectively keeping specific operations or layers in float32 (FP32). This
is useful for preserving model accuracy when certain operations are sensitive to
precision loss in FP16.

## Features

-   **FP16 Conversion**: Convert the entire model (inputs, outputs, weights, and
    operations) to FP16.
-   **Selective FP32 Operations**: Specify specific operation types (e.g.,
    `tfl.AddOp`, `tfl.Conv2DOp`) to remain in FP32.
-   **Selective FP32 Layers/Names**: Specify name patterns or substrings
    (matching operation locations or StableHLO composite names) to keep matching
    layers in FP32.
-   **Automatic Boundary Casting**: Inserts necessary `tfl.CastOp` operations at
    the boundaries between FP16 and FP32 operations to ensure type consistency.

## Usage

You can run the mixed precision tool from the command line either by running it
as a Python module or using the installed entry point:

```bash
# Run as module
python -m litert.python.tools.mixed_precision.mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_fp16.tflite \
  --convert_to_fp16

# Run as binary (if installed in the pip package)
mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_fp16.tflite \
  --convert_to_fp16
```

### Command Line Flags

-   `--input_file`: Path to the input TFLite model. (Required)
-   `--output_file`: Path to the output TFLite model. (Required)
-   `--convert_to_fp16`: If true, converts the model to FP16.
-   `--fp32_ops`: Comma-separated list of dialect-prefixed op types to keep in
    FP32 (e.g. `tfl.AddOp,tfl.CumsumOp`).
-   `--fp32_names`: Comma-separated list of location/layer name substrings or
    composite name patterns to keep matching operations in FP32 (e.g.,
    `layer_norm_1,my_custom_conv`).
-   `--clamp_add_ops_after_rms_norm`: If true, clamps add operations after RMS
    norm to prevent overflow.

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

Converts the model to FP16 but keeps operations belonging to layers matching
`"attention"` or `"custom_conv"` in FP32:

```bash
mixed_precision_main \
  --input_file=model.tflite \
  --output_file=model_mixed.tflite \
  --convert_to_fp16 \
  --fp32_names="attention,custom_conv"
```

## Mixed Precision Workflows

Here are common scenarios for using the mixed precision tool:

### Scenario 1: FP16 Model has Undesired Precision Loss

If you ran your model in full FP16 and it met your performance target but
suffered from unacceptable accuracy loss:

1.  **Identify the sensitive operations or layers** that cause the precision
    loss (e.g., custom normalization layers, specific activation functions, or
    operations like `tfl.AddOp` or `tfl.CumsumOp`).
2.  **Re-convert the model** from the original FP32 model using
    `mixed_precision_main`, passing the sensitive operations to keep in FP32.

    *   To keep specific operations (by type) in FP32:

        ```bash
        mixed_precision_main \
          --input_file=model_fp32.tflite \
          --output_file=model_mixed.tflite \
          --convert_to_fp16 \
          --fp32_ops="tfl.AddOp,tfl.CumsumOp"
        ```

    *   To keep specific layers or locations (by name pattern) in FP32:

        ```bash
        mixed_precision_main \
          --input_file=model_fp32.tflite \
          --output_file=model_mixed.tflite \
          --convert_to_fp16 \
          --fp32_names="attention_layer,norm_layer"
        ```

3.  **Run on GPU** with the delegate precision set to FP32 (e.g.,
    `is_precision_loss_allowed=0` or `LiteRtDelegatePrecision::kFp32`). The GPU
    will execute the FP16 parts of the model in FP16, and the selected sensitive
    parts in FP32, recovering accuracy while keeping most of the performance
    gains.

### Scenario 2: Tuning Performance of an FP32 Model

If you have a fully functional model running in FP32 but need to optimize its
execution time:

1.  **Start with default mixed-precision conversion**. Run the tool with
    `--convert_to_fp16`. This will convert most of the model to FP16 but
    automatically keep operations that are highly sensitive to
    precision/overflow (such as RMS Norm) in FP32.

    ```bash
    mixed_precision_main \
      --input_file=model_fp32.tflite \
      --output_file=model_mixed.tflite \
      --convert_to_fp16
    ```

2.  **Profile the performance and accuracy** of the converted model on your
    target device using GPU options set to FP32 precision.

3.  **Refine the FP32 boundaries**:

    *   If the model's accuracy is too low, identify the failing nodes and keep
        them in FP32 using the `--fp32_ops` or `--fp32_names` flags.
    *   If the performance still has room for improvement and accuracy is
        robust, you can try reducing the set of operations kept in FP32.

## Running Mixed-Precision Models on GPU

When running a mixed-precision model on the GPU, you must configure the GPU
delegate/accelerator options to preserve precision for the operations you
selected to keep in FP32.

By default, the GPU delegate may run the entire model in FP16 precision
(ignoring the FP32 tensors in the model) to optimize for speed. To ensure that
the GPU delegate respects the FP32/FP16 boundaries in your mixed-precision
model, you must configure the delegate to compile with **FP32 precision**.

### GPU Delegate Options

Depending on the API you are using to run the model, apply the following
configuration:

#### LiteRT (C/C++ API)

Set the GPU Accelerator compilation options to `kFp32` precision:

```cpp
litert::GpuOptions gpu_options = litert::GpuOptions::Create();
gpu_options.SetPrecision(litert::GpuOptions::Precision::kFp32);
```
