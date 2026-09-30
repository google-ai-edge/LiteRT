# Collection of Quantization Recipes

This directory contains a collection of quantization recipes that are commonly
used and supported on the TFLite runtime.

## Usage

These quantization recipes are provided in the JSON format. They can be used by
directly loading into a `Quantizer` instance as such:

```
qt = quantizer.Quantizer(f32_model_path)
qt.load_quantization_recipe(default_a8w8_recipe.json)
```

## Nomenclature

The filenames of recipes in this directory can be interpreted in the following
order:

`default` specifies the same scheme for all scopes and ops in the model.

`a<type>` means that the activation tensors are to be quantized to `type`. If
`type` is not prefixed by some letter, it indicates an integer type. E.g. `a8` =
int8 activation, `au8` = uint8 activation, and `af32` = float32 activation.

`w<type>` means that the weight tensors are to be quantized to `type`. If `type`
is not prefixed by some letter, it indicates an integer type. E.g. `a8` = int8
weight.

`[float|int]` specifies whether computation should be performed in floating
point or integer. If unspecified that means there's no ambiguity and that only
one computation type is relevant.

## Recipe Configuration

The schema of a quantization recipe is defined by the `OpQuantizationRecipe`
class that resides in `recipe_manager.py`. Generally, the fields expect string
names of the corresponding enums as defined within the `OpQuantizationRecipe`
class, with the following exceptions:

*   `regex` is target scope name in string. To specify a particular node, use
    the target op's output tensor name.

*   `algorithm_key` is desired quantization algorithm as defined by
    `AlgorithmName` in `algorithm_manager.py`.

## Customizing Input/Output Precision

If your hardware requires a specific data type (e.g., `uint8`, `int16`) for
model inputs/outputs, you can override the recipe after loading it using the
Python API. For example, to configure model inputs and outputs to use `uint8`:

```python
from litert_quantizer import qtyping

# Configure both INPUT and OUTPUT to use a specific type (UINT8 in this example)
for op in [qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT]:
  qt.update_quantization_recipe(
      regex='.*',
      operation_name=op,
      op_config=qtyping.OpQuantizationConfig(
          activation_tensor_config=qtyping.TensorQuantizationConfig(
              num_bits=8,  # Adjust bit width accordingly (e.g., 16 for INT16)
              symmetric=False,  # Adjust symmetric accordingly (e.g., True for INT16)
              granularity=qtyping.QuantGranularity.TENSORWISE,
              dtype=qtyping.TensorDataType.UINT,  # Change to your target type (e.g., TensorDataType.INT)
          )
      )
  )
```

## Examples

`sample_advanced_usage_recipe.json` is a simple example of configuring multiple
quantization schemes within a single recipe. This sample contains sample scope
names and should not be used as a general quantization recipe without
modification.

This sample demonstrates:

*   Quantizing different operations (separate entries for different `operation`)

*   4-bit quantization (`num_bits: 4`)

*   Mixed precision quantization with different compute precisions (separate
    entries for different `execution_mode`)

*   Selective quantization (opt out a specific node by specifying its output
    tensor name with `"algorithm_key": "no_quantize"`)
