# AI Edge Quantizer

**AI Edge Quantizer (AEQ)** is a flexible, high-performance post-training
quantization (PTQ) toolkit designed for
[LiteRT](https://ai.google.dev/edge/litert) (formerly TensorFlow Lite) and
[LiteRT-LM](https://ai.google.dev/edge/litert-lm). It enables developers
to optimize resource-intensive models (vision models, LLMs, and GenAI
pipelines) for edge deployment on mobile CPUs, GPUs, and NPUs.

## Key Features

* **Selective Quantization**: Target specific operations or layer subgraphs
  using regex scopes (e.g., FullyConnected Ops in FeedForward layers and leaving
  all other Ops as float).

* **Mixed-Precision Quantization**: Mix precision schemes across layers (e.g.,
  INT4 weights for FullyConnected in FeedForward layers but INT8 weights in
  Attention layers).

* **Advanced Quantization Algorithms**:
  * **Blockwise Quantization** (Block sizes: 32, 64, 128, 256)
  * **Hadamard Transformations** to suppress outlier activations and preserve
    accuracy for INT4/INT2 schemes
  * **GPTQ & OCTAV** optimization algorithms

* **Full Integer Quantization (Static Range, SRQ)**: INT8/INT16 activations
with INT8/INT4 weights, required for mobile hardware.

* **Integrated Numerical Validation**: Built-in tensor-level distortion
  analysis (MSE, SNR, Cosine Similarity, KL Divergence) compatible with
  [Model Explorer](https://ai.google.dev/edge/model-explorer).

## Build Status

Build Type         |    Status     |
-----------        | --------------|
Unit Tests (Linux) | [![Unit Tests Status Badge](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_unittests.yml) |
Nightly Release    | [![Nightly Release Status Badge](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_release.yml) |
Nightly Colab      | [![Nightly Colab Status Badge](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_colabs.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-quantizer/actions/workflows/nightly_colabs.yml) |

## Installation

### Requirements and Dependencies

 * Python versions: 3.10, 3.11, 3.12, 3.13
 * Operating system: Linux, MacOS
 * LiteRT: `ai-edge-litert-nightly`

### Install

Nightly PyPi package:

```bash
pip install ai-edge-quantizer-nightly
```

## Quick Start

The quantizer requires two inputs:

1. An unquantized source LiteRT (FP32 data type in the FlatBuffer
   format with `.tflite` extension) / LiteRT-LM (with `.litertlm` extension)
   model
2. A quantization recipe (details below)

and outputs a quantized LiteRT/LiteRT-LM model that's ready for deployment on
edge devices.

### Command Line (`aeq`)
Quantize a model directly from your terminal:

```bash
# Quantize standard .tflite model
aeq --model_file="path/to/input.tflite" \
    --recipe=dynamic_wi8_afp32 \
    --output_dir="/path/to/output"

# Quantize a .litertlm LLM container
aeq --model_file="path/to/gemma.litertlm" \
    --recipe=gemma4_mixed48 \
    --output_dir="/path/to/output"
```

### Python API
```python
from ai_edge_quantizer import quantizer, recipe

# 1. Initialize quantizer
qt = quantizer.Quantizer("path/to/model.tflite")
# 2. Load a ready-to-use recipe (e.g., dynamic int8 weights with float32
# activations).
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
# Quantize and export.
qt.quantize().export_model("path/to/quantized_model.tflite")
```

Please see the [getting started colab](colabs/getting_started.ipynb) for the
simplest quick start guide on those steps, and the
[selective quantization colab](colabs/selective_quantization_isnet.ipynb) for
more details on advanced features.

## Hardware & Recipe Decision Guide

Generally, we recommend dynamic quantization for CPU/GPU deployment and static
quantization for NPU deployment:

| Target Hardware | Recommended Recipe | Precision | Activation Calibration? |
| :--- | :--- | :---: | :---: |
| **CPU/GPU** | `dynamic_wi8_afp32` | Int8 Weight / FP32 Act | **No** |
| **NPU** | `static_wi8_ai8` / `static_wi8_ai16` | Int8 Weight / Int8 or Int16 Act | **Yes** (Requires calibration data, supported only via Python API) |

## Quantization Concepts & Methods

### LiteRT Model

Please refer to the [LiteRT documentation](https://ai.google.dev/edge/litert)
for ways to generate LiteRT models from Jax, PyTorch and TensorFlow. The input
source model should be an FP32 (unquantized) model in the FlatBuffer format with
`.tflite` extension.

### LiteRT-LM Model

Please refer to the
[LiteRT-LM documentation](https://ai.google.dev/edge/litert-lm) for details.

### Quantization Recipe

A quantization recipe encodes all information on how a model is to be
quantized, such as number of bits, data type, symmetry, scope name, etc.

Essentially, a quantization recipe is defined as a collection of commands of the
following type:

_“Apply **Quantization Algorithm X** on **Operator Y** under **Scope Z** with
**ConfigN**”._

For example:

_\"**Uniformly quantize** the **FullyConnected op** under scope **'dense1/'**
with **INT8 symmetric with Dynamic Quantization**"._

All the unspecified ops will be kept as FP32 (unquantized). The scope of an
operator in TFLite is defined as the output tensor name of the op, which
preserves the hierarchical model information from the source model (e.g., scope
in TF). The best way to obtain scope name is by visualizing the model with
[Model Explorer](https://ai.google.dev/edge/model-explorer).

### Quantization Methods

Currently, there are three ways to quantize an operator:

* **dynamic quantization (recommended)**: weights are quantized while
  activations remain in a float format and are not processed by AI Edge
  Quantizer (AEQ). The runtime kernel handles the on-the-fly quantization of
  these activations, as identified by `compute_precision=integer` and
  `explicit_dequantize=False`.
  * Pros: reduced model size and memory usage. Latency improvement due to
    integer computation. No sample data requirement (calibration).
  * Cons: on-the-fly quantization of activation tensors can affect model
    quality. Not supported in all hardware (e.g., some GPU and NPU).

* **weight only quantization**: only model weights are quantized, not
  activations. The actual operation (op) computation remains in float. The
  quantized weight is explicitly dequantized before being fed into the op, by
  inserting a dequantize op between the quantized weight and the consuming op.
  To enable this, `compute_precision` will be set to `float` and
  `explicit_dequantize` to `True`.
  * Pros: reduced model size and memory usage. No sample data requirement
    (calibration). Usually has the best model quality.
  * Cons: no latency benefit (may be worse) due to float computation with
    explicit dequantization.

* **static quantization**: both weights and activations are quantized. This
  requires a calibration phase to estimate quantization parameters of runtime
  tensors (activations).
  * Pros: reduced model size, memory usage, and latency.
  * Cons: requires sample data for calibration. Imposing static quantization
    parameters (derived from calibration) on runtime tensors can compromise
    quality.

We include commonly used recipes in [recipe.py](ai_edge_quantizer/recipe.py).
This is demonstrated in the
[getting started colab](colabs/getting_started.ipynb) example. Advanced users can
build their own recipe through the quantizer API.

## Quantization Workflow

Quantizing a model with AI Edge Quantizer follows a structured 7-step lifecycle:

1. Load Model
2. Load / Configure Recipe
3. **[Static recipes only]** Calibrate
4. Quantize & Export
5. Validate Accuracy
6. Visualize
7. Deploy

**Note on CLI Usage:** The `aeq` command-line tool executes
**Steps 1, 2, and4** in a single command for recipes that do not require
calibration (Dynamic Range, Weight-Only):

```bash
aeq --model_file="path/to/model.tflite" \
    --recipe=dynamic_wi8_afp32 \
    --output_dir="/path/to/output"
```

For Full-Integer Static Quantization requiring representative calibration
datasets (**Step 3**), use the **Python API**.

Detailed examples on **Steps 1-4** with **Python API** can be found in
[quantize_toy_model.py](ai_edge_quantizer/examples/mnist/quantize_toy_model.py).

### Step 1: Initialize Quantizer with Source Model

Load an unquantized FP32 `.tflite` model or a `.litertlm` generative model
bundle:

```python
from ai_edge_quantizer import quantizer

qt = quantizer.Quantizer("path/to/model.tflite")
```

### Step 2: Choose & Load a Quantization Recipe

Load a ready-to-use recipe (e.g., static int8 quantization) or custom recipe:

```python
from ai_edge_quantizer import recipe

qt.load_quantization_recipe(recipe.static_wi8_ai8())
```

#### Supported Operators and Recipes

Please refer to the [Operator Coverage](#operator-coverage) section for more
details on supported operators and configurations for each recipe.

#### Advanced Recipes & Customization

There are many ways the user can configure and customize the quantization recipe
beyond using a template in [recipe.py](ai_edge_quantizer/recipe.py). For example,
the user can configure the recipe to achieve these features:

* Selective quantization (exclude selected ops from being quantized)
* Flexible mixed scheme quantization (mixture of different precision, compute
  precision, scope, op, config, etc)
* 4-bit weight quantization
* Advanced algorithms (e.g., Hadamard Rotation, OCTAV)

The [selective quantization colab](colabs/selective_quantization_isnet.ipynb)
shows some of these more advanced features.

For specifics of the recipe schema, please refer to the `OpQuantizationRecipe`
in [recipe_manager.py](ai_edge_quantizer/recipe_manager.py).

For advanced usage involving mixed quantization, the following API may be
useful:

* Use `Quantizer:load_quantization_recipe()` in
  [quantizer.py](ai_edge_quantizer/quantizer.py) to load a custom recipe.
* Use `Quantizer:update_quantization_recipe()` in
  [quantizer.py](ai_edge_quantizer/quantizer.py) to extend or override
  specific parts of the recipe.

### Step 3: Calibrate with Representative Data (Static Quantization Only)

Static range quantization (e.g., `static_wi8_ai8`, `static_wi8_ai16`) quantizes
both weights and activations into integers, requiring a calibration phase with
representative sample data to calculate quantization statistics values (QSVs).

AI Edge Quantizer offers two calibration modes and two calibration APIs:

#### Calibration Modes

Both calibration modes use XNNPACK acceleration for inference during
calibration.

* **`CALIBRATION_PRESERVE_ALL_TENSORS` (Default)**: Preserves all intermediate
  tensors in memory, allowing Python-level tensor inspection and collecting
  arbitrary statistics across all layers. Recommended when using algorithms that
  require custom activation statistics (such as GPTQ) or when inspecting
  intermediate activations.
* **`CALIBRATION_PROFILER_BASED` (Recommended, except for GPTQ)**: Uses TFLite's
  internal C++ profiler-based calibration. It executes directly in C++ with
  lower memory overhead and is significantly faster. Currently, only min/max
  collection is supported by this mode, so it can be utilized for all
  quantization algorithms except GPTQ.

#### Calibration API: `Quantizer.calibrate()`

A calibration API managed directly by the `Quantizer` class. Requires to
organize calibration samples into a dictionary mapping signature keys (e.g.,
`'serving_default'`) to lists of input dictionaries. The `Quantizer.calibrate()`
runs the model across the dataset to compute statistics in a single call:

```python
from ai_edge_quantizer import calibrator, quantizer
import numpy as np

qt = quantizer.Quantizer("path/to/model.tflite")
qt.load_quantization_recipe("static_wi8_ai8")

# 1. Prepare representative calibration dataset.
calibration_data = {
    "serving_default": [
        {
            "input_tensor_name": np.random.uniform(
                -1.0, 1.0, size=(1, 28, 28, 1)
            ).astype(np.float32)
        }
        for _ in range(256)
    ]
}
# 2. Run calibration with the desired mode.
calibration_result = qt.calibrate(
    calibration_data,
    mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
)
```

#### Calibration API: `CalibrationInterpreter`

A calibration API designed as a drop-in replacement for the standard TFLite
`Interpreter`. Its primary advantage is that an existing model inference or
evaluation pipeline can be used directly for calibration.

As the existing evaluation loop runs inference via signature runners
(`get_signature_runner()`), the `CalibrationInterpreter` automatically tracks
and accumulates activation statistics. Once evaluation finishes, extract the
accumulated statistics and pass them to the `Quantizer`:

```python
from ai_edge_quantizer import calibrator, quantizer
import numpy as np

# 1. Initialize CalibrationInterpreter (drop-in replacement for tflite::Interpreter).
interpreter = calibrator.CalibrationInterpreter(
    "path/to/model.tflite",
    mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
)
runner = interpreter.get_signature_runner("serving_default")

# 2. Reuse the existing inference/evaluation loop.
for samples in dataset:
  runner(input_1=sample)  # input_1 is the input tensor name for this model.

# 3. Extract accumulated calibration statistics.
calibration_result = interpreter.get_calibration_results()
```

### Step 4: Quantize & Export the Model

Execute the quantization engine and export the resulting model:

```python
if qt.need_calibration:
  # For recipes that require calibration (e.g., static quantization):
  quantized_model = qt.quantize(calibration_result=calibration_result)
else:
  quantized_model = qt.quantize()
quantized_model.export_model("/path/to/output/quantized_model.tflite")
```

### Step 5: Validate Numerical Accuracy

Quantizing a model inherently introduces numerical noise. After calling
`qt.quantize()`, you can verify the mathematical distortion between the float
baseline and the quantized model using the built-in `validate()` method, which
returns a single `ComparisonResult` object mapping nodes to their error metric
values. You can print them or automatically save them to Model Explorer JSON
files:

```python
# 1. Default validation (evaluates MSE metric by default)
comparison_results = qt.validate(test_data=sample_data)
print(
    "Per-layer metrics:",
    comparison_results.get_all_tensor_results(),
)

# 2. Multi-metric validation (save all metrics and validation json data directly)
comparison_results = qt.validate(
    test_data=sample_data,
    error_metrics=[
        quantizer.ValidationErrorMetric.MSE,
        quantizer.ValidationErrorMetric.SNR,
    ],
    save_folder='/tmp/',
)
all_results = comparison_results.get_all_tensor_results()
for tensor_name, metrics in all_results.items():
  print(
      f"Tensor: {tensor_name} "
      f"- MSE: {metrics.get(quantizer.ValidationErrorMetric.MSE.value, 0.0):.6f} "
      f"- SNR: {metrics.get(quantizer.ValidationErrorMetric.SNR.value, 0.0):.6f}"
  )
```

### Step 6: Visualize Models with Model Explorer

The best way to obtain exact operator scope names and visually compare tensor
shapes and quantization scales between baseline float and quantized graphs is
using [Model Explorer](https://ai.google.dev/edge/model-explorer).

To visualize two exported `.tflite` models side-by-side in your terminal, run:

```bash
model_explorer --models \
  "/path/to/baseline_float.tflite,/path/to/quantized_model.tflite"
```

### Step 7: Deploy on Edge Hardware

Please refer to the
[LiteRT deployment documentation](https://ai.google.dev/edge/litert/inference)
for ways to deploy a quantized LiteRT model.

## Operator Coverage

### Allowed Configurations for Available recipes

|     |     |     |     |     |     |     |     |     |     |    |    |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |--- |
| **Config** | | DYNAMIC_WI8_AFP32 | DYNAMIC_WI4_AFP32 | DYNAMIC_WI4_AFP32_BLOCKWISE | DYNAMIC_WI2_AFP32_BLOCKWISE | STATIC_WI8_AI8 | STATIC_WI8_AI16 | STATIC_WI4_AI8 | STATIC_WI4_AI16 | WEIGHTONLY_WI8_AFP32 | WEIGHTONLY_WI4_AFP32 |
|activation| num\_bits | None | None | None | None | 8 | 16 | 8 | 16 | None | None |
| | symmetric |None | None | None | None | [TRUE, FALSE] | TRUE | [TRUE, FALSE] | TRUE | None | None |
| | granularity |None | None | None | None | TENSORWISE | TENSORWISE | TENSORWISE | TENSORWISE | None | None |
| | dtype| None | None | None | None | INT | INT | INT | INT | None | None |
| weight | num\_bits | 8 | 4 | 4 | 2 | 8 | 8 | 4 | 4 | 8 | 4 |
| | symmetric | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | [TRUE, FALSE] | [TRUE, FALSE] |
| | granularity | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[BLOCKWISE_32, BLOCKWISE_64, BLOCKWISE_128, BLOCKWISE_256\] | \[BLOCKWISE_32, BLOCKWISE_64, BLOCKWISE_128, BLOCKWISE_256\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] | \[CHANNELWISE, TENSORWISE\] |
| | dtype | INT | INT | INT | INT | INT | INT | INT | INT | INT | INT |
| explicit\_dequantize | | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | FALSE | TRUE | TRUE |
| compute\_precision || INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | INTEGER | FLOAT | FLOAT |

### Quantization Support for Operators with Weights

|     |     |     |     |     |     |     |     |     |    |    |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |--- |
| **Config** | DYNAMIC_WI8_AFP32 | DYNAMIC_WI4_AFP32 | DYNAMIC_WI4_AFP32_BLOCKWISE | DYNAMIC_WI2_AFP32_BLOCKWISE | STATIC_WI8_AI8 | STATIC_WI8_AI16 | STATIC_WI4_AI8 | STATIC_WI4_AI16 | WEIGHTONLY_WI8_AFP32 | WEIGHTONLY_WI4_AFP32 |
|BATCH_MATMUL     |<div align="center"> &check; </div>|     |     |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|CONV_2D          |<div align="center"> &check; </div>|<div align="center"> &check; </div>|     |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|CONV_2D_TRANSPOSE|<div align="center"> &check; </div>|     |     |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |     |<div align="center"> &check; </div>|    |
|DEPTHWISE_CONV_2D|<div align="center"> &check; </div>|     |     |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|    |     |<div align="center"> &check; </div>|    |
|EMBEDDING_LOOKUP |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|     |     |     |    |     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|FULLY_CONNECTED  |<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|<div align="center"> &check; </div>|

### Quantization Support for Activations-Only Operators

|     |     |     |
| --- | --- | --- |
| **Config** | STATIC_WI8_AI8 | STATIC_WI8_AI16 |
|ADD              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|AVERAGE_POOL_2D  |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|BROADCAST_TO     |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|CONCATENATION    |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|DIV              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|DYNAMIC_UPDATE_SLICE|<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|EQUAL            |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|GATHER           |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|GATHER_ND        |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|GELU             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|HARD_SWISH       |<div align="center"> &check; </div>|     |
|LOGISTIC         |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|MAX_POOL_2D      |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|MAXIMUM          |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|MEAN             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|MIRROR_PAD       |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|MUL              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|NOT_EQUAL        |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|PACK             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|PAD              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|PADV2            |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|REDUCE_MIN       |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|RELU             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|RESHAPE          |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|RESIZE_BILINEAR  |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|RESIZE_NEAREST_NEIGHBOR|<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|RSQRT            |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SELECT           |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SELECT_V2        |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SLICE            |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SOFTMAX          |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SPACE_TO_DEPTH   |<div align="center"> &check; </div>|     |
|SPLIT            |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SQRT             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SQUARED_DIFFERENCE |<div align="center"> &check; </div>|     |
|STRIDED_SLICE    |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SUB              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|SUM              |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|TANH             |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|TRANSPOSE        |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
|UNPACK           |<div align="center"> &check; </div>|<div align="center"> &check; </div>|
