# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sample end-to-end example for the MNIST toy model.

This script demonstrates the full power and flexibility of the LiteRT Quantizer
(AEQ) Python API by showcasing multiple quantization examples on an MNIST
model:
1. Programmatic Recipe: Using `recipe.py` to construct recipes directly in
   Python. This is the primary and recommended way to construct recipes.
2. Advanced PTQ: Selective layer matching and mixed precision quantization.
3. Static Quantization (SRQ): Full integer quantization requiring calibration
   data.
4. Blockwise Quantization: Sub-channel block-based quantization.
5. Hadamard Quantization: Using Hadamard rotations to estimate quantization
   parameters for better quality at lower bits.
6. External JSON Recipe: Reproducing quantization by loading a JSON recipe
   file (saved during model export) from disk.

Each example performs built-in side-by-side numerical validation against the
original float model and prints the validation score before running inference.
"""

from collections.abc import Sequence
import os
import random
from typing import Any

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf

import os
import io
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer import recipe
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_QuantGranularity = qtyping.QuantGranularity

# Choose a float model to be quantized. The example here is  Conv+FC MNIST
# model: 1 Conv layer, 1 pooling layer, 2 FC layers (conv_fc_mnist.tflite)
# You can also generate your own model. Some examples can be found in
# test/models/generate_mnist_test_model.py.
_FLOAT_MODEL_PATH = flags.DEFINE_string(
    'float_model_path',
    test_utils.get_path_to_datafile('../../tests/models/conv_fc_mnist.tflite'),
    'The trained floating point MNIST toy model TFLite flatbuffer path.',
)
_IMG_PATH = flags.DEFINE_string(
    'img_path',
    test_utils.get_path_to_datafile('data/sample6.png'),
    'Path to the MNIST image to be predicted.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/',
    'Directory to save the quantized model and recipe.',
)


def _check_user_inputs() -> None:
  """Check if the user input paths exist and create the save path if necessary."""
  if not os.path.exists(_FLOAT_MODEL_PATH.value):  # pyrefly: ignore[bad-argument-type]
    raise ValueError(
        'Model file does not exist. Please check the .tflite model path.'
    )
  if not os.path.exists(_IMG_PATH.value):  # pyrefly: ignore[bad-argument-type]
    raise ValueError('Image file does not exist. Please check the image path.')

  os.makedirs(_OUTPUT_DIR.value)


def _get_calibration_data(
    num_samples: int = 256,
) -> dict[str, list[dict[str, Any]]]:
  """Generate random dummy calibration data.

  The calibration data is a list of dictionaries, each of which contains
  input data for a single calibration step. The key is the signature input
  tensor name and the value is the input tensor numpy array.

  Args:
      num_samples: Number of samples to generate.

  Returns:
      A dictionary mapping signature keys to calibration sample lists.
  """
  (x_train_raw, _), _ = tf.keras.datasets.mnist.load_data()
  # Normalize pixel values to 0-1 and cast to float32.
  x_train_processed = (x_train_raw / 255.0).astype(np.float32)
  x_train_reshaped = x_train_processed.reshape([-1, 28, 28, 1])
  calibration_samples = [
      {'conv2d_input': random.choice(x_train_reshaped).reshape([-1, 28, 28, 1])}
      for _ in range(num_samples)
  ]
  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }


def _quantize_validate_save_and_return(
    qt: quantizer.Quantizer,
    output_dir: str,
    model_name: str,
    calibration_result: Any = None,
) -> bytes | None:
  """Quantize, validate, save the model artifact, and return bytes.

  Demonstrates calling qt.validate() to perform side-by-side numerical
  comparison between the quantized and float models.
  If test_data is None, qt.validate() generates random normal inputs.

  The `error_metrics` argument supports:
  - `None` (default) to evaluate `quantizer.ValidationErrorMetric.MSE` and
    return it in a dictionary.
  - A list of `quantizer.ValidationErrorMetric` enums (e.g.
    `[quantizer.ValidationErrorMetric.MSE,
    quantizer.ValidationErrorMetric.SNR]`) to evaluate the specified metrics and
    return a dictionary mapping metric names to their ComparisonResult objects.

  Currently supported error metrics include:
  1. `quantizer.ValidationErrorMetric.MSE`: Mean Squared Error.
  2. `quantizer.ValidationErrorMetric.MEDIAN_DIFF_RATIO`: Median Absolute
  Difference Ratio.
  3. `quantizer.ValidationErrorMetric.COSINE_SIMILARITY`: Cosine Similarity.
  4. `quantizer.ValidationErrorMetric.KL_DIVERGENCE`: Kullback-Leibler
  divergence.
  5. `quantizer.ValidationErrorMetric.SNR`: Signal-to-noise ratio.
  More details on these metrics (definitions and implementations) can be
  found in `utils/validation_utils.py`.

  Args:
      qt: The Quantizer instance before quantize() is called.
      output_dir: Directory to save the quantized model.
      model_name: Name of the model artifact to save.
      calibration_result: Optional calibration result for static quantization.

  Returns:
      Quantized model in bytes. (Note: AEQ produces `bytearray` buffers
      in-memory for efficient mutability. We cast to immutable `bytes` here to
      satisfy strict C++ type-checking in the TFLite Python Interpreter).
  """
  quant_result = qt.quantize(calibration_result)

  # Calling validate() without arguments runs the default metric (MSE)
  # and returns a ComparisonResult object.
  validation_results = qt.validate(
      save_folder=output_dir, model_name=model_name
  )
  print(
      f'\n--- Numerical Validation Results with default metric ({model_name})'
      ' ---'
  )
  sig_result = validation_results.get_signature_comparison_result()
  for tensor_name, error_vals in sig_result.output_tensors.items():
    for metric_name, error_val in error_vals.items():
      print(
          f'  Output Tensor: {tensor_name}, Metric: {metric_name}, Value:'
          f' {error_val:.6f}'
      )
  print('------------------------------------------\n')

  # 2) Calling validate() with a list of metrics runs those selected metrics
  # and likewise returns a ComparisonResult object containing all of them.
  multi_validation_results = qt.validate(
      error_metrics=[
          quantizer.ValidationErrorMetric.MSE,
          quantizer.ValidationErrorMetric.SNR,
      ],
      save_folder=output_dir,
      model_name=model_name,
  )
  print(
      f'\n--- Numerical Validation Results Multiple Metrics ({model_name}) ---'
  )
  sig_result = multi_validation_results.get_signature_comparison_result()
  for tensor_name, error_vals in sig_result.output_tensors.items():
    for metric_name, error_val in error_vals.items():
      print(
          f'  Output Tensor: {tensor_name}, Metric: {metric_name}, Value:'
          f' {error_val:.6f}'
      )
  print('------------------------------------------\n')

  quant_result.save(output_dir, model_name=model_name, overwrite=True)
  quantized_model = quant_result.quantized_model
  return (
      bytes(quantized_model)
      if isinstance(quantized_model, bytearray)
      else quantized_model
  )


def quantize_with_programmatic_recipe(
    *,
    float_model_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model using a programmatic recipe from recipe.py.

  EXAMPLE 1: Programmatic Recipe.

  Demonstrates how users can import and apply pre-packaged Python recipe
  builders. Using `recipe.py` to construct and load recipes is the primary
  and recommended way to use the LiteRT Quantizer.

  More details on how to create your own recipe builders and naming convention
  can be found in `recipe.py`.

  Args:
      float_model_path: Path to the float model.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  qt = quantizer.Quantizer(float_model_path)

  # Load a pre-defined dynamic range int8 recipe directly from recipe.py.
  qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())

  return _quantize_validate_save_and_return(
      qt, output_dir, model_name='mnist_prog_recipe'
  )


def quantize_with_advanced_ptq(
    *,
    float_model_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model demonstrating advanced PTQ features.

  EXAMPLE 2: Advanced PTQ (Mixed Precision & Selective Quantization).

  Showcases mixed-precision quantization and selective layer matching by
  applying different quantization configs to different operations/regex scopes.

  Args:
      float_model_path: Path to the float model.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  qt = quantizer.Quantizer(float_model_path)

  # Apply 8-bit weight-only quantization to Conv2D layers.
  qt.add_weight_only_config(
      regex='.*',
      operation_name=_OpName.CONV_2D,
      num_bits=8,
  )
  # Apply 4-bit dynamic range quantization to FullyConnected layers.
  qt.add_dynamic_config(
      regex='.*',
      operation_name=_OpName.FULLY_CONNECTED,
      num_bits=4,
  )

  return _quantize_validate_save_and_return(
      qt, output_dir, model_name='mnist_advanced_ptq'
  )


def quantize_with_static_range(
    *,
    float_model_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model using full integer static quantization.

  EXAMPLE 3: Static Range Quantization (SRQ).

  Demonstrates static quantization requiring calibration data to determine
  activation min/max ranges. Explains how calibration data must match TFLite
  signatures.

  Args:
      float_model_path: Path to the float model.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  qt = quantizer.Quantizer(float_model_path)

  # Load a static int8 recipe (int8 weights and int8 activations).
  qt.load_quantization_recipe(recipe.static_wi8_ai8())

  # SRQ requires calibration data to calculate activation ranges.
  # Calibration_data is structured as a dictionary mapping signature keys
  # (e.g., DEFAULT_SIGNATURE_KEY) to lists of input sample dicts.
  calibration_data = _get_calibration_data(num_samples=256)
  calibration_result = qt.calibrate(calibration_data)  # pyrefly: ignore[bad-argument-type]

  return _quantize_validate_save_and_return(
      qt,
      output_dir,
      model_name='mnist_static_quant',
      calibration_result=calibration_result,
  )


def quantize_with_blockwise(
    *,
    float_model_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model using sub-channel blockwise quantization.

  EXAMPLE 4: Block-based Quantization Granularity.

  Demonstrates block-based quantization granularity (e.g., 4-bit weights with
  block size 32), which is widely used for LLMs and advanced accelerators.

  Args:
      float_model_path: Path to the float model.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  qt = quantizer.Quantizer(float_model_path)

  # Apply 4-bit dynamic range quantization with block size 32.
  qt.load_quantization_recipe(recipe.dynamic_wi4b32_afp32())

  return _quantize_validate_save_and_return(
      qt, output_dir, model_name='mnist_blockwise'
  )


def quantize_with_hadamard(
    *,
    float_model_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model using Hadamard rotations.

  EXAMPLE 5: Hadamard Quantization.

  Demonstrates using Hadamard rotations (`hr`) to estimate quantization
  parameters. This is typically used to achieve better model quality at lower
  bit-widths (e.g., 4-bit weights). More details regarding Hadamard rotations
  can be found in `algorithms/uniform_quantize/hadamard_rotation.py`.

  In this example, we use the existing recipe `dynamic_wi4c_hr_afp32` from
  `recipe.py`. To configure advanced algorithm parameters such as
  `max_hadamard_size`, you can use `qt.add_quantization_config` to provide an
  explicit `_OpQuantConfig`.
  For example:
  ```python
  qt.add_quantization_config(
      regex='.*',
      operation_name=_OpName.FULLY_CONNECTED,
      algorithm_key='DECOMPOSED_HADAMARD_ROTATION',
      op_config=_OpQuantConfig(
          weight_tensor_config=_TensorQuantConfig(
              num_bits=4,
              symmetric=True,
              granularity=_QuantGranularity.CHANNELWISE,
              algorithm_params={'max_hadamard_size': 1024},
          ),
          compute_precision=_ComputePrecision.INTEGER,
      ),
  )
  ```

  Args:
      float_model_path: Path to the float model.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  qt = quantizer.Quantizer(float_model_path)

  # Apply 4-bit dynamic range quantization with Hadamard rotations.
  qt.load_quantization_recipe(recipe.dynamic_wi4c_hr_afp32())

  return _quantize_validate_save_and_return(
      qt, output_dir, model_name='mnist_hadamard'
  )


def quantize_with_recipe_json(
    *,
    float_model_path: str,
    recipe_json_path: str,
    output_dir: str,
) -> bytes | None:
  """Quantize the float model using an external JSON recipe file.

  EXAMPLE 6: External JSON Recipe (Reproducing Experiments).

  While using `recipe.py` is the primary and recommended way to construct and
  load recipes, JSON recipe files are useful for repeated experiments. A `.json`
  recipe is automatically saved during model export (e.g., when
  `quant_result.save()` is called). This example demonstrates how to reproduce
  quantization using a saved JSON recipe file.

  The external JSON recipe file specifies the quantization configuration for the
  model. The JSON file should be a list of dictionaries. This can be a single
  recipe rule or a stacked recipe (a list of multiple rule dicts evaluated
  sequentially, where later rules take precedence). Expected keys in each rule:
    - regex: A regular expression to match the model tensors.
    - operation: The operation type to match.
    - algorithm_key: The quantization algorithm to use.
    - op_config: The quantization configuration for the operation.
      - activation_tensor_config: The quantization configuration for the
        activation tensor.
      - weight_tensor_config: The quantization configuration for the weight
        tensor.
      - compute_precision: The quantization precision of the computation.

  A collection of common recipes can also be found in the top-level directory
  `recipes/`. Naming convention for recipes can be found in `recipe.py`.

  Args:
      float_model_path: Path to the float model.
      recipe_json_path: Path to the quantization recipe file.
      output_dir: Directory to save the quantized model.

  Returns:
      Quantized model in bytes.
  """
  # 1) Instantiate a new quantizer with the source float model path.
  qt = quantizer.Quantizer(float_model_path)

  # 2) Initialize the quantization recipe from a JSON file.
  qt.load_quantization_recipe(recipe_json_path)

  # 3) Calibrate the model if necessary. This is only necessary for quantization
  #    modes that involve integer computation. `qt.need_calibration` can be used
  #    to check if the quantization recipe needs calibration.
  calibration_result = (
      qt.calibrate(_get_calibration_data()) if qt.need_calibration else None  # pyrefly: ignore[bad-argument-type]
  )

  # 4) Quantize, validate, save, and return the model.
  return _quantize_validate_save_and_return(
      qt,
      output_dir,
      model_name='mnist_json_recipe',
      calibration_result=calibration_result,
  )


def inference(*, quantized_tflite: bytes, image_path: str) -> None:
  """Run inference on the quantized model.

  Args:
      quantized_tflite: Quantized model in bytes.
      image_path: Path to the image to be predicted.
  """

  def _read_img(img_path: str) -> np.ndarray:
    """Read MNIST image and normalize it.

    Args:
        img_path: Path to the image to be predicted.

    Returns:
        The normalized image.
    """
    image = Image.open(img_path)
    data = np.asarray(image, dtype=np.float32)
    if data.shape not in [(28, 28), (28, 28, 1)]:
      raise ValueError(
          'Invalid input image shape (MNIST image should have shape 28*28 or'
          ' 28*28*1)'
      )
    # Normalize the image if necessary.
    if data.max() > 1:
      data = data / 255.0
    return data.reshape((1, 28, 28, 1))

  tflite_interpreter = tfl_interpreter.Interpreter(
      model_content=quantized_tflite
  )
  tflite_interpreter.allocate_tensors()
  data = _read_img(image_path)
  tfl_interpreter_utils.invoke_interpreter_once(tflite_interpreter, [data])
  tflite_output_details = tflite_interpreter.get_output_details()[0]
  # Use `get_tensor_data` with `dequantize=True` to ensure that statically
  # quantized models (which output int8 values) have their outputs dequantized
  # back to float32 probabilities.
  result = tfl_interpreter_utils.get_tensor_data(
      tflite_interpreter, tflite_output_details, dequantize=True
  )
  category_probabilities = np.squeeze(result)
  predicted_category = np.argmax(category_probabilities)
  print(
      f'Model predicts the image as {predicted_category} with probability'
      f' {category_probabilities[predicted_category]}',
  )


def main(argv: Sequence[str]) -> None:
  """Run through various quantization examples and perform inference.

  This function demonstrates different ways to use the LiteRT Quantizer API,
  including quantizing with programmatic recipes, advanced PTQ, static range
  quantization, blockwise quantization, hadamard quantization, and JSON recipes.
  After each quantization, it performs a sample inference on an MNIST image.

  Args:
      argv: Command line arguments.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _check_user_inputs()

  print(
      '\n===================================================================='
  )
  print('1. Quantizing with programmatic recipe (from recipe.py)...')
  quantized_model_prog = quantize_with_programmatic_recipe(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      output_dir=_OUTPUT_DIR.value,
  )
  inference(quantized_tflite=quantized_model_prog, image_path=_IMG_PATH.value)  # pyrefly: ignore[bad-argument-type]

  print(
      '\n===================================================================='
  )
  print('2. Quantizing with advanced PTQ (mixed precision & selective)...')
  quantized_model_adv = quantize_with_advanced_ptq(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      output_dir=_OUTPUT_DIR.value,
  )
  inference(quantized_tflite=quantized_model_adv, image_path=_IMG_PATH.value)  # pyrefly: ignore[bad-argument-type]

  print(
      '\n===================================================================='
  )
  print('3. Quantizing with static range (requires calibration)...')
  quantized_model_static = quantize_with_static_range(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      output_dir=_OUTPUT_DIR.value,
  )
  inference(quantized_tflite=quantized_model_static, image_path=_IMG_PATH.value)  # pyrefly: ignore[bad-argument-type]

  print(
      '\n===================================================================='
  )
  print('4. Quantizing with blockwise granularity (sub-channel)...')
  quantized_model_block = quantize_with_blockwise(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      output_dir=_OUTPUT_DIR.value,
  )
  inference(quantized_tflite=quantized_model_block, image_path=_IMG_PATH.value)  # pyrefly: ignore[bad-argument-type]

  print(
      '\n===================================================================='
  )
  print(
      '5. Quantizing with Hadamard rotations (for better quality at lower'
      ' bits)...'
  )
  quantized_model_hadamard = quantize_with_hadamard(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      output_dir=_OUTPUT_DIR.value,
  )
  inference(
      quantized_tflite=quantized_model_hadamard, image_path=_IMG_PATH.value  # pyrefly: ignore[bad-argument-type]
  )

  print(
      '\n===================================================================='
  )
  print('6. Reproducing quantization with exported JSON recipe...')
  # Use the recipe saved during the programmatic recipe export (Example 1).
  saved_recipe_path = os.path.join(
      _OUTPUT_DIR.value, 'mnist_prog_recipe_recipe.json'
  )
  quantized_model_json = quantize_with_recipe_json(
      float_model_path=_FLOAT_MODEL_PATH.value,  # pyrefly: ignore[bad-argument-type]
      recipe_json_path=saved_recipe_path,
      output_dir=_OUTPUT_DIR.value,
  )
  inference(quantized_tflite=quantized_model_json, image_path=_IMG_PATH.value)  # pyrefly: ignore[bad-argument-type]
  print(
      '\n===================================================================='
  )
  print('All quantization examples completed successfully!')


if __name__ == '__main__':
  app.run(main)
