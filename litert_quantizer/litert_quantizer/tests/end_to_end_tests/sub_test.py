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

"""E2E tests for the quantizer for model with sub."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import os
import io
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_inputs, num_samples):
  data = []
  for _ in range(num_samples):
    data.append({
        f'input_{i+1}': _RNG.uniform(size=(1, 32, 32)).astype(np.float32)
        for i in range(num_inputs)
    })
  return data


def _get_calibration_data(num_inputs, num_samples: int = 512):
  calibration_samples = _get_dummy_data(num_inputs, num_samples)
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _get_test_data(num_inputs, num_samples: int = 8):
  return _get_calibration_data(num_inputs, num_samples)


class SubTest(parameterized.TestCase):

  def _custom_setup(self, test_model_file):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        f'../models/{test_model_file}'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_sub_model_full_integer(self, recipe_path):
    self._custom_setup('single_sub.tflite')
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(
        _get_calibration_data(num_inputs=2)
    )
    _ = self._quantizer.quantize(calibration_result)
    # Skip model size check because the quantized model doesn't decrease as
    # there are no weights in the model file.

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
        test_data=_get_test_data(num_inputs=2),
    )
    self._check_comparison_result(
        comparison_result,
        output_tolerance=1e-4,
    )

  @parameterized.parameters(
      '../../recipes/default_a8w8_recipe.json',
      '../../recipes/default_a16w8_recipe.json',
  )
  def test_sub1_constant_input_model_full_integer(self, recipe_path):
    self._custom_setup('single_sub1_constant_input.tflite')
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(
        _get_calibration_data(num_inputs=1)
    )
    quant_result = self._quantizer.quantize(calibration_result)
    # Check model size.
    with open(self.float_model_path, 'rb') as f:
      float_model_bytearray = bytearray(f.read())
    self.assertLess(
        len(quant_result.quantized_model), len(float_model_bytearray)
    )

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
        test_data=_get_test_data(num_inputs=1),
    )
    self._check_comparison_result(
        comparison_result,
        output_tolerance=1e-4,
    )

  # TODO: b/345503484 - Check weight tensor type of the quantized model.
  def _check_comparison_result(
      self,
      comparison_result,
      output_tolerance,
  ):
    # TODO: b/357959309 - Use comparison result directly for testing.
    _all_results = comparison_result.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      comparison_result = {
          k: v.get(metric, 0.0) for k, v in _all_results.items()
      }
      # Check final output.
      output_mse = comparison_result['PartitionedCall:0']
      self.assertLess(output_mse, output_tolerance)


if __name__ == '__main__':
  absltest.main()
