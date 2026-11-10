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

"""E2E tests for the quantizer for model with transpose."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils

_OpExecutionMode = qtyping.OpExecutionMode
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(
    num_samples: int, dtype: np.dtype = np.float32
) -> list[dict[str, Any]]:
  data = []
  for _ in range(num_samples):
    data.append({'input_1': _RNG.uniform(size=(2, 3)).astype(dtype)})
  return data


def _get_calibration_data(
    num_samples: int = 128, dtype: np.dtype = np.float32
) -> list[dict[str, Any]]:
  calibration_samples = _get_dummy_data(num_samples, dtype)
  calibration_data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
  }
  return calibration_data


def _get_test_data(
    num_samples: int = 8, dtype: np.dtype = np.float32
) -> list[dict[str, Any]]:
  return _get_calibration_data(num_samples, dtype)


class SumTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/single_sum.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_quantized',
          recipe_path='../../recipes/default_a8w8_recipe.json',
          tensor_type=9,
          tol=1e-4,
      ),
      dict(
          testcase_name='int16_quantized',
          recipe_path='../../recipes/default_a16w8_recipe.json',
          tensor_type=7,
          tol=2.5,  # TODO(b/379757798): Update tolerance after bug is fixed.
      ),
  )
  def test_sum_model_full_integer(self, recipe_path, tensor_type, tol):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)

    data = _get_calibration_data()
    calibration_result = self._quantizer.calibrate(data)

    quantization_result = self._quantizer.quantize(calibration_result)

    # Check input/output tensor type.
    quantized_model = tfl_flatbuffer_utils.read_model(
        quantization_result.quantized_model
    )
    self.assertLen(quantized_model.subgraphs, 1)
    subgraph = quantized_model.subgraphs[0]
    subgraph_tensors = subgraph.tensors
    self.assertLen(subgraph.inputs, 1)
    input_tensor = subgraph_tensors[subgraph.inputs[0]]
    output_tensor = subgraph_tensors[subgraph.outputs[0]]
    # See schema_py_generated.py for type code.
    self.assertEqual(input_tensor.type, tensor_type)
    self.assertEqual(output_tensor.type, tensor_type)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
        test_data=_get_test_data(num_samples=1),
    )
    self._check_comparison_result(comparison_result, output_tolerance=tol)

  def _check_comparison_result(self, comparison_result, output_tolerance):
    # TODO: b/357959309 - Use comparison result directly for testing.
    _all_results = comparison_result.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      comparison_result = {
          k: v.get(metric, 0.0) for k, v in _all_results.items()
      }
      output_mse = comparison_result['PartitionedCall:0']
      self.assertLess(output_mse, output_tolerance)


if __name__ == '__main__':
  absltest.main()
