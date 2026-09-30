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

"""E2E tests for the quantizer using a toy MNIST model."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_interpreter_utils


_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  data = []
  for _ in range(num_samples):
    data.append(
        {'conv2d_input': _RNG.uniform(size=(1, 28, 28, 1)).astype(np.float32)}
    )
  return data


def _get_calibration_data(num_samples: int = 256):
  samples = _get_dummy_data(num_samples)
  data = {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples,
  }
  return data


def _get_test_data(num_samples: int = 8):
  return _get_calibration_data(num_samples)


class MNISTTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        'models/conv_fc_mnist.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  def _assert_all_tensors_integer(self, quantized_model):
    """Asserts that all tensors in the given model are integers."""
    quantized_model = flatbuffer_utils.read_model_from_bytearray(
        quantized_model
    )
    for graph in quantized_model.subgraphs:
      for op in graph.operators:
        if op.inputs is None:
          break
        for input_idx in op.inputs:
          tensor = graph.tensors[input_idx]
          tensor_type = flatbuffer_utils.type_to_name(tensor.type)
          op_name = flatbuffer_utils.opcode_to_name(
              quantized_model, op.opcodeIndex
          )
          if 'QUANTIZE' in op_name:
            continue  # Skip quantize ops as they may have float inputs.

          self.assertFalse(tensor_type.startswith('FLOAT'))

  @parameterized.product(
      test_case=[
          # Tuple holds compute precision and whether to use explicit
          # dequantize.
          (_ComputePrecision.FLOAT, True),  # WEIGHT_ONLY.
          (_ComputePrecision.INTEGER, False),  # DRQ.
      ],
      symmetric_weight=[True, False],
      granularity=[
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ],
  )
  def test_mnist_toy_model_int8_weight_only(
      self, test_case, symmetric_weight, granularity
  ):
    compute_precision, explicit_dequantize = test_case
    # asym DRQ is not supported
    # TODO: b/335254997 - fail when trying to use unsupported recipe.
    if compute_precision == _ComputePrecision.INTEGER and not symmetric_weight:
      return
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=symmetric_weight,
                granularity=granularity,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 55000)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE]
    )
    self._check_comparison_result(
        comparison_result,
        weight_tolerance=1e-2
        if granularity == qtyping.QuantGranularity.CHANNELWISE
        else 1e-1,
        logits_tolerance=1e-2
        if granularity == qtyping.QuantGranularity.CHANNELWISE
        else 1e-1,
        output_tolerance=1e-4
        if granularity == qtyping.QuantGranularity.CHANNELWISE
        else 1e-2,
    )

  @parameterized.product(
      test_case=[
          # Tuple holds compute precision and whether to use explicit
          # dequantize.
          (_ComputePrecision.FLOAT, True),  # WEIGHT_ONLY.
          (_ComputePrecision.INTEGER, False),  # DRQ.
      ],
      symmetric_weight=[True, False],
  )
  def test_mnist_toy_model_int4_weight_only(self, test_case, symmetric_weight):

    compute_precision, explicit_dequantize = test_case
    # Asym DRQ is not supported.
    # TODO: b/335254997 - Fail when trying to use unsupported recipe.
    if compute_precision == _ComputePrecision.INTEGER and not symmetric_weight:
      return
    self._quantizer.update_quantization_recipe(
        regex='.*',
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=4,
                symmetric=symmetric_weight,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 30000)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE]
    )
    self._check_comparison_result(
        comparison_result,
        weight_tolerance=1e-3,
        logits_tolerance=2,
        output_tolerance=1e-2,
    )

  def test_mnist_toy_model_fp16_weight_only(self):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING,
        operation_name=_OpName.FULLY_CONNECTED,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            compute_precision=_ComputePrecision.FLOAT,
        ),
    )
    _ = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(self._quantizer._result.quantized_model), 105000)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE]
    )
    self._check_comparison_result(
        comparison_result,
        weight_tolerance=1e-5,
        logits_tolerance=1e-5,
        output_tolerance=1e-5,
    )

  @parameterized.parameters(
      '../recipes/default_a8w8_recipe.json',
      '../recipes/default_a16w8_recipe.json',
  )
  def test_mnist_toy_model_full_integer(self, recipe_path):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertTrue(self._quantizer.need_calibration)
    calibration_result = self._quantizer.calibrate(_get_calibration_data())
    quant_result = self._quantizer.quantize(calibration_result)
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), 55000)

    comparison_result = self._quantizer.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
        test_data=_get_test_data(),
    )
    self._check_comparison_result(
        comparison_result,
        weight_tolerance=1e-2,
        logits_tolerance=1e-1,
        output_tolerance=1e-4,
    )

  # TODO: b/345503484 - Check weight tensor type of the quantized model.
  def _check_comparison_result(
      self,
      comparison_result,
      weight_tolerance,
      logits_tolerance,
      output_tolerance,
  ):
    # TODO: b/357959309 - Use comparison result directly for testing.
    _all_results = comparison_result.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      tensors_results = {k: v.get(metric, 0.0) for k, v in _all_results.items()}
      # Check weight tensors.
      conv_weight_mse = tensors_results['sequential/conv2d/Conv2D']
      self.assertLess(conv_weight_mse, weight_tolerance)
      fc1_weight_mse = tensors_results['arith.constant1']
      self.assertLess(fc1_weight_mse, weight_tolerance)
      fc2_weight_mse = tensors_results['arith.constant']
      self.assertLess(fc2_weight_mse, weight_tolerance)
      # Check logits.
      logits_mse = tensors_results['sequential/dense_1/MatMul']
      self.assertLess(logits_mse, logits_tolerance)
      # Check final output.
      output_mse = tensors_results['StatefulPartitionedCall:0']
      self.assertLess(output_mse, output_tolerance)

  @parameterized.parameters(
      '../recipes/default_a8w8_recipe.json',
      '../recipes/default_a16w8_recipe.json',
  )
  def test_mnist_toy_model_full_integer_requantize_success(self, recipe_path):
    partly_quantized_model_path = test_utils.get_path_to_datafile(
        '../tests/models/partly_quantized_mnist.tflite'
    )
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    qt = quantizer.Quantizer(partly_quantized_model_path, recipe_path)
    self.assertTrue(qt.need_calibration)
    calibration_result = qt.calibrate(_get_calibration_data())
    quant_result = qt.quantize(calibration_result)

    # Check if the output model has `integer-only` tensors.
    self._assert_all_tensors_integer(quant_result.quantized_model)


if __name__ == '__main__':
  absltest.main()
