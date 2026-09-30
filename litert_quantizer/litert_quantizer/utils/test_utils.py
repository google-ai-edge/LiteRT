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

"""Utils for tests."""

from collections.abc import Sequence
import inspect as _inspect
import pathlib as _pathlib
import sys as _sys
from typing import Optional, Union

from absl.testing import parameterized

from litert_quantizer import model_validator
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import tfl_interpreter_utils
from litert_quantizer.utils import validation_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_AlgorithmName = quantizer.AlgorithmName
_Numeric = Union[int, float]


DEFAULT_ACTIVATION_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=False,
    granularity=qtyping.QuantGranularity.TENSORWISE,
)
DEFAULT_WEIGHT_QUANT_SETTING = _TensorQuantConfig(
    num_bits=8,
    symmetric=True,
    granularity=qtyping.QuantGranularity.CHANNELWISE,
)


def get_static_activation_quant_setting(
    num_bits: int, symmetric: bool
) -> _TensorQuantConfig:
  return _TensorQuantConfig(
      num_bits=num_bits,
      symmetric=symmetric,
      granularity=qtyping.QuantGranularity.TENSORWISE,
  )


def get_static_op_quant_config(
    activation_config: _TensorQuantConfig = DEFAULT_ACTIVATION_QUANT_SETTING,
    weight_config: _TensorQuantConfig = DEFAULT_WEIGHT_QUANT_SETTING,
) -> _OpQuantConfig:
  return qtyping.OpQuantizationConfig(
      activation_tensor_config=activation_config,
      weight_tensor_config=weight_config,
      compute_precision=_ComputePrecision.INTEGER,
  )


def get_path_to_datafile(path):
  """Get the path to the specified file in the data dependencies.

  The path is relative to the file calling the function.

  Args:
    path: a string resource path relative to the calling file.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  data_files_path = _pathlib.Path(_inspect.getfile(_sys._getframe(1))).parent  # pylint: disable=protected-access
  return str((data_files_path / path).resolve())


class BaseOpTestCase(parameterized.TestCase):
  """Base class for op-level tests."""

  def quantize_and_validate(
      self,
      model_path: str,
      algorithm_key: _AlgorithmName,
      op_name: _OpName,
      op_config: _OpQuantConfig,
      num_validation_samples: int = 4,
      num_calibration_samples: Optional[int] = None,
      error_metrics: Optional[
          Sequence[validation_utils.ValidationErrorMetric]
      ] = None,
      min_max_range: Optional[tuple[_Numeric, _Numeric]] = None,
  ) -> model_validator.ComparisonResult:
    """Quantizes and validates the given model with the given configurations.

    Args:
      model_path: The path to the model to be quantized.
      algorithm_key: The algorithm to be used for quantization.
      op_name: The name of the operation to be quantized.
      op_config: The configuration for the operation to be quantized.
      num_validation_samples: The number of samples to use for validation.
      num_calibration_samples: The number of samples to use for calibration. If
        None then it will be set to num_validation_samples * 8.
      error_metrics: The error metrics to use for validation.
      min_max_range: The min and max of the input range.

    Returns:
      A ComparisonResult object containing the combined comparison results.
    """
    quantizer_instance = quantizer.Quantizer(model_path)
    quantizer_instance.update_quantization_recipe(
        algorithm_key=algorithm_key,
        regex='.*',
        operation_name=op_name,
        op_config=op_config,
    )
    if quantizer_instance.need_calibration:
      if num_calibration_samples is None:
        num_calibration_samples = num_validation_samples * 8
      calibration_data = tfl_interpreter_utils.create_random_normal_input_data(
          quantizer_instance._float_model_buffer,  # pylint: disable=protected-access  # pyrefly: ignore[bad-argument-type]
          num_samples=num_calibration_samples,
          min_max_range=min_max_range,
      )
      calibration_result = quantizer_instance.calibrate(calibration_data)  # pyrefly: ignore[bad-argument-type]
      quantization_result = quantizer_instance.quantize(calibration_result)
    else:
      quantization_result = quantizer_instance.quantize()
    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        quantization_result.quantized_model,  # pyrefly: ignore[bad-argument-type]
        num_samples=num_validation_samples,
        min_max_range=min_max_range,
    )
    return quantizer_instance.validate(test_data, error_metrics=error_metrics)  # pyrefly: ignore[bad-argument-type]

  def assert_model_size_reduction_above_min_pct(
      self,
      validation_result: model_validator.ComparisonResult,
      min_pct: float,
  ):
    """Checks the model size reduction (percentage) against the given expectation."""
    _, reduction_pct = validation_result.get_model_size_reduction()
    self.assertGreater(reduction_pct, min_pct)

  def assert_weights_errors_below_tolerance(
      self,
      validation_result: model_validator.ComparisonResult,
      weight_tolerance: float,
  ):
    """Checks the weight tensor numerical behavior against the given tolerance."""
    self.assertNotEmpty(validation_result.available_signature_keys())
    for signature_key in validation_result.available_signature_keys():
      signature_result = validation_result.get_signature_comparison_result(
          signature_key
      )
      for tensor_metrics in signature_result.constant_tensors.values():
        for metric_name, result in tensor_metrics.items():
          self.assertLess(
              result, weight_tolerance, f'Metric {metric_name} failed.'
          )

  def assert_output_errors_below_tolerance(
      self,
      validation_result: model_validator.ComparisonResult,
      output_tolerance: float,
  ):
    """Checks the output tensor numerical behavior against the given tolerance."""
    self.assertNotEmpty(validation_result.available_signature_keys())
    for signature_key in validation_result.available_signature_keys():
      signature_result = validation_result.get_signature_comparison_result(
          signature_key
      )
      for tensor_metrics in signature_result.output_tensors.values():
        for metric_name, result in tensor_metrics.items():
          self.assertLess(
              result, output_tolerance, f'Metric {metric_name} failed.'
          )

  def assert_quantization_accuracy_and_size(
      self,
      algorithm_key: _AlgorithmName,
      model_path: str,
      op_name: _OpName,
      op_config: _OpQuantConfig,
      expected_model_size_reduction: float,
      weight_tolerance: float = 1e-4,
      output_tolerance: float = 1e-4,
      min_max_range: Optional[tuple[_Numeric, _Numeric]] = None,
  ):
    """Check if the quantization is successful and the result is valid."""
    validation_results = self.quantize_and_validate(
        model_path=model_path,
        algorithm_key=algorithm_key,
        op_name=op_name,
        op_config=op_config,
        min_max_range=min_max_range,
    )
    with self.subTest(name='ModelSizeReduction'):
      self.assert_model_size_reduction_above_min_pct(
          validation_results, expected_model_size_reduction
      )
    with self.subTest(name='WeightsErrors'):
      self.assert_weights_errors_below_tolerance(
          validation_results, weight_tolerance
      )
    with self.subTest(name='OutputErrors'):
      self.assert_output_errors_below_tolerance(
          validation_results, output_tolerance
      )

  def assert_quantization_accuracy(
      self,
      algorithm_key: _AlgorithmName,
      model_path: str,
      op_name: _OpName,
      op_config: _OpQuantConfig,
      num_validation_samples: int = 4,
      num_calibration_samples: Optional[int] = None,
      output_tolerance: float = 1e-4,
      min_max_range: Optional[tuple[_Numeric, _Numeric]] = None,
  ):
    """Checks if the output errors after quantization are within the tolerance."""
    validation_results = self.quantize_and_validate(
        model_path=model_path,
        algorithm_key=algorithm_key,
        num_validation_samples=num_validation_samples,
        num_calibration_samples=num_calibration_samples,
        op_name=op_name,
        op_config=op_config,
        min_max_range=min_max_range,
    )
    self.assert_output_errors_below_tolerance(
        validation_results, output_tolerance
    )
