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

"""E2E tests for the quantizer for model with fully connected op."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils


_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig
_AlgorithmName = quantizer.AlgorithmName

_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile(
    '../models/dequantized_weights/'
)


# TODO: b/398335637 - Better ways to construct op configs for different cases.
class FullyConnectedTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = _OpName.FULLY_CONNECTED

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-5),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_weight_only_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  def test_blockwise_dequantized_weight_recovery_accuracy_and_size(self):
    model = 'blockwise_i4rangedvalues_fc.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.BLOCKWISE_32,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    weight_tolerance = 1e-4
    output_tolerance = 1.5e-1
    expected_model_size_reduction = 75
    self.assert_quantization_accuracy_and_size(
        _AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      algorithms=[
          _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          _AlgorithmName.OCTAV,
      ],
      granularity=[
          qtyping.QuantGranularity.BLOCKWISE_32,
          qtyping.QuantGranularity.BLOCKWISE_64,
          qtyping.QuantGranularity.BLOCKWISE_128,
      ],
  )
  def test_int4_blockwise_weight_only_quantization_raise_error(
      self, algorithms, granularity
  ):
    algorithm_key = algorithms
    model = 'tensor_i4rangedvalues_fc.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )

    # Blockwise quantization is not supported for weight-only quantization.
    error_msg = 'Unsupported op'
    with self.assertRaisesRegex(ValueError, error_msg):
      self.quantize_and_validate(
          model_path, algorithm_key, self._op_name, op_config
      )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-2),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_dynamic_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):

    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      algorithms=[
          _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          _AlgorithmName.OCTAV,
      ],
      granularity=[
          qtyping.QuantGranularity.BLOCKWISE_32,
          qtyping.QuantGranularity.BLOCKWISE_64,
          qtyping.QuantGranularity.BLOCKWISE_128,
      ],
  )
  def test_int4_blockwise_dynamic_quantization_accuracy_and_size_within_tolerance(
      self, algorithms, granularity
  ):
    algorithm_key = algorithms
    weight_tolerance = 1e-2
    output_tolerance = 1e-1
    model = 'tensor_i4rangedvalues_fc.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )

    expected_model_size_reduction = 65
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-1),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_a8static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=8,
            symmetric=False,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      # (algorithm_name, weight_tolerance, output_tolerance)
      algorithm_and_tolerances=[
          (_AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY, 1e-5, 1e-1),
          (_AlgorithmName.MIN_MAX_UNIFORM_QUANT, 1e-2, 1e-1),
          (_AlgorithmName.OCTAV, 1e-3, 1e-1),
          # Only check size reduction for MSE.
          (_AlgorithmName.MSE, 10, 10),
      ],
      # (model_name, granularity)
      model_and_granularity=[
          (
              'tensor_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.TENSORWISE,
          ),
          (
              'channel_i4rangedvalues_fc.tflite',
              qtyping.QuantGranularity.CHANNELWISE,
          ),
      ],
      weight_bit_width=[4, 8],
  )
  def test_a16static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_and_tolerances, model_and_granularity, weight_bit_width
  ):
    algorithm_key, weight_tolerance, output_tolerance = algorithm_and_tolerances
    model, granularity = model_and_granularity
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      algorithms=[
          _AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          _AlgorithmName.OCTAV,
      ],
      granularity=[
          qtyping.QuantGranularity.BLOCKWISE_32,
          qtyping.QuantGranularity.BLOCKWISE_64,
          qtyping.QuantGranularity.BLOCKWISE_128,
      ],
      activation_bit_width=[8, 16],
  )
  def test_int4_blockwise_static_quantization_raise_error(
      self, algorithms, granularity, activation_bit_width
  ):
    algorithm_key = algorithms
    model = 'tensor_i4rangedvalues_fc.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model
    op_config = _OpQuantConfig(
        activation_tensor_config=_TensorQuantConfig(
            num_bits=activation_bit_width,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )

    # Blockwise quantization is not supported for static quantization.
    error_msg = 'Unsupported op'
    with self.assertRaisesRegex(ValueError, error_msg):
      self.quantize_and_validate(
          model_path, algorithm_key, self._op_name, op_config
      )

  @parameterized.product(
      weight_bit_width=[4, 8],
      algorithm=[
          _AlgorithmName.HADAMARD_ROTATION,
          _AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ],
  )
  def test_hadamard_rotation_accuracy_and_size_within_tolerance(
      self, weight_bit_width, algorithm
  ):
    algorithm_key = algorithm
    # TODO(b/417309114): Soft skip weight errors because they're rotated hence
    # not expected to match.
    weight_tolerance = 1
    output_tolerance = 2e-2
    model_path = test_utils.get_path_to_datafile(
        '../models/conv_fc_mnist.tflite'
    )
    granularity = qtyping.QuantGranularity.CHANNELWISE
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )

  @parameterized.product(
      weight_bit_width=[4, 8],
  )
  def test_oscar_accuracy_and_size_within_tolerance(self, weight_bit_width):
    algorithm_key = _AlgorithmName.OSCAR
    # Weight errors are soft skipped because weights are scaled by per-channel
    # activation scales (W' = W * s) and not expected to match raw unscaled W.
    weight_tolerance = 10
    output_tolerance = 2e-2
    model_path = test_utils.get_path_to_datafile(
        '../models/conv_fc_mnist.tflite'
    )
    granularity = qtyping.QuantGranularity.CHANNELWISE
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=weight_bit_width,
            symmetric=True,
            granularity=granularity,
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    expected_model_size_reduction = (
        80 if op_config.weight_tensor_config.num_bits == 4 else 65
    )
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction,
        weight_tolerance,
        output_tolerance,
    )


if __name__ == '__main__':
  absltest.main()
