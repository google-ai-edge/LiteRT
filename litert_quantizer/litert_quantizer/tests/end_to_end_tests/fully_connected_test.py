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

import logging
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


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
          # *_i4rangedvalues_fc.tflite weights are pre-quantized integers in
          # [-7, 6] multiplied by a scale, so naive min-max has zero weight
          # error whereas OCTAV's continuous tail clipping adds slight shrinkage
          # on top of 8-bit static activation/output quantization noise. See
          # test_int4_continuous_weights_octav_beats_min_max below for
          # continuous weights.
          (_AlgorithmName.OCTAV, 1e-3, 1.5e-1),
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

  def _create_continuous_weights_fc_model(
      self,
  ) -> tuple[str, str, np.ndarray]:
    base_model_path = (
        pathlib.Path(_TEST_MODEL_FOLDER) / 'channel_i4rangedvalues_fc.tflite'
    )
    model = tfl_flatbuffer_utils.read_model(base_model_path)
    subgraph = model.subgraphs[0]
    weight_tensors = [
        t
        for t in subgraph.tensors
        if list(t.shape) == [64, 128]
        and t.type == qtyping.TensorType.FLOAT32
        and model.buffers[t.buffer].data is not None
    ]
    self.assertLen(weight_tensors, 1)
    weight_tensor = weight_tensors[0]
    weight_tensor_name = tfl_flatbuffer_utils.get_tensor_name(weight_tensor)
    weight_shape = tuple(weight_tensor.shape)
    weight_buffer = model.buffers[weight_tensor.buffer]

    # The fixture's weights are pre-quantized INT4-range integers that min-max
    # reproduces losslessly; swap in continuous Laplace weights of the same
    # shape (in memory) so the OCTAV vs. MIN_MAX comparison is meaningful.
    rng = np.random.default_rng(1234)
    channel_scales = rng.uniform(0.5, 1.5, size=(weight_shape[0], 1)).astype(
        np.float32
    )
    continuous_weights = (
        rng.laplace(0.0, 0.08, size=weight_shape).astype(np.float32)
        * channel_scales
    )
    new_weight_bytes = (
        np.ascontiguousarray(continuous_weights, dtype=np.float32)
        .view(np.uint8)
        .ravel()
    )
    self.assertEqual(new_weight_bytes.nbytes, weight_buffer.data.nbytes)
    weight_buffer.data = new_weight_bytes

    continuous_model_path = (
        pathlib.Path(self.create_tempdir().full_path)
        / 'continuous_weights_fc.tflite'
    )
    tfl_flatbuffer_utils.write_model(model, continuous_model_path)
    return str(continuous_model_path), weight_tensor_name, continuous_weights

  def _get_weight_and_output_mse(
      self, comparison_result, weight_tensor_name: str
  ) -> tuple[float, float]:
    sig_keys = comparison_result.available_signature_keys()
    self.assertLen(sig_keys, 1)
    sig_result = comparison_result.get_signature_comparison_result(sig_keys[0])
    self.assertIn(weight_tensor_name, sig_result.constant_tensors)
    self.assertNotEmpty(sig_result.output_tensors)
    weight_mse = sig_result.constant_tensors[weight_tensor_name]['mse']
    output_mse = max(
        metrics['mse'] for metrics in sig_result.output_tensors.values()
    )
    return weight_mse, output_mse

  @parameterized.product(
      mode=['weight_only', 'dynamic', 'a8static', 'a16static'],
      granularity=[
          qtyping.QuantGranularity.TENSORWISE,
          qtyping.QuantGranularity.CHANNELWISE,
      ],
  )
  def test_int4_continuous_weights_octav_beats_min_max(
      self, mode: str, granularity: qtyping.QuantGranularity
  ):
    continuous_model_path, weight_tensor_name, _ = (
        self._create_continuous_weights_fc_model()
    )

    weight_config = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=granularity,
    )
    if mode == 'weight_only':
      op_config = _OpQuantConfig(
          weight_tensor_config=weight_config,
          compute_precision=_ComputePrecision.FLOAT,
          explicit_dequantize=True,
      )
    elif mode == 'dynamic':
      op_config = _OpQuantConfig(
          weight_tensor_config=weight_config,
          compute_precision=_ComputePrecision.INTEGER,
          explicit_dequantize=False,
      )
    elif mode == 'a8static':
      op_config = _OpQuantConfig(
          activation_tensor_config=_TensorQuantConfig(
              num_bits=8,
              symmetric=False,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          weight_tensor_config=weight_config,
          compute_precision=_ComputePrecision.INTEGER,
          explicit_dequantize=False,
      )
    else:
      self.assertEqual(mode, 'a16static')
      op_config = _OpQuantConfig(
          activation_tensor_config=_TensorQuantConfig(
              num_bits=16,
              symmetric=True,
              granularity=qtyping.QuantGranularity.TENSORWISE,
          ),
          weight_tensor_config=weight_config,
          compute_precision=_ComputePrecision.INTEGER,
          explicit_dequantize=False,
      )

    octav_result = self.quantize_and_validate(
        model_path=continuous_model_path,
        algorithm_key=_AlgorithmName.OCTAV,
        op_name=self._op_name,
        op_config=op_config,
        num_validation_samples=16,
    )
    min_max_result = self.quantize_and_validate(
        model_path=continuous_model_path,
        algorithm_key=_AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_name=self._op_name,
        op_config=op_config,
        num_validation_samples=16,
    )

    self.assert_model_size_reduction_above_min_pct(octav_result, 80)
    self.assert_model_size_reduction_above_min_pct(min_max_result, 80)

    octav_weight_mse, octav_output_mse = self._get_weight_and_output_mse(
        octav_result, weight_tensor_name
    )
    min_max_weight_mse, min_max_output_mse = self._get_weight_and_output_mse(
        min_max_result, weight_tensor_name
    )

    logging.info(
        'Continuous FC (%s, %s) MSE: OCTAV=(weight=%.6e, output=%.6e),'
        ' MIN_MAX=(weight=%.6e, output=%.6e)',
        mode,
        granularity.name,
        octav_weight_mse,
        octav_output_mse,
        min_max_weight_mse,
        min_max_output_mse,
    )
    # Tighter weight and output MSE tolerances on continuous weights.
    self.assertLess(octav_weight_mse, 6e-4)
    self.assertLess(octav_output_mse, 8e-2)
    # Strictly lower weight and output MSE than naive min-max quantization.
    self.assertLess(
        octav_weight_mse,
        min_max_weight_mse,
        f'Expected OCTAV weight MSE ({octav_weight_mse:.6e}) < MIN_MAX'
        f' ({min_max_weight_mse:.6e}) for ({mode}, {granularity.name}).',
    )
    self.assertLess(
        octav_output_mse,
        min_max_output_mse,
        f'Expected OCTAV output MSE ({octav_output_mse:.6e}) < MIN_MAX'
        f' ({min_max_output_mse:.6e}) for ({mode}, {granularity.name}).',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='tensorwise',
          granularity=qtyping.QuantGranularity.TENSORWISE,
          expected_num_scales=1,
          expected_scale_prefix=[0.06625064],
      ),
      dict(
          testcase_name='channelwise',
          granularity=qtyping.QuantGranularity.CHANNELWISE,
          expected_num_scales=64,
          expected_scale_prefix=[
              0.08209092,
              0.06690791,
              0.0791962,
              0.0373883,
          ],
      ),
  )
  def test_int2_in_int4_end_to_end(
      self,
      granularity: qtyping.QuantGranularity,
      expected_num_scales: int,
      expected_scale_prefix: list[float],
  ):
    continuous_model_path, weight_tensor_name, continuous_weights = (
        self._create_continuous_weights_fc_model()
    )
    op_config = _OpQuantConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=4,
            symmetric=True,
            granularity=granularity,
            algorithm_params={'int2_in_int4': True},
        ),
        compute_precision=_ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )

    qt = quantizer.Quantizer(continuous_model_path)
    qt.update_quantization_recipe(
        algorithm_key=_AlgorithmName.OCTAV,
        regex='.*',
        operation_name=self._op_name,
        op_config=op_config,
    )
    quantization_result = qt.quantize()
    self.assertIsNotNone(quantization_result.quantized_model)

    quantized_model = tfl_flatbuffer_utils.read_model(
        quantization_result.quantized_model
    )
    q_weight_tensors = [
        t
        for t in quantized_model.subgraphs[0].tensors
        if tfl_flatbuffer_utils.get_tensor_name(t) == weight_tensor_name
    ]
    self.assertLen(q_weight_tensors, 1)
    q_weight_tensor = q_weight_tensors[0]
    self.assertEqual(q_weight_tensor.type, qtyping.TensorType.INT4)

    packed = np.frombuffer(
        quantized_model.buffers[q_weight_tensor.buffer].data, dtype=np.uint8
    )
    unpacked = np.empty(packed.size * 2, dtype=np.int8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked[0::2] = ((low ^ 0x8) - 0x8).astype(np.int8)
    unpacked[1::2] = ((high ^ 0x8) - 0x8).astype(np.int8)
    self.assertEqual(unpacked.size, continuous_weights.size)
    self.assertSetEqual(set(np.unique(unpacked).tolist()), {-3, -1, 1, 3})

    np.testing.assert_array_equal(q_weight_tensor.quantization.zeroPoint, 0)
    stored_scale = np.asarray(
        q_weight_tensor.quantization.scale, dtype=np.float32
    )
    self.assertLen(stored_scale, expected_num_scales)
    np.testing.assert_allclose(
        stored_scale[: len(expected_scale_prefix)],
        expected_scale_prefix,
        rtol=1e-6,
        atol=1e-6,
    )

    validation_result = qt.validate()
    self.assert_model_size_reduction_above_min_pct(validation_result, 80)
    weight_mse, output_mse = self._get_weight_and_output_mse(
        validation_result, weight_tensor_name
    )
    self.assertLess(weight_mse, 4e-3)
    self.assertLess(output_mse, 0.5)

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
