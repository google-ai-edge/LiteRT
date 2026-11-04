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

"""E2E tests for the quantizer for model with embedding_lookup."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils

_ComputePrecision = qtyping.ComputePrecision
_OpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_OpQuantConfig = qtyping.OpQuantizationConfig

_RNG = np.random.default_rng(66)


def _get_dummy_data(num_samples):
  data = []
  for _ in range(num_samples):
    data.append({'inputs': _RNG.uniform(size=(1,)).astype(np.int32)})
  return data


# def _get_test_data(num_samples: int = 8):
#   return _get_dummy_data(num_samples)


class EmbeddingLookupTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self.float_model_path = test_utils.get_path_to_datafile(
        '../models/embedding_lookup.tflite'
    )
    self._quantizer = quantizer.Quantizer(self.float_model_path)

  @parameterized.parameters(
      ('../../recipes/default_af32w8float_recipe.json', 1700),
      ('../../recipes/default_af32w4float_recipe.json', 1600),
      ('../../recipes/dynamic_legacy_wi8_afp32_recipe.json', 1400),
      ('../../recipes/dynamic_wi8_afp32_recipe.json', 1400),
  )
  def test_embedding_lookup_model_int_weight_only(
      self, recipe_path, expected_model_size
  ):
    recipe_path = test_utils.get_path_to_datafile(recipe_path)
    self._quantizer.load_quantization_recipe(recipe_path)
    self.assertFalse(self._quantizer.need_calibration)
    quant_result = self._quantizer.quantize()
    # Check model size.
    self.assertLess(len(quant_result.quantized_model), expected_model_size)

  @parameterized.product(
      weight_bit_width=[4, 8],
      quantization_method=['add_weight_only_config', 'add_dynamic_config'],
  )
  def test_embedding_lookup_model_mse_quantization(
      self, weight_bit_width, quantization_method
  ):
    config_setter = getattr(self._quantizer, quantization_method)
    config_setter(
        regex='.*',
        operation_name=_OpName.EMBEDDING_LOOKUP,
        num_bits=weight_bit_width,
        algorithm_key=quantizer.AlgorithmName.MSE,
    )
    self.assertFalse(self._quantizer.need_calibration)
    quant_result = self._quantizer.quantize()
    # Only check model size for now.
    self.assertLess(len(quant_result.quantized_model), 1700)

  def test_embedding_lookup_model_fp16_weight_only(self):
    self._quantizer.update_quantization_recipe(
        regex='.*',
        algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING,
        operation_name=_OpName.EMBEDDING_LOOKUP,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT
            ),
            compute_precision=_ComputePrecision.FLOAT,  # WEIGHT_ONLY
            explicit_dequantize=True,
        ),
    )
    quant_result = self._quantizer.quantize()
    self.assertLess(len(quant_result.quantized_model), 2000)

    # TODO: b/364405203 - Enable after 0 signature works.
    # comparison_result = self._quantizer.validate(
    #     error_metrics=[quantizer.ValidationErrorMetric.MSE],
    #     signature_test_data=_get_test_data(),
    # )
    # self._check_comparison_result(
    #     comparison_result,
    #     weight_tolerance=1e-5,
    #     output_tolerance=1e-5,
    # )

  @parameterized.product(weight_bit_width=[4, 8])
  def test_hadamard_rotation_accuracy_and_size_within_tolerance(
      self, weight_bit_width
  ):
    self._quantizer.load_quantization_recipe([
        {
            'regex': '.*',
            'operation': qtyping.TFLOperationName.EMBEDDING_LOOKUP,
            'algorithm_key': quantizer.AlgorithmName.HADAMARD_ROTATION,
            'op_config': {
                'weight_tensor_config': {
                    'dtype': qtyping.TensorDataType.INT,
                    'num_bits': weight_bit_width,
                    'granularity': qtyping.QuantGranularity.CHANNELWISE,
                },
                'compute_precision': qtyping.ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ])
    quant_result = self._quantizer.quantize()
    with self.subTest(name='ModelSizeReduction'):
      self.assertLess(len(quant_result.quantized_model), 1650)

    validation_result = self._quantizer.validate(
        test_data={
            'lookup': [
                {'lookup': np.random.randint(0, 15, size=(1,), dtype=np.int32)}
            ],
        },
        error_metrics=[quantizer.ValidationErrorMetric.MSE],
    )
    with self.subTest(name='OutputErrors'):
      self.assert_output_errors_below_tolerance(validation_result, 1e-5)

  @parameterized.named_parameters(
      # `D = 8 = 2 * 4`: `Q` is the symmetric Sylvester `H_2`.
      dict(
          testcase_name='sylvester_q',
          table_width=8,
          sizing_mode='min_compute',
          randomize=False,
          expected_num_blocks=2,
      ),
      # `D = 24 = 6 * 4`: `Q` is an asymmetric Cayley rotation, so this checks
      # that the lookup output is un-rotated with `Q.T`.
      dict(
          testcase_name='cayley_q',
          table_width=24,
          sizing_mode='min_compute',
          randomize=False,
          expected_num_blocks=6,
      ),
      # `D = 24 = 3 * 8`.
      dict(
          testcase_name='cayley_q_max_power_of_two',
          table_width=24,
          sizing_mode='max_power_of_two',
          randomize=False,
          expected_num_blocks=3,
      ),
      dict(
          testcase_name='cayley_q_randomized',
          table_width=24,
          sizing_mode='min_compute',
          randomize=True,
          expected_num_blocks=6,
      ),
  )
  def test_decomposed_kronecker_rotation_accuracy(
      self, table_width, sizing_mode, randomize, expected_num_blocks
  ):
    table = _RNG.uniform(-1, 1, size=(16, table_width)).astype(np.float32)
    qt = quantizer.Quantizer(self._create_embedding_model(table))
    qt.load_quantization_recipe([
        {
            'regex': '.*',
            'operation': qtyping.TFLOperationName.EMBEDDING_LOOKUP,
            'algorithm_key': (
                quantizer.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION
            ),
            'op_config': {
                'weight_tensor_config': {
                    'dtype': qtyping.TensorDataType.INT,
                    'num_bits': 8,
                    'granularity': qtyping.QuantGranularity.CHANNELWISE,
                    'algorithm_params': {
                        'q_init_mode': 'auto',
                        'sizing_mode': sizing_mode,
                        'randomize': randomize,
                    },
                },
                'compute_precision': qtyping.ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ])
    quantized_model = qt.quantize().quantized_model
    self.assertEqual(
        [
            list(tensor.shape)
            for tensor in (
                tfl_flatbuffer_utils.read_model(quantized_model)
                .subgraphs[0]
                .tensors
            )
            if tfl_flatbuffer_utils.get_tensor_name(tensor).endswith(
                '_cross_block_matrix'
            )
        ],
        [[expected_num_blocks, expected_num_blocks]],
    )

    # The signature of `embedding_lookup.tflite` has no outputs, so
    # `validate()` would not compare the lookup results. Look up every row of
    # the quantized model instead.
    interpreter = tfl_interpreter_utils.create_tfl_interpreter(quantized_model)
    lookup_results = []
    for row in range(len(table)):
      tfl_interpreter_utils.invoke_interpreter_once(
          interpreter, [np.array([row], dtype=np.int32)]
      )
      lookup_results.append(
          interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
      )
    self.assertLess(
        np.mean((np.concatenate(lookup_results) - table) ** 2), 1e-4
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='block_hadamard',
          q_init_mode='none',
          expected_rotation_matrix_types={
              '_hadamard_matrix': qtyping.TensorType.INT8,
          },
      ),
      dict(
          testcase_name='kronecker',
          q_init_mode='auto',
          expected_rotation_matrix_types={
              '_hadamard_matrix': qtyping.TensorType.INT8,
              '_cross_block_matrix': qtyping.TensorType.INT8,
          },
      ),
  )
  def test_second_pass_quantizes_inserted_rotation_matrices(
      self, q_init_mode, expected_rotation_matrix_types
  ):
    # Pass 1 rotates and quantizes the table, and inserts rotation ops with
    # float32 weights.
    self._quantizer.load_quantization_recipe([
        {
            'regex': '.*',
            'operation': qtyping.TFLOperationName.EMBEDDING_LOOKUP,
            'algorithm_key': (
                quantizer.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION
            ),
            'op_config': {
                'weight_tensor_config': {
                    'dtype': qtyping.TensorDataType.INT,
                    'num_bits': 8,
                    'granularity': qtyping.QuantGranularity.CHANNELWISE,
                    'algorithm_params': {'q_init_mode': q_init_mode},
                },
                'compute_precision': qtyping.ComputePrecision.FLOAT,
                'explicit_dequantize': True,
            },
        },
    ])
    second_pass = quantizer.Quantizer(
        self._quantizer.quantize().quantized_model
    )
    # Pass 2 selects the inserted rotation ops with the regex documented in
    # `hadamard_rotation.md`.
    second_pass.add_dynamic_config(
        regex='_(?:hadamard_|cross_)?rotated;',
        operation_name=_OpName.FULLY_CONNECTED,
        num_bits=8,
    )
    quantized_model = tfl_flatbuffer_utils.read_model(
        second_pass.quantize().quantized_model
    )

    rotation_matrix_types = {}
    for tensor in quantized_model.subgraphs[0].tensors:
      name = tfl_flatbuffer_utils.get_tensor_name(tensor)
      for suffix in ('_hadamard_matrix', '_cross_block_matrix'):
        if name.endswith(suffix):
          rotation_matrix_types[suffix] = tensor.type
    self.assertEqual(rotation_matrix_types, expected_rotation_matrix_types)

  def _create_embedding_model(self, table: np.ndarray) -> str:
    """Writes `embedding_lookup.tflite` with its `[16, 8]` table replaced.

    Only the table and output shapes change, so the model is still a single
    lookup. Wider tables give Kronecker factorizations `D = m * d` with a
    non-power-of-2 `m`.

    Args:
      table: The new `[16, D]` float32 embedding table.

    Returns:
      The path to the written model.
    """
    model = tfl_flatbuffer_utils.read_model(self.float_model_path)
    (embedding_op,) = model.subgraphs[0].operators
    table_tensor = model.subgraphs[0].tensors[embedding_op.inputs[1]]
    output_tensor = model.subgraphs[0].tensors[embedding_op.outputs[0]]
    table_tensor.shape = list(table.shape)
    output_tensor.shape = [1, table.shape[1]]
    output_tensor.shapeSignature = None
    model.buffers[table_tensor.buffer].data = np.frombuffer(
        table.tobytes(), dtype=np.uint8
    )
    model_path = self.create_tempfile().full_path
    tfl_flatbuffer_utils.write_model(model, model_path)
    return model_path


if __name__ == '__main__':
  absltest.main()
