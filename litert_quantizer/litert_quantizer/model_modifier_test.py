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

"""Tests for the ModelModifier class."""

import gc
import pathlib
import tempfile
import tracemalloc

from absl.testing import absltest
from absl.testing import parameterized
from flatbuffers import flexbuffers

import os
import io
from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_litert.tools import mmap_utils
from litert_quantizer import model_modifier
from litert_quantizer import params_generator
from litert_quantizer import qtyping
from litert_quantizer import recipe_manager
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('.')


class BaseModelModifierTest(parameterized.TestCase):
  """Base test class that pre-loads a model and creates a modifier for it."""

  _model_path: str = str(
      pathlib.Path(TEST_DATA_PREFIX_PATH) / 'tests/models/conv_fc_mnist.tflite'
  )

  def setUp(self):
    super().setUp()
    self._model_content: bytes = tfl_flatbuffer_utils.get_model_content(  # pyrefly: ignore[bad-assignment]
        self._model_path
    )
    self._model = tfl_flatbuffer_utils.read_model(self._model_content)
    self._model_modifier = model_modifier.ModelModifier(self._model)


class ModelModifierTestSmallModel(BaseModelModifierTest):
  _packed_buffer_data_size: int = 201984
  _global_recipe: qtyping.ModelQuantizationRecipe = [
      {
          'regex': '.*',
          'operation': 'FULLY_CONNECTED',
          'algorithm_key': 'min_max_uniform_quantize',
          'op_config': {
              'weight_tensor_config': {
                  'dtype': qtyping.TensorDataType.INT,
                  'num_bits': 8,
                  'symmetric': False,
                  'granularity': qtyping.QuantGranularity.CHANNELWISE,
                  'block_size': 0,
              },
              # Equivalent to WEIGHT_ONLY.
              'compute_precision': qtyping.ComputePrecision.FLOAT,
              'explicit_dequantize': True,
          },
      },
  ]

  def test_pack_buffer_data_succeeds(self):
    packed_buffer_data = model_modifier._PackedBufferData(self._model)
    self.assertEqual(
        packed_buffer_data.packed_size, self._packed_buffer_data_size
    )

  def test_modify_model_succeeds_with_recipe(self):
    recipe_manager_instance = recipe_manager.RecipeManager()
    params_generator_instance = params_generator.ParamsGenerator(self._model)

    recipe_manager_instance.load_quantization_recipe(self._global_recipe)
    tensor_quantization_params = (
        params_generator_instance.generate_quantization_parameters(
            recipe_manager_instance
        )
    )
    new_model_binary = self._model_modifier.modify_model(
        tensor_quantization_params
    )
    self.assertIsInstance(
        flatbuffer_utils.convert_bytearray_to_object(new_model_binary),
        qtyping.ModelT,
    )
    self.assertLess(len(new_model_binary), len(self._model_content))

  def test_modify_model_serialize_to_path_succeeds(self):
    recipe_manager_instance = recipe_manager.RecipeManager()
    params_generator_instance = params_generator.ParamsGenerator(self._model)

    recipe_manager_instance.load_quantization_recipe(self._global_recipe)
    tensor_quantization_params = (
        params_generator_instance.generate_quantization_parameters(
            recipe_manager_instance
        )
    )
    path = tempfile.mktemp()
    serialized_model = self._model_modifier.modify_model(
        tensor_quantization_params,
        serialize_to_path=path,
    )
    self.assertTrue(os.path.exists(path))
    self.assertEqual(serialized_model, mmap_utils.get_file_contents(path))

  def test_modify_model_preserves_original_model(self):
    recipe_manager_instance = recipe_manager.RecipeManager()
    params_generator_instance = params_generator.ParamsGenerator(self._model)

    recipe_manager_instance.load_quantization_recipe(self._global_recipe)
    tensor_quantization_params = (
        params_generator_instance.generate_quantization_parameters(
            recipe_manager_instance
        )
    )
    model_buffer = flatbuffer_utils.convert_object_to_bytearray(
        self._model_modifier._model
    )
    self._model_modifier.modify_model(tensor_quantization_params)
    self.assertEqual(
        flatbuffer_utils.convert_object_to_bytearray(
            self._model_modifier._model
        ),
        model_buffer,
    )

  def test_modify_model_peak_memory_usage_in_acceptable_range(self):
    """Test ModifyModel peak memory usage."""

    recipe_manager_instance = recipe_manager.RecipeManager()
    params_generator_instance = params_generator.ParamsGenerator(self._model)

    recipe_manager_instance.load_quantization_recipe(self._global_recipe)
    tensor_quantization_params = (
        params_generator_instance.generate_quantization_parameters(
            recipe_manager_instance
        )
    )

    # Run once and garbage-collect to make sure we're really only measuring
    # what's allocated during the operation itself (e.g. avoid module import
    # stuff).
    res = self._model_modifier.modify_model(tensor_quantization_params)
    del res
    gc.collect()

    # Trace the peak memory usage of calling `ModelModifier.modify_model`.
    tracemalloc.start()
    tracemalloc.reset_peak()
    quantized_model = self._model_modifier.modify_model(
        tensor_quantization_params
    )
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    self.assertLessEqual(
        mem_peak, len(self._model_content) + len(quantized_model)
    )

  def test_has_transform_before_output_true_dequant(self):
    instructions = {
        'tensor1': qtyping.TensorTransformationInsts(
            'tensor1',
            0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=0,
                    producer=0,
                    consumers=[-1],
                )
            ],
        )
    }
    self.assertTrue(
        self._model_modifier._has_transform_before_output(
            instructions, qtyping.QuantTransformation.ADD_DEQUANTIZE
        )
    )

  def test_has_transform_before_output_false_dequant(self):
    instructions = {
        'tensor1': qtyping.TensorTransformationInsts(
            'tensor1',
            0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_DEQUANTIZE,
                    tensor_id=0,
                    producer=0,
                    consumers=[1],
                )
            ],
        )
    }
    self.assertFalse(
        self._model_modifier._has_transform_before_output(
            instructions, qtyping.QuantTransformation.ADD_DEQUANTIZE
        )
    )

  def test_has_transform_before_output_true_quant(self):
    instructions = {
        'tensor1': qtyping.TensorTransformationInsts(
            'tensor1',
            0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                    tensor_id=0,
                    producer=0,
                    consumers=[-1],
                )
            ],
        )
    }
    self.assertTrue(
        self._model_modifier._has_transform_before_output(
            instructions, qtyping.QuantTransformation.ADD_QUANTIZE
        )
    )

  def test_has_transform_before_output_false_quant(self):
    instructions = {
        'tensor1': qtyping.TensorTransformationInsts(
            'tensor1',
            0,
            instructions=[
                qtyping.TransformationInst(
                    transformation=qtyping.QuantTransformation.ADD_QUANTIZE,
                    tensor_id=0,
                    producer=0,
                    consumers=[1],
                )
            ],
        )
    }
    self.assertFalse(
        self._model_modifier._has_transform_before_output(
            instructions, qtyping.QuantTransformation.ADD_QUANTIZE
        )
    )

  def test_pad_offset(self):
    arr_len = 3
    self.assertEqual(model_modifier._round_up_16(arr_len), 16)

    arr_len = 16
    self.assertEqual(model_modifier._round_up_16(arr_len), 16)

    arr_len = 17
    self.assertEqual(model_modifier._round_up_16(arr_len), 32)


class ModelModifierTestLargeModel(ModelModifierTestSmallModel):

  _model_path = str(
      pathlib.Path(TEST_DATA_PREFIX_PATH)
      / 'tests/models/toy_model_with_kv_cache_multi_signature.tflite'
  )
  _packed_buffer_data_size: int = 745472
  _global_recipe: qtyping.ModelQuantizationRecipe = [
      {
          'regex': '.*',
          'operation': '*',
          'algorithm_key': 'min_max_uniform_quantize',
          'op_config': {
              'weight_tensor_config': {
                  'dtype': qtyping.TensorDataType.INT,
                  'num_bits': 8,
                  'symmetric': False,
                  'granularity': qtyping.QuantGranularity.CHANNELWISE,
                  'block_size': 0,
              },
              # Equivalent to WEIGHT_ONLY.
              'compute_precision': qtyping.ComputePrecision.FLOAT,
              'explicit_dequantize': True,
          },
      },
  ]


class ModelModifierTestWithSignature(BaseModelModifierTest):

  _model_path = str(
      pathlib.Path(TEST_DATA_PREFIX_PATH) / 'tests/models/single_fc.tflite',
  )

  def test_update_signature_defs_succeeds_dequant(self):
    # This is a simplified test that only checks if the function runs without
    # crashing and returns a model. A more thorough test with a model
    # with a known signature was added in `quantizer_test`.
    model = flatbuffer_utils.read_model_from_bytearray(self._model_content)
    updated_model = self._model_modifier._update_signature_defs(
        model, '_dequant'
    )
    self.assertIsNotNone(updated_model)

  def test_update_signature_defs_succeeds_quant(self):
    # This checks if the function runs without crashing and returns a model.
    model = flatbuffer_utils.read_model_from_bytearray(self._model_content)
    updated_model = self._model_modifier._update_signature_defs(
        model, '_quantized'
    )
    self.assertIsNotNone(updated_model)


class ModelModifierTestMoe(BaseModelModifierTest):

  _model_path = str(
      pathlib.Path(TEST_DATA_PREFIX_PATH)
      / 'tests/models/single_moe_multiaxis.tflite'
  )
  _MOE_SUBGRAPH_INDEX = 0
  _MOE_OP_INDEX = 0

  def _set_blockwise_quantization(
      self,
      tensor_idx: int,
      scale_tensor_idx: int,
      tensor_type: qtyping.TensorType,
  ) -> None:
    """Configures a tensor with BlockwiseQuantization scale details.

    Args:
      tensor_idx: The index of the tensor to configure.
      scale_tensor_idx: The index of the scale tensor.
      tensor_type: The quantized type of the tensor.
    """
    tensor = self._model.subgraphs[self._MOE_SUBGRAPH_INDEX].tensors[tensor_idx]
    tensor.type = tensor_type
    tensor.quantization = qtyping.QuantizationParametersT()
    tensor.quantization.detailsType = (
        qtyping.QuantizationDetails.BlockwiseQuantization
    )
    blockwise_params = qtyping.BlockwiseQuantizationT()
    blockwise_params.scales = scale_tensor_idx
    # Placeholder; `_update_quantized_moe_ops` only checks that it's non-empty.
    blockwise_params.blockShape = [1]
    tensor.quantization.details = blockwise_params

  @parameterized.named_parameters(
      dict(
          testcase_name='int8',
          tensor_type=qtyping.TensorType.INT8,
          expected_weight_type='int8',
      ),
      dict(
          testcase_name='int4',
          tensor_type=qtyping.TensorType.INT4,
          expected_weight_type='int4',
      ),
  )
  def test_update_quantized_moe_ops(self, tensor_type, expected_weight_type):
    subgraph = self._model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    op = subgraph.operators[self._MOE_OP_INDEX]
    # Mark weights with BlockwiseQuantization scale details.
    (
        src_idx,
        top_weights_idx,
        top_indices_idx,
        gate_weight_idx,
        ff1_weight_idx,
        linear_weight_idx,
        per_expert_scale_idx,
    ) = op.inputs
    gate_scale_idx, ff1_scale_idx, linear_scale_idx = (100, 101, 102)
    self._set_blockwise_quantization(
        gate_weight_idx, gate_scale_idx, tensor_type
    )
    self._set_blockwise_quantization(ff1_weight_idx, ff1_scale_idx, tensor_type)
    self._set_blockwise_quantization(
        linear_weight_idx, linear_scale_idx, tensor_type
    )

    updated_model = self._model_modifier._update_quantized_moe_ops(self._model)
    updated_subgraph = updated_model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    updated_op = updated_subgraph.operators[self._MOE_OP_INDEX]

    with self.subTest('weight_type'):
      custom_options = flexbuffers.Loads(bytes(updated_op.customOptions))
      self.assertEqual(custom_options['weight_type'], expected_weight_type)

    with self.subTest('inputs'):
      self.assertEqual(
          updated_op.inputs,
          [
              src_idx,
              top_weights_idx,
              top_indices_idx,
              gate_weight_idx,
              gate_scale_idx,
              ff1_weight_idx,
              ff1_scale_idx,
              linear_weight_idx,
              linear_scale_idx,
              per_expert_scale_idx,
          ],
      )

  def test_update_quantized_moe_ops_skips_float_weights(self):
    updated_model = self._model_modifier._update_quantized_moe_ops(self._model)
    updated_subgraph = updated_model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    updated_op = updated_subgraph.operators[self._MOE_OP_INDEX]

    with self.subTest('weight_type'):
      custom_options = flexbuffers.Loads(bytes(updated_op.customOptions))
      self.assertEqual(custom_options['weight_type'], 'fp32')

    with self.subTest('inputs'):
      self.assertLen(updated_op.inputs, 7)

  def test_update_quantized_moe_ops_unsupported_weight_type_raises_error(self):
    subgraph = self._model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    gate_weight_idx = subgraph.operators[self._MOE_OP_INDEX].inputs[3]
    subgraph.tensors[gate_weight_idx].type = qtyping.TensorType.INT2

    with self.assertRaisesRegex(
        ValueError, r'Quantized MoE weights must be INT8 or INT4, got INT2\.'
    ):
      self._model_modifier._update_quantized_moe_ops(self._model)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_details',
          quant_params=qtyping.QuantizationParametersT(),
      ),
      dict(
          testcase_name='empty_block_shape',
          quant_params=qtyping.QuantizationParametersT(
              detailsType=qtyping.QuantizationDetails.BlockwiseQuantization,
              details=qtyping.BlockwiseQuantizationT(blockShape=[]),
          ),
      ),
  )
  def test_update_quantized_moe_ops_missing_blockwise_details_raises_error(
      self, quant_params
  ):
    subgraph = self._model.subgraphs[self._MOE_SUBGRAPH_INDEX]
    gate_weight_idx = subgraph.operators[self._MOE_OP_INDEX].inputs[3]
    gate_weight = subgraph.tensors[gate_weight_idx]
    gate_weight.type = qtyping.TensorType.INT8
    gate_weight.quantization = quant_params

    with self.assertRaisesRegex(
        ValueError,
        r'Weight tensor arith\.constant is missing BlockwiseQuantization'
        r' details with a non-empty block_shape\.',
    ):
      self._model_modifier._update_quantized_moe_ops(self._model)


if __name__ == '__main__':
  absltest.main()
