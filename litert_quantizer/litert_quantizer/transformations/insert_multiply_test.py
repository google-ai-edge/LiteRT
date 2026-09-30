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

"""Test insertion of the multiply transformation op."""

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import insert_multiply
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')


class InsertMultiplyFullyConnectedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/single_fc_bias.tflite'
    )
    self.model = tfl_flatbuffer_utils.read_model(model_path)
    self.multiplier = np.array([2.0, 0.5], dtype=np.float32)
    self.params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.ones(1),
        zero_point=np.zeros(1),
        custom_algorithm_param={'multiplier': self.multiplier},
    )

  def test_insert_multiply_raise_unsupported_qparams(self):
    with self.assertRaisesRegex(
        ValueError, 'uniform quantization'
    ):
      insert_multiply.insert_multiply(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[-1],
              quant_params=qtyping.NonLinearQuantParams(
                  num_bits=16, quantized_data=None
              ),
          )
      )

  def test_insert_multiply_raise_missing_multiplier(self):
    with self.assertRaisesRegex(
        ValueError, 'multiplier'
    ):
      insert_multiply.insert_multiply(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[-1],
              quant_params=qtyping.UniformQuantParams(
                  num_bits=8,
                  quantized_dimension=None,
                  scale=np.ones(1),
                  zero_point=np.zeros(1),
              ),
          )
      )

  def test_insert_multiply_raise_non_float32_tensor(self):
    self.model.subgraphs[0].tensors[0].type = qtyping.TensorType.INT32
    with self.assertRaisesRegex(
        ValueError, 'float32 tensors'
    ):
      insert_multiply.insert_multiply(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[-1],
              quant_params=self.params,
          )
      )

  def test_insert_multiply_op_updates_graph_correctly(self):
    subgraph = self.model.subgraphs[0]
    initial_tensor_count = len(subgraph.tensors)
    initial_op_count = len(subgraph.operators)
    original_input_tensor_id = 0

    info = insert_multiply.insert_multiply(
        transformation_utils.TransformationInput(
            tensor_id=original_input_tensor_id,
            model=self.model,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],  # Consumer is the FC op
            quant_params=self.params,
        )
    )

    with self.subTest('Info'):
      self.assertEqual(info.op_id, 0)
      self.assertEqual(info.num_ops_added, 1)

    with self.subTest('Tensor and op counts'):
      # Added 2 tensors: constant multiplier + activation output
      self.assertLen(subgraph.tensors, initial_tensor_count + 2)
      self.assertLen(subgraph.operators, initial_op_count + 1)

    # Op 0: MUL
    mul_op = subgraph.operators[0]
    with self.subTest('Inserted MUL op'):
      self.assertEqual(
          self.model.operatorCodes[mul_op.opcodeIndex].builtinCode,
          qtyping.BuiltinOperator.MUL,
      )

    with self.subTest('Inserted MUL op input tensor'):
      self.assertEqual(mul_op.inputs[0], original_input_tensor_id)

    # Multiplier constant tensor
    multiplier_tensor_id = mul_op.inputs[1]
    multiplier_tensor = subgraph.tensors[multiplier_tensor_id]
    mul_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        multiplier_tensor, self.model.buffers
    )
    with self.subTest('Multiplier constant tensor'):
      self.assertEqual(multiplier_tensor.type, qtyping.TensorType.FLOAT32)
      self.assertIsNotNone(mul_tensor_data)
      np.testing.assert_allclose(mul_tensor_data, self.multiplier)

    # MUL output tensor
    mul_output_tensor_id = mul_op.outputs[0]
    with self.subTest('Inserted MUL op output tensor'):
      self.assertEqual(info.output_tensor_id, mul_output_tensor_id)

    # Op 1: FULLY_CONNECTED (consumer updated)
    fc_op = subgraph.operators[1]
    with self.subTest('Updated fully connected op'):
      self.assertEqual(
          self.model.operatorCodes[fc_op.opcodeIndex].builtinCode,
          qtyping.BuiltinOperator.FULLY_CONNECTED,
      )
      self.assertEqual(fc_op.inputs[0], mul_output_tensor_id)

  def test_insert_multiply_shares_multiplier_constant_tensor(self):
    subgraph = self.model.subgraphs[0]
    original_input_tensor_id = 0
    info1 = insert_multiply.insert_multiply(
        transformation_utils.TransformationInput(
            tensor_id=original_input_tensor_id,
            model=self.model,
            subgraph=subgraph,
            producer=-1,
            consumers=[0],
            quant_params=self.params,
        )
    )
    multiplier_tensor_id_1 = subgraph.operators[info1.op_id].inputs[1]

    info2 = insert_multiply.insert_multiply(
        transformation_utils.TransformationInput(
            tensor_id=original_input_tensor_id,
            model=self.model,
            subgraph=subgraph,
            producer=-1,
            consumers=[1],
            quant_params=self.params,
        )
    )
    multiplier_tensor_id_2 = subgraph.operators[info2.op_id].inputs[1]

    self.assertEqual(multiplier_tensor_id_1, multiplier_tensor_id_2)


if __name__ == '__main__':
  absltest.main()
