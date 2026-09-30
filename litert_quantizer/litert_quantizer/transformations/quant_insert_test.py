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

"""Test for various transformations used by quantization toolkit."""

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import quant_insert
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("..")


class QuantInsertTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/insert_dequant_test.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._orig_test_model_path)

  def test_quant_insert_constant(self):
    """Test quant insert lib on a constant tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quant_opcode = qtyping.BuiltinOperator.QUANTIZE
    # insert quant on the constant before the add node
    quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=7,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )

    # check quant op code is added to the model
    self.assertEqual(
        model.operatorCodes[0].builtinCode,
        quant_opcode,
    )

    # check new tensor is correct created
    self.assertIn(b"_quantized", subgraph.tensors[9].name)
    self.assertEqual(subgraph.tensors[9].type, qtyping.TensorType.INT8)
    self.assertEqual(subgraph.tensors[7].type, qtyping.TensorType.UINT8)
    # checking if consumer has the correct input
    self.assertEqual(subgraph.operators[5].inputs[0], 6)
    self.assertEqual(subgraph.operators[5].inputs[1], 9)

    # checking the inserted node has the correct input/output
    self.assertEqual(subgraph.operators[4].outputs[0], 9)
    self.assertEqual(subgraph.operators[4].inputs[0], 7)
    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[4].opcodeIndex, 0)

  def test_quant_insert_activation(self):
    """Test quant insert lib on activation tensors."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quant_opcode = qtyping.BuiltinOperator.QUANTIZE
    # insert quant on the output of a conv node
    quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=4,
            model=model,
            subgraph=subgraph,
            producer=1,
            consumers=[3],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )

    # check quant op code is added to the model
    self.assertEqual(
        model.operatorCodes[0].builtinCode,
        quant_opcode,
    )

    # check new tensor is correctly created
    self.assertIn(b"_quantized", subgraph.tensors[9].name)
    self.assertEqual(subgraph.tensors[9].type, qtyping.TensorType.INT8)
    # check original source tensor is updated
    self.assertEqual(subgraph.tensors[4].type, qtyping.TensorType.UINT8)

    # checking if consumer haves the correct input
    self.assertEqual(subgraph.operators[4].inputs[0], 9)
    self.assertEqual(subgraph.operators[4].inputs[1], 5)

    # checking the inserted node has the correct input/output
    self.assertEqual(subgraph.operators[3].outputs[0], 9)
    self.assertEqual(subgraph.operators[3].inputs[0], 4)
    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[3].opcodeIndex, 0)

  def test_quant_insert_constant_multiple_consumers(self):
    """Test quant insert lib on tensors with multiple consumers."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quant_opcode = qtyping.BuiltinOperator.QUANTIZE
    # insert quant on the input of a conv node
    post_trans_info = quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=2,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[1, 2],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )
    self.assertEqual(post_trans_info.op_id, 1)
    self.assertEqual(post_trans_info.num_ops_added, 1)

    # check quant op code is added to the model
    self.assertEqual(
        model.operatorCodes[0].builtinCode,
        quant_opcode,
    )

    # check new tensor is correct created
    self.assertIn(b"_quantized", subgraph.tensors[9].name)
    self.assertEqual(subgraph.tensors[9].type, qtyping.TensorType.INT8)
    # check original source tensor has the correct type
    self.assertEqual(subgraph.tensors[2].type, qtyping.TensorType.UINT8)

    # checking the inserted node has the correct input/output
    self.assertEqual(subgraph.operators[1].outputs[0], 9)
    self.assertEqual(subgraph.operators[1].inputs[0], 2)
    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[1].opcodeIndex, 0)

    # checking if consumer haves the correct input
    self.assertEqual(subgraph.operators[2].inputs[1], 9)
    self.assertEqual(subgraph.operators[3].inputs[1], 9)

  def test_quant_insert_activation_multiple_consumers(self):
    """Test quant insert lib on tensors with multiple consumers."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quant_opcode = qtyping.BuiltinOperator.QUANTIZE
    # insert quant on the output of a conv node
    quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=1,
            model=model,
            subgraph=subgraph,
            producer=0,
            consumers=[1, 2],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )

    # check quant op code is added to the model
    self.assertEqual(
        model.operatorCodes[0].builtinCode,
        quant_opcode,
    )

    # check new tensor is correct created
    self.assertIn(b"_quantized", subgraph.tensors[9].name)
    self.assertEqual(subgraph.tensors[9].type, qtyping.TensorType.INT8)
    # check original source tensor is updated
    self.assertEqual(subgraph.tensors[1].type, qtyping.TensorType.UINT8)

    # checking the inserted node has the correct input/output
    self.assertEqual(subgraph.operators[1].outputs[0], 9)
    self.assertEqual(subgraph.operators[1].inputs[0], 1)
    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[1].opcodeIndex, 0)

    # checking if consumer haves the correct input
    self.assertEqual(subgraph.operators[2].inputs[0], 9)
    self.assertEqual(subgraph.operators[3].inputs[0], 9)

  def test_quant_insert_activation_multiple_consumers_select(self):
    """Test quant insert lib on tensors with multiple consumers but only insert for one of them."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quant_opcode = qtyping.BuiltinOperator.QUANTIZE
    # insert quant on the output of a conv node
    quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=1,
            model=model,
            subgraph=subgraph,
            producer=0,
            consumers=[1],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )

    # check quant op code is added to the model
    self.assertEqual(
        model.operatorCodes[0].builtinCode,
        quant_opcode,
    )

    # check new tensor is correct created
    self.assertIn(b"_quantized", subgraph.tensors[9].name)
    self.assertEqual(subgraph.tensors[9].type, qtyping.TensorType.INT8)
    # check original source tensor is updated
    self.assertEqual(subgraph.tensors[1].type, qtyping.TensorType.UINT8)

    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[1].opcodeIndex, 0)

    # checking if consumer haves the correct input
    self.assertEqual(subgraph.operators[2].inputs[0], 9)
    self.assertEqual(subgraph.operators[3].inputs[0], 1)

    # checking the inserted node has the correct input/output
    self.assertEqual(subgraph.operators[1].outputs[0], 9)
    self.assertEqual(subgraph.operators[1].inputs[0], 1)

  def test_dequant_insert_on_graph_output(self):
    """Test dequant insert lib on graph output."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    # insert dequant on the graph output
    quant_insert.insert_quant(
        transformation_utils.TransformationInput(
            tensor_id=8,
            model=model,
            subgraph=subgraph,
            producer=4,
            consumers=[-1],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([1]), np.array([0])
            ),
        )
    )
    # checking inserted node is the quant node
    self.assertEqual(subgraph.operators[5].opcodeIndex, 0)
    # check if the graph output is updated
    self.assertEqual(subgraph.outputs[0], 9)


if __name__ == "__main__":
  absltest.main()
