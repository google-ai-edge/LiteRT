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

"""Test insertion of the Hadamard rotation custom op."""

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import insert_hadamard_rotation
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')


class InsertHadamardRotationFullyConnectedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/single_fc_bias.tflite'
    )
    self.model = tfl_flatbuffer_utils.read_model(model_path)
    self.params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.ones(1),
        zero_point=np.zeros(1),
        hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
            random_binary_vector=np.ones(1),
            hadamard_size=2,
        ),
    )

  def test_raise_unsupported_qparams(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: 'uniform quantization' in str(err)
    ):
      insert_hadamard_rotation.insert_hadamard_rotation(
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

  def test_raise_missing_hadamard_data(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: 'quantization params are not set' in str(err)
    ):
      insert_hadamard_rotation.insert_hadamard_rotation(
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

  def test_raise_non_float32_tensor(self):
    self.model.subgraphs[0].tensors[0].type = qtyping.TensorType.INT32
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: 'float32 tensors' in str(err)
    ):
      insert_hadamard_rotation.insert_hadamard_rotation(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[-1],
              quant_params=self.params,
          ),
      )

  def test_insert_single_custom_op(self):
    # Insert aeq.hadamard_rotation before fully_connected
    info = insert_hadamard_rotation.insert_hadamard_rotation(
        transformation_utils.TransformationInput(
            tensor_id=0,
            model=self.model,
            subgraph=self.model.subgraphs[0],
            producer=-1,
            consumers=[-1],
            quant_params=self.params,
        )
    )
    subgraph = self.model.subgraphs[0]
    self.assertEqual(info.op_id, 0)
    self.assertEqual(info.num_ops_added, 1)
    # Model had 4 tensors, added 1.
    self.assertEqual(info.output_tensor_id, 4)
    self.assertLen(subgraph.tensors, 5)
    # Model had 1 op, added a new one.
    self.assertLen(self.model.operatorCodes, 2)
    self.assertEqual(
        self.model.operatorCodes[1].builtinCode,
        qtyping.BuiltinOperator.CUSTOM,
    )
    # First op is now the custom op, precedes fully_connected.
    self.assertEqual(
        self.model.operatorCodes[subgraph.operators[0].opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.CUSTOM,
    )
    # Input to the custom op is graph input
    self.assertEqual(subgraph.operators[0].inputs[0], 0)
    # Input to the FC is the custom op output
    self.assertEqual(subgraph.operators[1].inputs[0], 4)


class InsertHadamardRotationEmbeddingLookupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / 'tests/models/embedding_lookup.tflite'
    )
    self.model = tfl_flatbuffer_utils.read_model(model_path)
    self.params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.ones(1),
        zero_point=np.zeros(1),
        hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
            random_binary_vector=np.ones(1),
            hadamard_size=2,
        ),
    )

  def test_insert_single_custom_op(self):
    # Insert aeq.hadamard_rotation after embedding_lookup
    info = insert_hadamard_rotation.insert_hadamard_rotation(
        transformation_utils.TransformationInput(
            tensor_id=2,
            model=self.model,
            subgraph=self.model.subgraphs[0],
            producer=0,
            consumers=[-1],
            quant_params=self.params,
        )
    )
    subgraph = self.model.subgraphs[0]
    self.assertEqual(info.op_id, 1)
    self.assertEqual(info.num_ops_added, 1)
    # Model had 3 tensors, added 1.
    self.assertEqual(info.output_tensor_id, 3)
    self.assertLen(subgraph.tensors, 4)
    # Model had 1 op, added a new one.
    self.assertLen(self.model.operatorCodes, 2)
    self.assertEqual(
        self.model.operatorCodes[1].builtinCode,
        qtyping.BuiltinOperator.CUSTOM,
    )
    # Second op is now the custom op, after embedding_lookup.
    self.assertEqual(
        self.model.operatorCodes[subgraph.operators[1].opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.CUSTOM,
    )
    # Input to the custom op is embedding's output
    self.assertEqual(subgraph.operators[1].inputs[0], 2)
    # Custom op's output is the new tensor
    self.assertEqual(subgraph.operators[1].outputs[0], 3)


if __name__ == '__main__':
  absltest.main()
