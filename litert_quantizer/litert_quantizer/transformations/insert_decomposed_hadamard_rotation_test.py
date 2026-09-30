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

"""Test insertion of the Decomposed Hadamard rotation ops."""

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import insert_decomposed_hadamard_rotation
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')


class InsertDecomposedHadamardRotationFullyConnectedTest(absltest.TestCase):

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
      insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
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
      insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
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
      insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[-1],
              quant_params=self.params,
          ),
      )

  def test_insert_decomposed_ops(self):
    # Insert Decomposed Hadamard ops before fully_connected
    info = (
        insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
            transformation_utils.TransformationInput(
                tensor_id=0,
                model=self.model,
                subgraph=self.model.subgraphs[0],
                producer=-1,
                consumers=[0],  # Consumer is the FC op
                quant_params=self.params,
            )
        )
    )
    subgraph = self.model.subgraphs[0]
    self.assertEqual(info.op_id, 0)
    self.assertEqual(info.num_ops_added, 3)
    # Model had 4 tensors, added 6 tensors (3 activations 3 constants).
    self.assertEqual(info.output_tensor_id, 9)
    self.assertLen(subgraph.tensors, 10)
    # Model had 1 op code, added RESHAPE and FC.
    self.assertLen(self.model.operatorCodes, 3)
    self.assertEqual(
        self.model.operatorCodes[1].builtinCode,
        qtyping.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(
        self.model.operatorCodes[2].builtinCode,
        qtyping.BuiltinOperator.FULLY_CONNECTED,
    )

    # Op 0: RESHAPE
    reshape_op = subgraph.operators[0]
    self.assertEqual(
        self.model.operatorCodes[reshape_op.opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(reshape_op.inputs[0], 0)  # Graph input
    self.assertEqual(reshape_op.outputs[0], 5)  # Reshape output

    # Op 1: FULLY_CONNECTED
    fc_op = subgraph.operators[1]
    self.assertEqual(
        self.model.operatorCodes[fc_op.opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.FULLY_CONNECTED,
    )
    self.assertEqual(fc_op.inputs[0], 5)  # Reshape output
    self.assertEqual(fc_op.inputs[1], 6)  # Hadamard matrix tensor
    self.assertEqual(subgraph.tensors[6].type, qtyping.TensorType.FLOAT32)
    self.assertEqual(fc_op.outputs[0], 7)  # FC output

    # Op 2: RESHAPE (post)
    post_reshape_op = subgraph.operators[2]
    self.assertEqual(
        self.model.operatorCodes[post_reshape_op.opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.RESHAPE,
    )
    self.assertEqual(post_reshape_op.inputs[0], 7)  # FC output
    self.assertEqual(post_reshape_op.outputs[0], 9)  # Post Reshape output

    # Op 3: Original FULLY_CONNECTED
    orig_fc_op = subgraph.operators[3]
    self.assertEqual(
        self.model.operatorCodes[orig_fc_op.opcodeIndex].builtinCode,
        qtyping.BuiltinOperator.FULLY_CONNECTED,
    )
    # Input to the original FC is the post reshape output
    self.assertEqual(orig_fc_op.inputs[0], 9)


class InsertDecomposedHadamardRotationEmbeddingLookupTest(absltest.TestCase):

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

  def test_insert_decomposed_ops(self):
    # Insert Decomposed Hadamard ops after embedding_lookup
    info = (
        insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
            transformation_utils.TransformationInput(
                tensor_id=2,  # Output of embedding_lookup
                model=self.model,
                subgraph=self.model.subgraphs[0],
                producer=0,
                consumers=[-1],  # Output is a graph output
                quant_params=self.params,
            )
        )
    )
    subgraph = self.model.subgraphs[0]
    self.assertEqual(info.op_id, 1)
    self.assertEqual(info.num_ops_added, 3)
    # Model had 3 tensors, added 6 (3 activations 3 constants).
    self.assertEqual(info.output_tensor_id, 8)
    self.assertLen(subgraph.tensors, 9)
    # Model had 1 op code, added RESHAPE and FC.
    self.assertLen(self.model.operatorCodes, 3)

    # Op 0: EMBEDDING_LOOKUP (Original)
    # Op 1: RESHAPE
    reshape_op = subgraph.operators[1]
    self.assertEqual(reshape_op.inputs[0], 2)  # Embedding lookup output
    self.assertEqual(reshape_op.outputs[0], 4)

    # Op 2: FULLY_CONNECTED
    fc_op = subgraph.operators[2]
    self.assertEqual(fc_op.inputs[0], 4)
    self.assertEqual(fc_op.inputs[1], 5)  # Hadamard matrix
    self.assertEqual(subgraph.tensors[5].type, qtyping.TensorType.FLOAT32)
    self.assertEqual(fc_op.outputs[0], 6)

    # Op 3: RESHAPE (post)
    post_reshape_op = subgraph.operators[3]
    self.assertEqual(post_reshape_op.inputs[0], 6)
    self.assertEqual(post_reshape_op.outputs[0], 8)

    # Check graph output
    self.assertIn(8, subgraph.outputs)
    self.assertNotIn(2, subgraph.outputs)


if __name__ == '__main__':
  absltest.main()
