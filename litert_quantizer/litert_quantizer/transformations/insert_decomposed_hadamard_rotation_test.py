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

import copy
import pathlib

from absl.testing import absltest
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping
from litert_quantizer.transformations import insert_decomposed_hadamard_rotation
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile('..')

# 2-stage rotation with `m = 2` blocks of size `d = 4`. Both factors are
# asymmetric, so inserting a transposed factor changes the model outputs: the
# 3-4-5 rotation `Q` and the sign-flipped Hadamard matrix `diag(s) @ H_4`.
_CROSS_BLOCK_MATRIX = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=np.float32)
_KRONECKER_PARAMS = qtyping.UniformQuantParams(
    num_bits=8,
    quantized_dimension=None,
    scale=np.ones(1),
    zero_point=np.zeros(1),
    hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
        random_binary_vector=np.array([1, -1, 1, 1], dtype=np.int8),
        hadamard_size=4,
        cross_block_matrix=_CROSS_BLOCK_MATRIX,
    ),
)
# The offline weight rotation `W_rot = W @ R` that the inserted ops undo (see
# `hadamard_rotation._rotate_with_diagonal_hadamard`):
# `R = Q.T (x) diag(s) @ H_4`.
_KRONECKER_ROTATION = np.kron(
    _CROSS_BLOCK_MATRIX.T,
    0.5
    * np.array(
        [[1, 1, 1, 1], [-1, 1, -1, 1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=np.float32,
    ),
)
_KRONECKER_OPS = [
    qtyping.BuiltinOperator.RESHAPE,
    qtyping.BuiltinOperator.FULLY_CONNECTED,
    qtyping.BuiltinOperator.TRANSPOSE,
    qtyping.BuiltinOperator.FULLY_CONNECTED,
    qtyping.BuiltinOperator.TRANSPOSE,
    qtyping.BuiltinOperator.RESHAPE,
]


def _get_constant_data(
    model: qtyping.ModelT, tensor: qtyping.TensorT
) -> np.ndarray:
  data = tfl_flatbuffer_utils.get_tensor_data(tensor, model.buffers)
  assert data is not None
  return data


def _set_constant_data(
    model: qtyping.ModelT, tensor: qtyping.TensorT, data: np.ndarray
) -> None:
  model.buffers[tensor.buffer].data = np.frombuffer(
      data.astype(np.float32).tobytes(), dtype=np.uint8
  )


def _get_builtin_codes(
    model: qtyping.ModelT, ops: list[qtyping.OperatorT]
) -> list[int]:
  return [model.operatorCodes[op.opcodeIndex].builtinCode for op in ops]


def _create_interpreter(model: qtyping.ModelT):
  return tfl_interpreter_utils.create_tfl_interpreter(
      bytes(flatbuffer_utils.convert_object_to_bytearray(model))
  )


def _add_signature_output(model: qtyping.ModelT, tensor_id: int) -> None:
  """Adds `tensor_id` as the output of the model's only signature.

  The signature of `embedding_lookup.tflite` has no outputs, so without this the
  transformation has no signature output to update.

  Args:
    model: The model to update.
    tensor_id: The index of the output tensor in the signature's subgraph.
  """
  signature_output = flatbuffer_utils.schema_fb.TensorMapT()
  signature_output.name = b'output_0'
  signature_output.tensorIndex = tensor_id
  (signature_def,) = model.signatureDefs
  signature_def.outputs = [signature_output]


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

  def test_raise_hadamard_size_one(self):
    params = qtyping.UniformQuantParams(
        num_bits=8,
        quantized_dimension=None,
        scale=np.ones(1),
        zero_point=np.zeros(1),
        hadamard=qtyping.UniformQuantParams.HadamardRotationParams(
            random_binary_vector=np.ones(1),
            hadamard_size=1,
        ),
    )
    with self.assertRaisesRegex(ValueError, 'at least 2, got 1'):
      insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
          transformation_utils.TransformationInput(
              tensor_id=0,
              model=self.model,
              subgraph=self.model.subgraphs[0],
              producer=-1,
              consumers=[0],
              quant_params=params,
          )
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

  def test_insert_kronecker_ops_preserves_fc_output(self):
    subgraph = self.model.subgraphs[0]
    (fc_op,) = subgraph.operators
    input_id, weight_id, bias_id = fc_op.inputs
    weight = _get_constant_data(self.model, subgraph.tensors[weight_id])
    bias = _get_constant_data(self.model, subgraph.tensors[bias_id])
    fc_input = np.random.default_rng(0).standard_normal(
        (1, 2, 8), dtype=np.float32
    )
    expected_output = fc_input @ weight.T + bias
    # Rotate the weight as `hadamard_rotation` does before quantization.
    _set_constant_data(
        self.model, subgraph.tensors[weight_id], weight @ _KRONECKER_ROTATION
    )

    info = (
        insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
            transformation_utils.TransformationInput(
                tensor_id=input_id,
                model=self.model,
                subgraph=subgraph,
                producer=-1,
                consumers=[0],
                quant_params=_KRONECKER_PARAMS,
            )
        )
    )

    self.assertEqual(
        _get_builtin_codes(
            self.model, subgraph.operators[: info.num_ops_added]
        ),
        _KRONECKER_OPS,
    )
    interpreter = _create_interpreter(self.model)
    tfl_interpreter_utils.invoke_interpreter_once(interpreter, [fc_input])
    output = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index']
    )
    np.testing.assert_allclose(output, expected_output, atol=1e-5)


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

  def test_insert_kronecker_ops_preserves_embedding_lookup_output(self):
    subgraph = self.model.subgraphs[0]
    (embedding_op,) = subgraph.operators
    _, table_id = embedding_op.inputs
    (output_id,) = embedding_op.outputs
    _add_signature_output(self.model, output_id)
    table = _get_constant_data(self.model, subgraph.tensors[table_id])
    # Rotate the table as `hadamard_rotation` does before quantization.
    _set_constant_data(
        self.model, subgraph.tensors[table_id], table @ _KRONECKER_ROTATION
    )

    info = (
        insert_decomposed_hadamard_rotation.insert_decomposed_hadamard_rotation(
            transformation_utils.TransformationInput(
                tensor_id=output_id,
                model=self.model,
                subgraph=subgraph,
                producer=0,
                consumers=[-1],
                quant_params=_KRONECKER_PARAMS,
            )
        )
    )

    self.assertEqual(
        _get_builtin_codes(
            self.model, subgraph.operators[1 : 1 + info.num_ops_added]
        ),
        _KRONECKER_OPS,
    )
    # The signature returns the un-rotated rows.
    lookup = np.array([5], dtype=np.int32)
    outputs = tfl_interpreter_utils.invoke_interpreter_signature(
        _create_interpreter(self.model), {'lookup': lookup}
    )
    np.testing.assert_allclose(outputs['output_0'], table[lookup], atol=1e-5)

  def test_insert_on_graph_output_updates_only_own_signature(self):
    _add_signature_output(self.model, 2)
    # Add a copy of the subgraph with its own signature, like the separate
    # prefill and decode embedders of an LLM. Signature tensor indices are
    # relative to the signature's subgraph, so rotating the output of subgraph 0
    # must not redirect the copy's output to a tensor of subgraph 0.
    self.model.subgraphs.append(copy.deepcopy(self.model.subgraphs[0]))
    signature_copy = copy.deepcopy(self.model.signatureDefs[0])
    signature_copy.signatureKey = b'lookup_copy'
    signature_copy.subgraphIndex = 1
    self.model.signatureDefs.append(signature_copy)

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

    self.assertEqual(
        self.model.signatureDefs[0].outputs[0].tensorIndex,
        info.output_tensor_id,
    )
    self.assertEqual(self.model.signatureDefs[1].outputs[0].tensorIndex, 2)


if __name__ == '__main__':
  absltest.main()
