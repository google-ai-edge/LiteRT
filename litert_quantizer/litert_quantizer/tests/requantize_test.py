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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E tests for re-quantizing models that already contain quantized weights."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_OpName = qtyping.TFLOperationName
_AlgorithmName = quantizer.AlgorithmName

_DYNAMIC_WI8_AFP32_RECIPE = test_utils.get_path_to_datafile(
    '../recipes/dynamic_wi8_afp32_recipe.json'
)
# Weight input index of both FULLY_CONNECTED and EMBEDDING_LOOKUP.
_WEIGHT_INDEX = 1


def _get_weight_tensors(
    model: qtyping.ModelT, op_name: _OpName
) -> dict[str, qtyping.TensorT]:
  """Returns the weight tensors of all `op_name` ops, keyed by tensor name."""
  op_code = tfl_flatbuffer_utils.TFL_OP_NAME_TO_CODE[op_name]
  weights = {}
  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      builtin_code = flatbuffer_utils.get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex]
      )
      if builtin_code != op_code:
        continue
      tensor = subgraph.tensors[op.inputs[_WEIGHT_INDEX]]
      weights[tfl_flatbuffer_utils.get_tensor_name(tensor)] = tensor
  return weights


def _get_buffer_bytes(model: qtyping.ModelT, tensor: qtyping.TensorT) -> bytes:
  return np.asarray(model.buffers[tensor.buffer].data).tobytes()


def _get_embedding_lookup_model() -> bytes:
  """Returns `embedding_lookup.tflite` with a blockwise-compatible table.

  The (16, 8) table of the checked-in model is too narrow for 32-element
  blocks, so it is tiled in memory to (16, 32) (with a matching output shape).
  The quantizer does not run the model for dynamic recipes, so the model only
  needs to be structurally valid.
  """
  model = tfl_flatbuffer_utils.read_model(
      pathlib.Path(
          test_utils.get_path_to_datafile('models/embedding_lookup.tflite')
      )
  )
  subgraph = model.subgraphs[0]
  op = subgraph.operators[0]
  table = subgraph.tensors[op.inputs[_WEIGHT_INDEX]]
  output = subgraph.tensors[op.outputs[0]]
  table_data = tfl_flatbuffer_utils.get_tensor_data(table, model.buffers)
  wide_table_data = np.tile(table_data, (1, 4))
  table.shape = np.array(wide_table_data.shape, dtype=np.int32)
  output.shape[-1] = wide_table_data.shape[-1]
  if output.shapeSignature is not None:
    output.shapeSignature[-1] = wide_table_data.shape[-1]
  model.buffers[table.buffer].data = np.frombuffer(
      wide_table_data.tobytes(), dtype=np.uint8
  )
  return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


class RequantizeTest(parameterized.TestCase):

  def _assert_blockwise_int4(self, tensor: qtyping.TensorT):
    self.assertEqual(tensor.type, qtyping.TensorType.INT4)
    self.assertEqual(
        tensor.quantization.detailsType,
        qtyping.QuantizationDetails.BlockwiseQuantization,
    )
    self.assertEqual(tensor.quantization.details.blockSize, 32)

  def _assert_requantize_keeps_blockwise_int4_weights(
      self,
      float_model: bytes,
      op_name: _OpName,
      algorithm_key: _AlgorithmName,
  ):
    # First pass: blockwise INT4 weights for `op_name`.
    qt = quantizer.Quantizer(float_model)
    qt.add_dynamic_config(
        regex='.*',
        operation_name=op_name,
        num_bits=4,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
        algorithm_key=algorithm_key,
    )
    first_pass_model = tfl_flatbuffer_utils.read_model(
        qt.quantize().quantized_model
    )
    first_pass_weights = _get_weight_tensors(first_pass_model, op_name)
    self.assertNotEmpty(first_pass_weights)
    for tensor in first_pass_weights.values():
      self._assert_blockwise_int4(tensor)

    # Second pass: dynamic_wi8_afp32 on the already quantized model.
    qt = quantizer.Quantizer(
        bytes(flatbuffer_utils.convert_object_to_bytearray(first_pass_model)),
        _DYNAMIC_WI8_AFP32_RECIPE,
    )
    second_pass_model = tfl_flatbuffer_utils.read_model(
        qt.quantize().quantized_model
    )

    # Weights quantized in the first pass are left untouched.
    second_pass_weights = _get_weight_tensors(second_pass_model, op_name)
    self.assertSameElements(second_pass_weights, first_pass_weights)
    for name, first_pass_tensor in first_pass_weights.items():
      second_pass_tensor = second_pass_weights[name]
      self._assert_blockwise_int4(second_pass_tensor)
      self.assertEqual(
          _get_buffer_bytes(second_pass_model, second_pass_tensor),
          _get_buffer_bytes(first_pass_model, first_pass_tensor),
      )

  @parameterized.parameters(
      _AlgorithmName.MIN_MAX_UNIFORM_QUANT, _AlgorithmName.OCTAV
  )
  def test_requantize_blockwise_int4_fully_connected_succeeds(
      self, algorithm_key
  ):
    float_model = tfl_flatbuffer_utils.get_model_content(
        pathlib.Path(
            test_utils.get_path_to_datafile(
                'models/dequantized_weights/blockwise_i4rangedvalues_fc.tflite'
            )
        )
    )
    self._assert_requantize_keeps_blockwise_int4_weights(
        bytes(float_model), _OpName.FULLY_CONNECTED, algorithm_key
    )

  @parameterized.parameters(
      _AlgorithmName.MIN_MAX_UNIFORM_QUANT, _AlgorithmName.OCTAV
  )
  def test_requantize_blockwise_int4_embedding_lookup_succeeds(
      self, algorithm_key
  ):
    self._assert_requantize_keeps_blockwise_int4_weights(
        _get_embedding_lookup_model(), _OpName.EMBEDDING_LOOKUP, algorithm_key
    )


if __name__ == '__main__':
  absltest.main()
