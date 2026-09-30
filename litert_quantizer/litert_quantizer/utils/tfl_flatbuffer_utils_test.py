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

"""Tests for tfl_flatbuffer_utils.py."""

import pathlib

from absl.testing import absltest
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../tests/models")


# TODO: b/328830092 - Add test cases for model require buffer offset.
class FlatbufferUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "conv_fc_mnist.tflite"
    )

    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)

  def test_get_model_buffer(self):
    model_buffer = tfl_flatbuffer_utils.get_model_buffer(self._test_model_path)
    file_stats = pathlib.Path(self._test_model_path).stat()
    self.assertLen(model_buffer, file_stats.st_size)

  def test_parse_op_tensors(self):
    subgraph0 = self._test_model.subgraphs[0]
    conv2d_op = subgraph0.operators[0]
    op_tensors = tfl_flatbuffer_utils.parse_op_tensors(
        conv2d_op, subgraph0.tensors
    )
    # conv2d have three inputs and one output
    self.assertLen(op_tensors, 4)

    average_pool_op = subgraph0.operators[1]
    op_tensors = tfl_flatbuffer_utils.parse_op_tensors(
        average_pool_op, subgraph0.tensors
    )
    # averagepool have one input and one output
    self.assertLen(op_tensors, 2)

  def test_parse_fc_bmm_conv_tensors(self):
    subgraph0 = self._test_model.subgraphs[0]
    conv2d_op = subgraph0.operators[0]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            conv2d_op, subgraph0.tensors
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 28, 28, 1))
    self.assertEqual(tuple(weight.shape), (8, 3, 3, 1))
    self.assertEqual(tuple(bias.shape), (8,))  # pyrefly: ignore[missing-attribute]
    self.assertEqual(tuple(output.shape), (1, 28, 28, 8))

    fc_with_bias = subgraph0.operators[3]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            fc_with_bias,
            subgraph0.tensors,
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 1568))
    self.assertEqual(tuple(weight.shape), (32, 1568))
    self.assertEqual(tuple(bias.shape), (32,))  # pyrefly: ignore[missing-attribute]
    self.assertEqual(tuple(output.shape), (1, 32))

    fc_no_bias = subgraph0.operators[4]
    inputs, weight, bias, output = (
        tfl_flatbuffer_utils.parse_fc_bmm_conv_tensors(
            fc_no_bias,
            subgraph0.tensors,
        )
    )
    self.assertEqual(tuple(inputs.shape), (1, 32))
    self.assertEqual(tuple(weight.shape), (10, 32))
    self.assertIsNone(bias)
    self.assertEqual(tuple(output.shape), (1, 10))

  def test_buffer_to_tensors(self):
    buffer_to_tensor_map = tfl_flatbuffer_utils.buffer_to_tensors(
        self._test_model
    )
    # Read from Netron/Model Explorer
    buffer_id = self._test_model.subgraphs[0].tensors[5].buffer
    tensors = buffer_to_tensor_map[buffer_id]
    self.assertLen(tensors, 1)
    conv2d_filter_tensor = tensors[0]
    self.assertEqual(tuple(conv2d_filter_tensor.shape), (8, 3, 3, 1))

  def test_buffer_to_tensors_has_unique_values(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "constant_tensor_and_buffer_only_sharing_weight_fcs.tflite",
    )
    test_model = tfl_flatbuffer_utils.read_model(test_model_path)
    buffer_to_tensor_map = tfl_flatbuffer_utils.buffer_to_tensors(test_model)
    self.assertEqual(sum(len(v) for v in buffer_to_tensor_map.values()), 8)
    # The following buffer is shared by two tensors, each shared by two FC ops.
    # This is where before we had multiple enrties for the same tensor.
    tensor_for_name = {}
    for subgraph in test_model.subgraphs:
      for tensor in subgraph.tensors:
        tensor_for_name[tfl_flatbuffer_utils.get_tensor_name(tensor)] = tensor
    buffer_id = tensor_for_name["arith.constant"].buffer
    self.assertLen(buffer_to_tensor_map[buffer_id], 2)
    got_tensor_names = [
        tfl_flatbuffer_utils.get_tensor_name(tensor)
        for tensor in buffer_to_tensor_map[buffer_id]
    ]
    self.assertEqual(
        got_tensor_names,
        ["arith.constant", "arith.constant1"],
    )

  def test_get_tensor_name(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_tensors = subgraph0.tensors
    conv2d_op = subgraph0.operators[0]
    weight_tensor = subgraph_tensors[conv2d_op.inputs[1]]
    weight_tensor_name = tfl_flatbuffer_utils.get_tensor_name(weight_tensor)
    self.assertEqual(weight_tensor_name, "sequential/conv2d/Conv2D")

  # TODO: b/325123193 - test tensor with data outside of flatbuffer.
  def test_get_tensor_data(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_tensors = subgraph0.tensors
    conv2d_op = subgraph0.operators[0]
    # Check tensor with data
    weight_tensor = subgraph_tensors[conv2d_op.inputs[1]]
    weight_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        weight_tensor, self._test_model.buffers
    )
    self.assertEqual(
        tuple(weight_tensor.shape), tuple(weight_tensor_data.shape)  # pytype: disable=attribute-error
    )
    self.assertAlmostEqual(weight_tensor_data[0][0][0][0], -0.12941549718379974)  # pyrefly: ignore[unsupported-operation]

    # Check tensor with no data
    input_tensor = subgraph_tensors[conv2d_op.inputs[0]]
    input_tensor_data = tfl_flatbuffer_utils.get_tensor_data(
        input_tensor, self._test_model.buffers
    )
    self.assertIsNone(input_tensor_data)

  def test_has_same_quantization_succeeds(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor0.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor1.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    self.assertTrue(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_succeds_not_quantized(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.type = 10
    self.assertTrue(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_fails_different_scale(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    self.assertFalse(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_has_same_quantization_fails_different_zp(self):
    tensor0, tensor1 = self._test_model.subgraphs[0].tensors[:2]
    tensor0.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor0.quantization.zeroPoint = np.array([3, 2, 1]).astype(np.int32)
    tensor1.quantization.scale = np.array([1, 2, 3]).astype(np.float32)
    tensor1.quantization.zeroPoint = np.array([1, 2, 3]).astype(np.int32)
    self.assertFalse(
        tfl_flatbuffer_utils.has_same_quantization(tensor0, tensor1)
    )

  def test_check_is_float_model_true_when_model_is_float(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "conv_fc_mnist.tflite"
    )
    model = tfl_flatbuffer_utils.read_model(test_model_path)
    self.assertTrue(tfl_flatbuffer_utils.is_float_model(model))

  def test_check_is_float_model_false_when_model_is_quantized(self):
    test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH) / "mnist_quantized.tflite"
    )
    model = tfl_flatbuffer_utils.read_model(test_model_path)
    self.assertFalse(tfl_flatbuffer_utils.is_float_model(model))

  def test_get_subgraph_input_output_operators(self):
    subgraph = self._test_model.subgraphs[0]
    input_op, output_op = (
        tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
    )
    self.assertEqual(input_op.op_key, qtyping.TFLOperationName.INPUT)
    self.assertEmpty(input_op.inputs)
    self.assertListEqual(list(input_op.outputs), [0])
    self.assertEqual(output_op.op_key, qtyping.TFLOperationName.OUTPUT)
    self.assertListEqual(list(output_op.inputs), [12])
    self.assertEmpty(output_op.outputs)


if __name__ == "__main__":
  absltest.main()
