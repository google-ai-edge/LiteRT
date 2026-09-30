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

"""test for quantize tensor."""

import logging
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.transformations import quantize_tensor
from litert_quantizer.transformations import transformation_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("..")


class QuantizeTensorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_test_model_path = str(
        pathlib.Path(TEST_DATA_PREFIX_PATH)
        / "tests/models/insert_dequant_test.tflite"
    )
    self._model = tfl_flatbuffer_utils.read_model(self._orig_test_model_path)

  def test_quantize_constant_tensor(self):
    """test quantizing a constant tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.ones([1, 112, 112, 32], dtype=np.int8)
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    buffer_id = subgraph.tensors[7].buffer
    self.assertListEqual(
        np.array(model.buffers[buffer_id].data).tolist(),
        data.flatten().tolist(),
    )
    quant_param = subgraph.tensors[7].quantization
    self.assertListEqual(np.array(quant_param.scale).tolist(), [1])
    self.assertEqual(np.array(quant_param.zeroPoint).tolist(), [1])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_quantize_activation_tensor(self):
    """test quantizing an activation tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=4,
            model=model,
            subgraph=subgraph,
            producer=1,
            consumers=[3],
            quant_params=qtyping.UniformQuantParams(
                8, None, np.array([22]), np.array([127])
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    quant_param = subgraph.tensors[4].quantization
    self.assertListEqual(np.array(quant_param.scale).tolist(), [22])
    self.assertListEqual(np.array(quant_param.zeroPoint).tolist(), [127])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_duplicate_quantization_is_detected(self):
    """test quantizing a constant tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.ones([1, 112, 112, 32], dtype=np.int8)
    buffer_id = subgraph.tensors[7].buffer
    quant_params = qtyping.UniformQuantParams(
        8, None, np.ones(1), np.ones(1), True, data
    )
    orig_buffer_data = model.buffers[buffer_id].data

    # Buffer was written with the same quant params, don't overwrite.
    buffer_origin = {buffer_id: quant_params}
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=quant_params,
            buffer_origin=buffer_origin,  # pyrefly: ignore[bad-argument-type]
        )
    )

    # Transformation is still correctly applied.
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    tensor_quant_params = subgraph.tensors[7].quantization
    self.assertListEqual(np.array(tensor_quant_params.scale).tolist(), [1])
    self.assertEqual(np.array(tensor_quant_params.zeroPoint).tolist(), [1])
    self.assertEqual(tensor_quant_params.quantizedDimension, 0)

    # Quantized buffer was not packed.
    self.assertIs(model.buffers[buffer_id].data, orig_buffer_data)

  def test_overwriting_quantization_is_detected(self):
    """test quantizing a constant tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.ones([1, 112, 112, 32], dtype=np.int8)
    buffer_id = subgraph.tensors[7].buffer
    quant_params_4bit = qtyping.UniformQuantParams(
        4, None, np.ones(1), np.ones(1), True, data
    )
    quant_params_8bit = qtyping.UniformQuantParams(
        8, None, np.ones(1), np.ones(1), True, data
    )
    orig_buffer_data = model.buffers[buffer_id].data

    # Buffer was already written but with different quant params, so it should
    # be overwritten.
    buffer_origin = {buffer_id: quant_params_4bit}
    with self.assertLogs(level=logging.WARNING) as logs:
      ret = quantize_tensor.quantize_tensor(
          transformation_utils.TransformationInput(
              tensor_id=7,
              model=model,
              subgraph=subgraph,
              producer=-1,
              consumers=[4],
              quant_params=quant_params_8bit,
              buffer_origin=buffer_origin,  # pyrefly: ignore[bad-argument-type]
          )
      )

    # Make sure a warning was logged.
    self.assertIn(
        "overriding other previously quantized data", "\n".join(logs.output)
    )

    # Transformation is still correctly applied.
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    tensor_quant_params = subgraph.tensors[7].quantization
    self.assertListEqual(np.array(tensor_quant_params.scale).tolist(), [1])
    self.assertEqual(np.array(tensor_quant_params.zeroPoint).tolist(), [1])
    self.assertEqual(tensor_quant_params.quantizedDimension, 0)

    # Quantized buffer was packed.
    self.assertIsNot(model.buffers[buffer_id].data, orig_buffer_data)
    self.assertIs(buffer_origin[buffer_id], quant_params_8bit)

  def test_quantize_tensor_with_per_channel_quantization(self):
    """test quantizing an activation tensor."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=4,
            model=model,
            subgraph=subgraph,
            producer=1,
            consumers=[3],
            quant_params=qtyping.UniformQuantParams(
                8, 3, np.ones([22]), np.zeros([22])
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    quant_param = subgraph.tensors[4].quantization
    self.assertListEqual(
        np.array(quant_param.scale).tolist(), np.ones([22]).tolist()
    )
    self.assertListEqual(
        np.array(quant_param.zeroPoint).tolist(), np.zeros([22]).tolist()
    )
    self.assertEqual(quant_param.quantizedDimension, 3)

  def test_quantize_tensor_with_nonlinear_quantization(self):
    """test quantizing an activation tensor with non-linear quantization."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=4,
            model=model,
            subgraph=subgraph,
            producer=1,
            consumers=[3],
            quant_params=qtyping.NonLinearQuantParams(16, None),
        )
    )
    self.assertEqual(subgraph.tensors[4].type, qtyping.TensorType.FLOAT16)

  def test_blockwise_quantization_with_zero_point(self):
    """Test blockwise quantization with explicit zero point."""
    subgraph = self._model.subgraphs[0]
    model = self._model
    tensor_wh = 112
    test_tensor_id = 7
    data = np.ones([1, tensor_wh, tensor_wh, 32]).astype(np.int8)
    quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=test_tensor_id,
            model=model,
            subgraph=subgraph,
            producer=1,
            consumers=[3],
            quant_params=qtyping.UniformQuantParams(
                num_bits=8,
                quantized_dimension=None,
                scale=np.ones([1, tensor_wh, tensor_wh, 1]).astype(np.float16),
                zero_point=np.zeros([1, tensor_wh, tensor_wh, 1]),
                symmetric=True,
                quantized_data=data,
                block_size=32,
            ),
        )
    )
    quant_param = subgraph.tensors[test_tensor_id].quantization
    self.assertEqual(
        quant_param.detailsType,
        qtyping.QuantizationDetails.BlockwiseQuantization,
    )
    self.assertEqual(quant_param.details.blockSize, 32)
    # Check if the scale and zero point tensors are inserted correctly.
    self.assertEqual(quant_param.details.scales, 9)
    # So far we don't have zero point in blockwise quantization.
    self.assertEqual(quant_param.details.zeroPoints, -1)

  def test_int2_constant_packed_correctly(self):
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.array(
        [
            0b00,  # First byte
            0b01,
            0b10,
            0b11,
            0b11,  # Second byte
            0b10,
            0b01,
            0b00,
            0b10,  # Third byte, four bits of padding will be added.
            0b11,
        ],
        dtype=np.int8,
    )
    expected = np.array([0b11100100, 0b00011011, 0b00001110])
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                2, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    buffer_id = subgraph.tensors[7].buffer
    np.testing.assert_array_equal(model.buffers[buffer_id].data, expected)
    quant_param = subgraph.tensors[7].quantization
    np.testing.assert_array_equal(quant_param.scale, [1])
    np.testing.assert_array_equal(quant_param.zeroPoint, [1])
    self.assertEqual(quant_param.quantizedDimension, 0)

  def test_int4_constant_packed_correctly(self):
    subgraph = self._model.subgraphs[0]
    model = self._model
    data = np.array(
        [
            0x0,
            0x1,
            0x2,
            0x3,
            0x4,
            0x5,
            0x6,
            0x7,
            0x8,
            0x9,
            0xA,
            0xB,
            0xC,
            0xD,
            0xE,
        ],
        dtype=np.int8,
    )
    expected = np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0x0E])
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=7,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                4, None, np.ones(1), np.ones(1), True, data
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    buffer_id = subgraph.tensors[7].buffer
    np.testing.assert_array_equal(model.buffers[buffer_id].data, expected)
    quant_param = subgraph.tensors[7].quantization
    np.testing.assert_array_equal(quant_param.scale, [1])
    np.testing.assert_array_equal(quant_param.zeroPoint, [1])
    self.assertEqual(quant_param.quantizedDimension, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="int5",
          num_bits=5,
      ),
      dict(
          testcase_name="int3",
          num_bits=3,
      ),
  )
  def test_int_constant_not_packed(self, num_bits):
    subgraph = self._model.subgraphs[0]
    model = self._model
    tensor_id = 7
    data = np.array([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7], dtype=np.int8)
    expected = np.array([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7])
    ret = quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=tensor_id,
            model=model,
            subgraph=subgraph,
            producer=-1,
            consumers=[4],
            quant_params=qtyping.UniformQuantParams(
                num_bits=num_bits,
                quantized_dimension=None,
                scale=np.ones(1),
                zero_point=np.ones(1),
                symmetric=True,
                quantized_data=data,
            ),
        )
    )
    self.assertEqual(ret.op_id, 0)
    self.assertEqual(ret.num_ops_added, 0)
    buffer_id = subgraph.tensors[tensor_id].buffer
    np.testing.assert_array_equal(model.buffers[buffer_id].data, expected)
    quant_param = subgraph.tensors[tensor_id].quantization
    np.testing.assert_array_equal(quant_param.scale, [1])
    np.testing.assert_array_equal(quant_param.zeroPoint, [1])
    self.assertEqual(quant_param.quantizedDimension, 0)


if __name__ == "__main__":
  absltest.main()
