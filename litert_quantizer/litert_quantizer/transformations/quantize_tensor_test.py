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

from collections.abc import Sequence
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

  def _quantize_block_shape_tensor(
      self,
      tensor_shape: list[int],
      block_size: int,
      scale: np.ndarray,
      quantized_dimension: int | None = None,
      quantized_dimensions: tuple[int, ...] | None = None,
      zero_point: np.ndarray | None = None,
      num_bits: int = 8,
      tensor_id: int = 4,
      consumers: Sequence[int] = (),
  ) -> qtyping.QuantizationParametersT:
    """Quantizes `tensor_id` with block_shape params, returns quantization."""
    subgraph = self._model.subgraphs[0]
    subgraph.tensors[tensor_id].shape = np.array(tensor_shape, dtype=np.int32)
    quantize_tensor.quantize_tensor(
        transformation_utils.TransformationInput(
            tensor_id=tensor_id,
            model=self._model,
            subgraph=subgraph,
            producer=1,
            consumers=list(consumers),
            quant_params=qtyping.UniformQuantParams(
                num_bits=num_bits,
                quantized_dimension=quantized_dimension,
                quantized_dimensions=quantized_dimensions,
                scale=scale,
                zero_point=(
                    np.zeros_like(scale) if zero_point is None else zero_point
                ),
                block_size=block_size,
            ),
        )
    )
    return subgraph.tensors[tensor_id].quantization

  @parameterized.named_parameters(
      dict(
          testcase_name="channelwise",
          tensor_shape=[2, 3, 64],
          quantized_dimensions=(0, 1),
          scale_shape=(2, 3, 1),
          expected_block_shape=[1, 1, 64],
          expected_block_size=64,
      ),
      dict(
          testcase_name="blockwise",
          tensor_shape=[2, 3, 64],
          quantized_dimensions=(0, 1),
          block_size=32,
          scale_shape=(2, 3, 2),
          expected_block_shape=[1, 1, 32],
          expected_block_size=32,
      ),
      dict(
          testcase_name="blocks_not_along_last_dim",
          tensor_shape=[2, 3, 64],
          quantized_dimensions=(0, 2),
          scale_shape=(2, 1, 64),
          expected_block_shape=[1, 3, 1],
          expected_block_size=0,
      ),
      dict(
          testcase_name="channelwise_4d",
          tensor_shape=[2, 3, 1, 64],
          quantized_dimensions=(0, 1),
          scale_shape=(2, 3, 1, 1),
          expected_block_shape=[1, 1, 1, 64],
          expected_block_size=64,
      ),
      dict(
          testcase_name="blockwise_4d",
          tensor_shape=[2, 3, 1, 64],
          quantized_dimensions=(0, 1),
          block_size=32,
          scale_shape=(2, 3, 1, 2),
          expected_block_shape=[1, 1, 1, 32],
          expected_block_size=32,
      ),
      dict(
          testcase_name="per_tensor_2d",
          tensor_shape=[4, 8],
          scale_shape=(1, 1),
          zero_point=np.array([[-0.5]], dtype=np.float32),
          num_bits=2,
          expected_block_shape=[4, 8],
          expected_block_size=0,
      ),
      dict(
          testcase_name="per_channel_2d",
          tensor_shape=[4, 8],
          quantized_dimension=0,
          scale_shape=(4, 1),
          zero_point=np.array([[-0.5]], dtype=np.float32),
          num_bits=2,
          expected_block_shape=[1, 8],
          expected_block_size=8,
      ),
      dict(
          testcase_name="blockwise_2d",
          tensor_shape=[4, 8],
          quantized_dimension=1,
          block_size=4,
          scale_shape=(4, 2),
          zero_point=np.array([[-0.5]], dtype=np.float32),
          num_bits=2,
          expected_block_shape=[1, 4],
          expected_block_size=4,
      ),
  )
  def test_quantize_tensor_block_shape(
      self,
      tensor_shape,
      scale_shape,
      expected_block_shape,
      expected_block_size,
      quantized_dimension=None,
      quantized_dimensions=None,
      block_size=0,
      zero_point=None,
      num_bits=8,
  ):
    details = self._quantize_block_shape_tensor(
        tensor_shape=tensor_shape,
        block_size=block_size,
        scale=np.ones(scale_shape, dtype=np.float32),
        quantized_dimension=quantized_dimension,
        quantized_dimensions=quantized_dimensions,
        zero_point=zero_point,
        num_bits=num_bits,
    ).details
    self.assertSequenceEqual(list(details.blockShape), expected_block_shape)
    self.assertEqual(details.blockSize, expected_block_size)

  def test_quantize_tensor_with_multi_axis_quantization_writes_scales(self):
    scale = np.random.uniform(size=(2, 3, 2))
    quant_param = self._quantize_block_shape_tensor(
        tensor_shape=[2, 3, 64],
        quantized_dimensions=(0, 1),
        block_size=32,
        scale=scale,
    )
    self.assertEqual(
        quant_param.detailsType,
        qtyping.QuantizationDetails.BlockwiseQuantization,
    )
    self.assertEqual(quant_param.details.zeroPoints, -1)
    scales_tensor = self._model.subgraphs[0].tensors[quant_param.details.scales]
    self.assertEqual(scales_tensor.type, qtyping.TensorType.FLOAT32)
    np.testing.assert_array_equal(
        tfl_flatbuffer_utils.get_tensor_data(
            scales_tensor, self._model.buffers
        ),
        scale.astype(np.float32),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="2d_per_tensor",
          tensor_shape=[4, 8],
          quantized_dimensions=None,
          block_size=0,
          scale=np.array([[0.5]], dtype=np.float32),
          expected_scale_shape=[1, 1],
          expected_zp_shape=[1, 1],
      ),
      dict(
          testcase_name="3d_multi_axis",
          tensor_shape=[2, 3, 64],
          quantized_dimensions=(0, 1),
          block_size=32,
          scale=np.full((2, 3, 2), 0.5, dtype=np.float32),
          expected_scale_shape=[2, 3, 2],
          expected_zp_shape=[1, 1, 1],
      ),
  )
  def test_quantize_tensor_with_centered_int2_writes_scales_and_zero_points(
      self,
      tensor_shape,
      quantized_dimensions,
      block_size,
      scale,
      expected_scale_shape,
      expected_zp_shape,
  ):
    subgraph = self._model.subgraphs[0]
    existing_scale_id = transformation_utils.add_new_constant_tensor(
        "existing_scale",
        scale,
        qtyping.TensorType.FLOAT32,
        subgraph,
        self._model,
    )
    existing_zp_id = transformation_utils.add_new_constant_tensor(
        "existing_zp",
        np.array([-0.5], dtype=np.float32),
        qtyping.TensorType.FLOAT32,
        subgraph,
        self._model,
    )

    quant_param = self._quantize_block_shape_tensor(
        tensor_shape=tensor_shape,
        block_size=block_size,
        scale=scale,
        quantized_dimensions=quantized_dimensions,
        zero_point=np.array([-0.5], dtype=np.float32),
        num_bits=2,
        tensor_id=7,
    )
    self.assertEqual(subgraph.tensors[7].type, qtyping.TensorType.INT2)
    self.assertEqual(
        quant_param.detailsType,
        qtyping.QuantizationDetails.BlockwiseQuantization,
    )
    details = quant_param.details

    scales_tensor = subgraph.tensors[details.scales]
    self.assertEqual(scales_tensor.type, qtyping.TensorType.FLOAT32)
    self.assertSequenceEqual(list(scales_tensor.shape), expected_scale_shape)
    self.assertNotEqual(
        scales_tensor.buffer, subgraph.tensors[existing_scale_id].buffer
    )
    np.testing.assert_array_equal(
        tfl_flatbuffer_utils.get_tensor_data(
            scales_tensor, self._model.buffers
        ),
        scale,
    )

    zp_tensor = subgraph.tensors[details.zeroPoints]
    self.assertEqual(zp_tensor.type, qtyping.TensorType.FLOAT32)
    self.assertSequenceEqual(list(zp_tensor.shape), expected_zp_shape)
    self.assertNotEqual(
        zp_tensor.buffer, subgraph.tensors[existing_zp_id].buffer
    )
    np.testing.assert_array_equal(
        tfl_flatbuffer_utils.get_tensor_data(zp_tensor, self._model.buffers),
        np.full(expected_zp_shape, -0.5, dtype=np.float32),
    )

  def test_quantize_tensor_with_multi_axis_quantization_bad_scale_shape_fails(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, r"does not match the expected shape \(2, 3, 2\)"
    ):
      self._quantize_block_shape_tensor(
          tensor_shape=[2, 3, 64],
          quantized_dimensions=(0, 1),
          block_size=32,
          scale=np.ones((2, 3, 4)),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="non_uniform_zero_point",
          quantized_dimension=0,
          block_size=0,
          scale=np.ones((4, 1), dtype=np.float32),
          zero_point=np.array([[-0.5], [0.5]], dtype=np.float32),
          expected_error="Expected one zero point",
      ),
      dict(
          testcase_name="blockwise_without_quantized_dim",
          quantized_dimension=None,
          block_size=4,
          scale=np.ones((4, 2), dtype=np.float32),
          zero_point=np.array([[-0.5]], dtype=np.float32),
          expected_error=(
              "quantized_dimension must be specified when block_size > 0"
          ),
      ),
  )
  def test_quantize_centered_int2_invalid_params_raises(
      self,
      quantized_dimension,
      block_size,
      scale,
      zero_point,
      expected_error,
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      self._quantize_block_shape_tensor(
          tensor_shape=[4, 8],
          block_size=block_size,
          scale=scale,
          quantized_dimension=quantized_dimension,
          zero_point=zero_point,
          num_bits=2,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="fc_drq_float32_input",
          builtin_code=qtyping.BuiltinOperator.FULLY_CONNECTED,
          input_type=qtyping.TensorType.FLOAT32,
          expected_version=17,
      ),
      dict(
          testcase_name="fc_srq_int8_input",
          builtin_code=qtyping.BuiltinOperator.FULLY_CONNECTED,
          input_type=qtyping.TensorType.INT8,
          expected_version=16,
      ),
      dict(
          testcase_name="embedding_lookup",
          builtin_code=qtyping.BuiltinOperator.EMBEDDING_LOOKUP,
          input_type=qtyping.TensorType.INT32,
          expected_version=6,
      ),
      dict(
          testcase_name="dequantize",
          builtin_code=qtyping.BuiltinOperator.DEQUANTIZE,
          input_type=qtyping.TensorType.INT2,
          expected_version=10,
      ),
      dict(
          testcase_name="custom_op",
          builtin_code=qtyping.BuiltinOperator.CUSTOM,
          custom_op_name="test_custom_op",
          input_type=qtyping.TensorType.FLOAT32,
          expected_version=1,
      ),
  )
  def test_quantize_centered_int2_updates_consumer_op_version(
      self, builtin_code, input_type, expected_version, custom_op_name=None
  ):
    subgraph = self._model.subgraphs[0]
    consumer_op = subgraph.operators[4]
    subgraph.tensors[consumer_op.inputs[0]].type = input_type
    opcode_idx = transformation_utils.add_op_code(
        builtin_code,
        self._model.operatorCodes,
        custom_op_name=custom_op_name,
    )
    consumer_op.opcodeIndex = opcode_idx
    self._model.operatorCodes[opcode_idx].version = 1

    self._quantize_block_shape_tensor(
        tensor_shape=[4, 8],
        block_size=0,
        scale=np.ones((4, 1), dtype=np.float32),
        quantized_dimension=0,
        zero_point=np.array([[-0.5]], dtype=np.float32),
        num_bits=2,
        tensor_id=7,
        consumers=[4],
    )
    self.assertEqual(
        self._model.operatorCodes[opcode_idx].version, expected_version
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="add",
          builtin_code=qtyping.BuiltinOperator.ADD,
      ),
      dict(
          testcase_name="batch_matmul",
          builtin_code=qtyping.BuiltinOperator.BATCH_MATMUL,
      ),
  )
  def test_quantize_centered_int2_unsupported_consumer_raises(
      self, builtin_code
  ):
    subgraph = self._model.subgraphs[0]
    opcode_idx = transformation_utils.add_op_code(
        builtin_code, self._model.operatorCodes
    )
    subgraph.operators[4].opcodeIndex = opcode_idx

    with self.assertRaisesRegex(
        ValueError, "BlockwiseQuantization with block_shape is not supported"
    ):
      self._quantize_block_shape_tensor(
          tensor_shape=[4, 8],
          block_size=0,
          scale=np.ones((4, 1), dtype=np.float32),
          quantized_dimension=0,
          zero_point=np.array([[-0.5]], dtype=np.float32),
          num_bits=2,
          tensor_id=7,
          consumers=[4],
      )

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
