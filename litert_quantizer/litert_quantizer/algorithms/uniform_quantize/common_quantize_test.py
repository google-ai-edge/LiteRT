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

import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import default_policy
from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class CommonQuantizeTest(parameterized.TestCase):
  """Tests for general quantize functions."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "conv_fc_mnist.tflite"
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}

  def test_check_op_quantization_config_with_negative_min_weight_elements_raises_error(
      self,
  ):
    op_quant_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=_TensorQuantConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,  # DRQ.
        min_weight_elements=-1,
    )
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        lambda err: "min_weight_elements must be non-negative" in str(err),
    ):
      common_quantize.check_op_quantization_config(
          _TFLOpName.FULLY_CONNECTED,
          op_quant_config,
          default_policy.DEFAULT_CONFIG_CHECK_POLICY,
      )

  def test_get_activation_min_max_float(self):
    tensor_content = np.array(
        [-np.inf, np.inf, 1.0, 2.0, -10.0, 10.0], dtype=np.float32
    )
    qsv = common_quantize.get_activation_min_max(
        tensor_content,
        valid_float_range_min=-1000.0,
        valid_float_range_max=1000.0,
    )
    self.assertEqual(qsv["min"].item(), -10.0)
    self.assertEqual(qsv["max"].item(), 10.0)

  def test_get_activation_min_max_int(self):
    tensor_content = np.array([1, 2, -10, 10], dtype=np.int32)
    qsv = common_quantize.get_activation_min_max(
        tensor_content,
    )
    self.assertEqual(qsv["min"].item(), -10)
    self.assertEqual(qsv["max"].item(), 10)

  def test_collect_activation_tensor_statistics_returns_none_for_constant(self):
    with mock.patch.object(
        tfl_flatbuffer_utils,
        "get_tensor_data",
        return_value=np.array([1]),
        autospec=True,
        spec_set=True,
    ):
      res = common_quantize.collect_activation_tensor_statistics(
          0, self._graph_info, {}
      )
      self.assertIsNone(res)

  def test_collect_activation_tensor_statistics_returns_qsv(self):
    # Mocking to return None (meaning it's an activation)
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=None,
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            return_value="my_tensor",
            autospec=True,
            spec_set=True,
        )
    )
    tensor_content_map = {
        "my_tensor": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }
    res = common_quantize.collect_activation_tensor_statistics(
        0, self._graph_info, tensor_content_map
    )
    self.assertIsNotNone(res)
    tensor_name, _, qsv = res
    self.assertEqual(tensor_name, "my_tensor")
    self.assertEqual(qsv["min"].item(), 1.0)
    self.assertEqual(qsv["max"].item(), 3.0)
    self.assertEqual(qsv["num_samples"].item(), 3)

  def test_get_tensor_indices_requiring_calibration(self):
    # Create a mock tfl_op with inputs and outputs
    mock_op = mock.create_autospec(
        qtyping.OperatorT(), instance=True, spec_set=True
    )
    mock_op.inputs = [0, 1, 2, -1]  # -1 is optional tensor
    mock_op.outputs = [3, 4, -1]

    with mock.patch.object(
        common_quantize, "check_if_quantized", return_value=False
    ):
      # Ignore input at pos 1, output at pos 0 (which is tensor 3)
      res = common_quantize.get_tensor_indices_requiring_calibration(
          mock_op, self._graph_info, inputs_to_ignore=[1], outputs_to_ignore=[0]
      )
      # Should include inputs 0 and 2. Should include output 4.
      self.assertEqual(res, [0, 2, 4])

  def test_check_if_quantized_recognizes_blockwise_quantization_details(self):
    unquantized = qtyping.TensorT()
    self.assertFalse(common_quantize.check_if_quantized(unquantized))

    empty_quant = qtyping.TensorT()
    empty_quant.quantization = qtyping.QuantizationParametersT()
    self.assertFalse(common_quantize.check_if_quantized(empty_quant))

    scale_quant = qtyping.TensorT()
    scale_quant.quantization = qtyping.QuantizationParametersT()
    scale_quant.quantization.scale = [0.5]
    self.assertTrue(common_quantize.check_if_quantized(scale_quant))

    blockwise_quant = qtyping.TensorT()
    blockwise_quant.quantization = qtyping.QuantizationParametersT()
    blockwise_quant.quantization.detailsType = (
        qtyping.QuantizationDetails.BlockwiseQuantization
    )
    blockwise_quant.quantization.details = qtyping.BlockwiseQuantizationT()
    self.assertTrue(common_quantize.check_if_quantized(blockwise_quant))

  def _build_blockwise_fc_srq_fixture(
      self,
      scale_data: np.ndarray | None,
      scale_type: qtyping.TensorType = qtyping.TensorType.FLOAT32,
      zp_data: np.ndarray | None = None,
  ) -> tuple[qtyping.OpInfo, qtyping.GraphInfo]:
    buffers = [qtyping.BufferT()]

    def add_buffer(arr: np.ndarray) -> int:
      buf = qtyping.BufferT()
      buf.data = np.frombuffer(arr.tobytes(), dtype=np.uint8)
      buffers.append(buf)
      return len(buffers) - 1

    input_tensor = qtyping.TensorT()
    input_tensor.name = b"input"
    input_tensor.shape = [1, 4]
    input_tensor.type = qtyping.TensorType.FLOAT32
    input_tensor.buffer = 0

    weight_tensor = qtyping.TensorT()
    weight_tensor.name = b"weight"
    weight_tensor.shape = [2, 4]
    weight_tensor.type = qtyping.TensorType.INT2
    weight_tensor.buffer = add_buffer(np.zeros(2, dtype=np.uint8))
    weight_tensor.quantization = qtyping.QuantizationParametersT()
    weight_tensor.quantization.detailsType = (
        qtyping.QuantizationDetails.BlockwiseQuantization
    )

    bias_tensor = qtyping.TensorT()
    bias_tensor.name = b"bias"
    bias_tensor.shape = [2]
    bias_tensor.type = qtyping.TensorType.FLOAT32
    bias_tensor.buffer = add_buffer(np.array([1.0, -2.0], dtype=np.float32))

    output_tensor = qtyping.TensorT()
    output_tensor.name = b"output"
    output_tensor.shape = [1, 2]
    output_tensor.type = qtyping.TensorType.FLOAT32
    output_tensor.buffer = 0

    scales_tensor = qtyping.TensorT()
    scales_tensor.name = b"weight_scales"
    scales_tensor.shape = (
        list(scale_data.shape) if scale_data is not None else [2, 1]
    )
    scales_tensor.type = scale_type
    scales_tensor.buffer = (
        add_buffer(scale_data) if scale_data is not None else 0
    )

    tensors = [
        input_tensor,
        weight_tensor,
        bias_tensor,
        output_tensor,
        scales_tensor,
    ]
    details = qtyping.BlockwiseQuantizationT()
    details.scales = 4
    if zp_data is not None:
      zp_tensor = qtyping.TensorT()
      zp_tensor.name = b"weight_zp"
      zp_tensor.shape = list(zp_data.shape)
      zp_tensor.type = qtyping.TensorType.FLOAT32
      zp_tensor.buffer = add_buffer(zp_data)
      tensors.append(zp_tensor)
      details.zeroPoints = 5
    else:
      details.zeroPoints = -1
    weight_tensor.quantization.details = details

    fc_op = qtyping.OperatorT()
    fc_op.inputs = [0, 1, 2]
    fc_op.outputs = [3]
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=0,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=_TensorQuantConfig(
                num_bits=8, symmetric=False
            ),
            weight_tensor_config=_TensorQuantConfig(
                num_bits=2,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
    )
    graph_info = qtyping.GraphInfo(subgraph_tensors=tensors, buffers=buffers)
    return op_info, graph_info

  @parameterized.named_parameters(
      dict(
          testcase_name="tensorwise_float16_scale",
          scale_data=np.array([[0.5]], dtype=np.float16),
          scale_type=qtyping.TensorType.FLOAT16,
          zp_data=None,
          input_scale=np.array([0.25], dtype=np.float16),
          expected_bias_scale=[0.125],
          expected_quantized_dim=None,
          expected_quantized_bias=[8, -16],
      ),
      dict(
          testcase_name="channelwise_centered_int2",
          scale_data=np.array([[0.5], [2.0]], dtype=np.float32),
          scale_type=qtyping.TensorType.FLOAT32,
          zp_data=np.array([[-0.5]], dtype=np.float32),
          input_scale=np.array([0.25], dtype=np.float32),
          expected_bias_scale=[0.125, 0.5],
          expected_quantized_dim=0,
          expected_quantized_bias=[8, -4],
      ),
  )
  def test_materialize_fc_conv_bias_with_blockwise_weight_succeeds(
      self,
      scale_data,
      scale_type,
      zp_data,
      input_scale,
      expected_bias_scale,
      expected_quantized_dim,
      expected_quantized_bias,
  ):
    op_info, graph_info = self._build_blockwise_fc_srq_fixture(
        scale_data=scale_data,
        scale_type=scale_type,
        zp_data=zp_data,
    )

    def get_activation_quant_params(*unused_args):
      return qtyping.UniformQuantParams(
          num_bits=8,
          quantized_dimension=None,
          scale=input_scale,
          zero_point=np.array([0], dtype=np.int32),
          symmetric=False,
      )

    op_tensor_params = common_quantize.materialize_fc_conv(
        get_activation_quant_params,
        op_info,
        graph_info,
        tensor_name_to_qsv={},
        tensor_quant_params_cache=mock.MagicMock(),
    )
    bias_consumers = op_tensor_params[2].consumers
    self.assertIsNotNone(bias_consumers)
    bias_params = bias_consumers[0].parameters
    self.assertIsInstance(bias_params, qtyping.UniformQuantParams)
    self.assertEqual(bias_params.scale.dtype, np.float32)
    np.testing.assert_allclose(
        bias_params.scale, np.array(expected_bias_scale, dtype=np.float32)
    )
    self.assertEqual(bias_params.quantized_dimension, expected_quantized_dim)
    np.testing.assert_array_equal(
        bias_params.quantized_data,
        np.array(expected_quantized_bias, dtype=np.int32),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="subchannel_blockwise",
          scale_data=np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
          zp_data=None,
          expected_error=(
              "SRQ bias materialization requires per-tensor or per-channel"
              " weight scales"
          ),
      ),
      dict(
          testcase_name="missing_scale_buffer",
          scale_data=None,
          zp_data=None,
          expected_error="BlockwiseQuantization scale tensor has no buffer",
      ),
      dict(
          testcase_name="mismatched_zp_shape",
          scale_data=np.array([[0.5], [2.0]], dtype=np.float32),
          zp_data=np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32),
          expected_error=(
              "BlockwiseQuantization zero_points shape .* does not match"
              " weight scale shape"
          ),
      ),
  )
  def test_materialize_fc_conv_bias_with_invalid_blockwise_weight_raises(
      self, scale_data, zp_data, expected_error
  ):
    op_info, graph_info = self._build_blockwise_fc_srq_fixture(
        scale_data=scale_data,
        zp_data=zp_data,
    )

    def get_activation_quant_params(*unused_args):
      return qtyping.UniformQuantParams(
          num_bits=8,
          quantized_dimension=None,
          scale=np.array([0.25], dtype=np.float32),
          zero_point=np.array([0], dtype=np.int32),
          symmetric=False,
      )

    with self.assertRaisesRegex(ValueError, expected_error):
      common_quantize.materialize_fc_conv(
          get_activation_quant_params,
          op_info,
          graph_info,
          tensor_name_to_qsv={},
          tensor_quant_params_cache=mock.MagicMock(),
      )


if __name__ == "__main__":
  absltest.main()
