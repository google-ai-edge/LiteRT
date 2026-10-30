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

from collections.abc import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import gptq
from litert_quantizer.utils import tfl_flatbuffer_utils


def _create_tensor(
    name: str, shape: Sequence[int], quantization=None
) -> qtyping.TensorT:
  tensor = qtyping.TensorT()
  tensor.name = name.encode("utf-8")
  tensor.shape = list(shape)
  tensor.quantization = quantization
  return tensor


def _create_op(
    inputs: Sequence[int], outputs: Sequence[int]
) -> qtyping.OperatorT:
  op = qtyping.OperatorT()
  op.inputs = list(inputs)
  op.outputs = list(outputs)
  return op


class GptqTest(parameterized.TestCase):

  def test_calibrate_computes_correct_qsvs(self):
    input_tensor = _create_tensor("input", shape=(1, 2, 3))
    output_tensor = _create_tensor("output", shape=(1, 2, 3))
    op = _create_op(inputs=[0], outputs=[1])

    graph_info = qtyping.GraphInfo(
        subgraph_tensors=[input_tensor, output_tensor], buffers=[]
    )

    val = 1e39
    tensor_content_map = {
        "input": np.array([[[1.0, 2.0, -val], [4.0, 5.0, 6.0]]]),
        "output": np.array([[[7.0, 8.0, val], [10.0, 11.0, 12.0]]]),
    }

    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            side_effect=["input", "output"],
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=None,
            autospec=True,
            spec_set=True,
        )
    )
    qsvs = gptq.calibrate(op, graph_info, tensor_content_map)

    self.assertIn("input", qsvs)
    self.assertAlmostEqual(qsvs["input"]["min"], 1.0)
    self.assertAlmostEqual(qsvs["input"]["max"], 6.0)
    self.assertIn("hessian", qsvs["input"])
    self.assertEqual(qsvs["input"]["num_samples"], 1)
    # Expected hessian: 2 * X.T @ X / num_samples where
    # X = [[1,2,3],[4,5,6]]
    # X.T @ X = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
    # hessian = 2 * [[17, 22, 27], [22, 29, 36], [27, 36, 45]] / 1
    expected_hessian_input = 2.0 * np.array([
        [17.0, 22.0, 24.0 - val],
        [22.0, 29.0, 30.0 - 2 * val],
        [24.0 - val, 30.0 - 2 * val, 36.0 + val**2],
    ])
    np.testing.assert_allclose(
        qsvs["input"]["hessian"], expected_hessian_input
    )
    self.assertIn("output", qsvs)
    self.assertAlmostEqual(qsvs["output"]["min"], 7.0)
    self.assertAlmostEqual(qsvs["output"]["max"], 12.0)
    self.assertIn("hessian", qsvs["output"])
    self.assertEqual(qsvs["output"]["num_samples"], 1)
    expected_hessian_output = 2.0 * np.array([
        [149.0, 166.0, 120.0 + 7 * val],
        [166.0, 185.0, 132.0 + 8 * val],
        [120.0 + 7 * val, 132.0 + 8 * val, 144.0 + val**2],
    ])
    np.testing.assert_allclose(
        qsvs["output"]["hessian"], expected_hessian_output
    )

  def test_calibrate_ignores_outputs_if_specified(self):
    input_tensor = _create_tensor("input", shape=(1, 1, 1))
    output_tensor = _create_tensor("output", shape=(1, 1, 1))
    op = _create_op(inputs=[0], outputs=[1])
    graph_info = qtyping.GraphInfo(
        subgraph_tensors=[input_tensor, output_tensor], buffers=[]
    )
    tensor_content_map = {
        "input": np.array([[[1.0]]]),
        "output": np.array([[[2.0]]]),
    }

    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            side_effect=["input"],
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=None,
            autospec=True,
            spec_set=True,
        )
    )
    qsvs = gptq.calibrate(
        op,
        graph_info,
        tensor_content_map,
        outputs_to_ignore=[0],
    )
    self.assertIn("input", qsvs)
    self.assertNotIn("output", qsvs)

  def test_calibrate_ignores_already_quantized_inputs(self):
    input1 = _create_tensor("in1", shape=(1, 1, 1))
    quant_params = qtyping.QuantizationParametersT()
    quant_params.scale = [0.1]
    quantized_input2 = _create_tensor(
        "in2_quant", shape=(1, 1, 1), quantization=quant_params
    )
    output = _create_tensor("out", shape=(1, 1, 1))
    op = _create_op(inputs=[0, 1], outputs=[2])
    graph_info = qtyping.GraphInfo(
        subgraph_tensors=[
            input1,
            quantized_input2,
            output,
        ],
        buffers=[],
    )
    tensor_content_map = {
        "in1": np.array([[[1.0]]]),
        "in2_quant": np.array([[[5.0]]]),
        "out": np.array([[[2.0]]]),
    }

    def get_tensor_name_side_effect(tensor):
      if tensor == input1:
        return "in1"
      if tensor == quantized_input2:
        return "in2_quant"
      if tensor == output:
        return "out"
      return "unknown"

    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_name",
            side_effect=get_tensor_name_side_effect,
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            tfl_flatbuffer_utils,
            "get_tensor_data",
            return_value=None,
            autospec=True,
            spec_set=True,
        )
    )
    qsvs = gptq.calibrate(
        op,
        graph_info,
        tensor_content_map,
    )
    self.assertIn("in1", qsvs)
    self.assertNotIn("in2_quant", qsvs)
    self.assertIn("out", qsvs)

  def test_get_tensor_quant_params_applies_gptq_correctly(self):
    op_info = qtyping.OpInfo(
        op=qtyping.OperatorT(),
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_quant_config=qtyping.OpQuantizationConfig(),
        subgraph_op_index=-1,
    )
    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    tensor_content = np.array(
        [[1.1, 2.1914, 0.6], [-1.1, 0.1, -0.6]], dtype=np.float32
    )
    tensor_qsv = {"min": np.array([[-1.1]]), "max": np.array([[2.2]])}
    activation_tensor_qsv = {
        "hessian": np.array(
            [[15.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]],
            dtype=np.float32,
        ),
        "num_samples": 1,
    }
    tensor_qsv["activation_tensor_qsv"] = activation_tensor_qsv
    quant_params = gptq.get_tensor_quant_params(
        op_info,
        tensor_quant_config,
        tensor_content,
        tensor_qsv,
    )
    self.assertIsNotNone(quant_params.quantized_data)
    self.assertEqual(quant_params.quantized_data.shape, tensor_content.shape)  # pytype: disable=attribute-error
    self.assertEqual(quant_params.quantized_data.dtype, np.int8)  # pytype: disable=attribute-error
    # Check if scales and zp are correctly calculated.
    np.testing.assert_allclose(quant_params.scale, np.array([[2.2 / 127]]))
    np.testing.assert_allclose(quant_params.zero_point, np.array([[0]]))
    # The weight update caused by GPTQ changes quantization result for w[0,1].
    expected_quantized_data = np.array(
        [[64, 126, 35], [-64, 6, -35]], dtype=np.int8
    )
    np.testing.assert_allclose(
        quant_params.quantized_data, expected_quantized_data
    )

  def test_get_tensor_quant_params_without_tensor_qsv(self):
    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=qtyping.OperatorT(),
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_quant_config
        ),
        subgraph_op_index=-1,
    )
    tensor_content = np.array(
        [[1.1, 2.1914, 0.6], [-1.1, 0.1, -0.6]], dtype=np.float32
    )
    activation_tensor_qsv = {
        "hessian": np.array(
            [[15.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.0]],
            dtype=np.float32,
        ),
        "num_samples": 1,
    }
    tensor_qsv = {"activation_tensor_qsv": activation_tensor_qsv}
    quant_params = gptq.get_tensor_quant_params(
        op_info,
        tensor_quant_config,
        tensor_content,
        tensor_qsv=tensor_qsv,
    )
    self.assertIsNotNone(quant_params.quantized_data)
    self.assertEqual(quant_params.quantized_data.shape, tensor_content.shape)  # pytype: disable=attribute-error
    self.assertEqual(quant_params.quantized_data.dtype, np.int8)  # pytype: disable=attribute-error
    np.testing.assert_allclose(quant_params.scale, np.array([[2.1914 / 127]]))
    np.testing.assert_allclose(quant_params.zero_point, np.array([[0]]))
    expected_quantized_data = np.array(
        [[64, 127, 35], [-64, 6, -35]], dtype=np.int8
    )
    np.testing.assert_allclose(
        quant_params.quantized_data, expected_quantized_data
    )

  def test_get_tensor_quant_params_without_tensor_content(
      self,
  ):
    op_info = qtyping.OpInfo(
        op=qtyping.OperatorT(),
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_quant_config=qtyping.OpQuantizationConfig(),
        subgraph_op_index=-1,
    )
    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    tensor_qsv = {"min": np.array([[-1.1]]), "max": np.array([[2.2]])}
    quant_params = gptq.get_tensor_quant_params(
        op_info,
        tensor_quant_config,
        tensor_content=None,
        tensor_qsv=tensor_qsv,
    )
    self.assertIsNone(quant_params.quantized_data)
    np.testing.assert_allclose(quant_params.scale, np.array([[2.2 / 127]]))
    np.testing.assert_allclose(quant_params.zero_point, np.array([[0]]))

  @mock.patch.object(
      gptq.uniform_quantize_tensor,
      "extract_block_size_from_granularity",
      return_value=2,
  )
  def test_gptq_blockwise_selects_correct_scale_per_column(
      self, mock_extract_block_size
  ):
    del mock_extract_block_size  # Unused.
    op_info = qtyping.OpInfo(
        op=qtyping.OperatorT(),
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        op_quant_config=qtyping.OpQuantizationConfig(),
        subgraph_op_index=-1,
    )
    tensor_quant_config = qtyping.TensorQuantizationConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
    )
    tensor_content = np.array(
        [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
        ],
        dtype=np.float32,
    ) / 127
    # scale for c0b0=1/127, c0b1=2/127, c1b0=3/127, c1b1=4/127
    tensor_qsv = {
        "min": np.array([[-1.0, -2.0], [-3.0, -4.0]]),
        "max": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    activation_tensor_qsv = {
        "hessian": np.eye(4, dtype=np.float32),
        "num_samples": 1,
    }
    tensor_qsv["activation_tensor_qsv"] = activation_tensor_qsv
    quant_params = gptq.get_tensor_quant_params(
        op_info,
        tensor_quant_config,
        tensor_content,
        tensor_qsv,
    )

    self.assertIsNotNone(quant_params.quantized_data)
    self.assertEqual(
        quant_params.quantized_data.shape, tensor_content.shape  # pytype: disable=attribute-error
    )
    self.assertEqual(quant_params.quantized_data.dtype, np.int8)  # pytype: disable=attribute-error
    # Check if scales and zp are correctly calculated.
    expected_scale = np.array(
        [[1 / 127, 2 / 127], [3 / 127, 4 / 127]], dtype=np.float32
    )
    expected_scale = (
        expected_scale.astype(ml_dtypes.bfloat16)
        .astype(np.float16)
        .astype(np.float32)
    )
    np.testing.assert_allclose(
        quant_params.scale,
        expected_scale,
    )
    np.testing.assert_allclose(
        quant_params.zero_point, np.array([[0, 0], [0, 0]])
    )
    # With identity hessian, gptq should not affect quantization result
    # across columns.
    # If blockwise scale selection is correct, values should be quantized
    # according to rounded scales.
    expected_quantized_data = np.full_like(tensor_content, 10, dtype=np.int8)
    np.testing.assert_allclose(
        quant_params.quantized_data, expected_quantized_data
    )


if __name__ == "__main__":
  absltest.main()
