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
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile("../../tests/models")
_TFLOpName = qtyping.TFLOperationName
_TensorQuantConfig = qtyping.TensorQuantizationConfig


class NaiveMinMaxQuantizeTest(parameterized.TestCase):
  """Tests for general functions innaive min-max quantize algorithm.

  See naive_min_max_quantize_op_tests for op specific tests.
  """

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

  @parameterized.parameters(
      (qtyping.QuantGranularity.TENSORWISE),
      (qtyping.QuantGranularity.CHANNELWISE),
  )
  def test_init_qsvs(self, granularity):
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                8,
                symmetric=True,
                granularity=granularity,
            ),
        ),
    )

    initial_qsvs = naive_min_max_quantize.init_qsvs(
        op_info,
        self._graph_info,
    )
    self.assertIn("sequential/flatten/Reshape", initial_qsvs)
    input_tensor_qsv = initial_qsvs["sequential/flatten/Reshape"]
    self.assertEmpty(input_tensor_qsv)
    self.assertIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        initial_qsvs,
    )
    output_tensor_qsv = initial_qsvs[
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    ]
    self.assertEmpty(output_tensor_qsv)

    self.assertIn("arith.constant1", initial_qsvs)
    weight_tensor_qsv = initial_qsvs["arith.constant1"]
    if granularity is qtyping.QuantGranularity.CHANNELWISE:
      mins_maxs_shape = (32, 1)
    else:
      mins_maxs_shape = (1, 1)
    self.assertTupleEqual(weight_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertTupleEqual(weight_tensor_qsv["max"].shape, mins_maxs_shape)

    self.assertIn("arith.constant2", initial_qsvs)
    bias_tensor_qsv = initial_qsvs["arith.constant2"]
    if granularity is qtyping.QuantGranularity.CHANNELWISE:
      mins_maxs_shape = (32,)
    else:
      mins_maxs_shape = (1,)
    self.assertTupleEqual(bias_tensor_qsv["min"].shape, mins_maxs_shape)
    self.assertTupleEqual(bias_tensor_qsv["max"].shape, mins_maxs_shape)

    initial_qsvs = naive_min_max_quantize.init_qsvs(
        op_info,
        self._graph_info,
        inputs_to_ignore=[0],
        outputs_to_ignore=[0],
    )
    self.assertNotIn("sequential/flatten/Reshape", initial_qsvs)
    self.assertNotIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        initial_qsvs,
    )

  def test_min_max_calibrate(self):
    # Sample input/output data for the fc op.
    tensor_content_map = {
        "sequential/flatten/Reshape": np.array([[1, 2, 3, 4, 5]]),
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd": np.array(
            [[6, 7, 8, 9, 10]]
        ),
    }
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    fc_op = subgraph0.operators[3]
    # ignore 1(weight), and 2(bias) in inputs.
    op_qsvs = naive_min_max_quantize.min_max_calibrate(
        fc_op,
        self._graph_info,
        tensor_content_map,
        [1, 2],
        [],
    )
    self.assertIn("sequential/flatten/Reshape", op_qsvs)
    input_tensor_qsv = op_qsvs["sequential/flatten/Reshape"]
    self.assertTupleEqual(input_tensor_qsv["min"].shape, (1, 1))
    self.assertEqual(input_tensor_qsv["min"], np.array([[1]]))
    self.assertTupleEqual(input_tensor_qsv["max"].shape, (1, 1))
    self.assertEqual(input_tensor_qsv["max"], np.array([[5]]))
    self.assertIn(
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd",
        op_qsvs,
    )
    output_tensor_qsv = op_qsvs[
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    ]
    self.assertTupleEqual(output_tensor_qsv["min"].shape, (1, 1))
    self.assertEqual(output_tensor_qsv["min"], np.array([[6]]))
    self.assertTupleEqual(output_tensor_qsv["max"].shape, (1, 1))
    self.assertEqual(output_tensor_qsv["max"], np.array([[10]]))
    # weight and bias are excluded.
    self.assertNotIn("arith.constant1", op_qsvs)
    self.assertNotIn("arith.constant2", op_qsvs)

  def test_get_tensor_quant_params_for_blockwise_weight(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    weight_tensor_config = _TensorQuantConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.BLOCKWISE_32,
    )
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=weight_tensor_config,
        ),
    )
    test_data = np.random.uniform(low=-10, high=10, size=(4, 32)).astype(
        np.float32
    )
    quant_params = naive_min_max_quantize.get_tensor_quant_params(
        op_info=op_info,
        tensor_quant_config=weight_tensor_config,
        tensor_content=test_data,
    )
    zp = quant_params.zero_point
    self.assertEqual(zp.shape, (4, 1))
    self.assertTrue(np.array_equal(zp, np.zeros([4, 1])))

    self.assertEqual(quant_params.scale.shape, (4, 1))
    expected_scales = np.max(np.abs(test_data), axis=1, keepdims=True) / 7.0
    expected_scales = (
        expected_scales.astype(ml_dtypes.bfloat16)
        .astype(np.float16)
        .astype(np.float32)
    )
    self.assertTrue(np.allclose(quant_params.scale, expected_scales, atol=1e-5))

    self.assertIsNotNone(quant_params.quantized_data)
    self.assertTupleEqual(
        cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
    )
    self.assertEqual(quant_params.block_size, 32)
    self.assertEqual(quant_params.quantized_dimension, 1)

  def test_get_tensor_quant_params_for_uint8_activation(self):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    activation_tensor_config = _TensorQuantConfig(
        num_bits=8,
        symmetric=False,
        granularity=qtyping.QuantGranularity.TENSORWISE,
        dtype=qtyping.TensorDataType.UINT,
    )
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
        ),
    )
    test_data = np.array([[0.0, 1.0, 2.5, 5.1, 25.5]]).astype(np.float32)
    quant_params = naive_min_max_quantize.get_tensor_quant_params(
        op_info=op_info,
        tensor_quant_config=activation_tensor_config,
        tensor_content=test_data,
    )
    self.assertEqual(quant_params.num_bits, 8)
    self.assertFalse(quant_params.signed)
    self.assertFalse(quant_params.symmetric)
    # Golden values calculation:
    # min = 0.0, max = 25.5
    # scale = (max - min) / (qmax - qmin) = (25.5 - 0.0) / (255 - 0) = 0.1
    # zero_point = round(-min / scale) + qmin = round(0.0) + 0 = 0
    # quantized = round(test_data / scale) + zero_point = [0, 10, 25, 51, 255]
    self.assertTrue(np.allclose(quant_params.scale, 0.1, atol=1e-6))
    self.assertTrue(
        np.array_equal(quant_params.zero_point, np.array([[0]], dtype=np.uint8))
    )
    self.assertIsNotNone(quant_params.quantized_data)
    expected_quantized = np.array([[0, 10, 25, 51, 255]], dtype=np.uint8)
    self.assertTrue(
        np.array_equal(quant_params.quantized_data, expected_quantized)
    )

  def test_get_tensor_quant_params_for_uint8_activation_with_negative_values(
      self,
  ):
    subgraph0 = self._test_model.subgraphs[0]
    subgraph_op_index = 3
    fc_op = subgraph0.operators[subgraph_op_index]
    activation_tensor_config = _TensorQuantConfig(
        num_bits=8,
        symmetric=False,
        granularity=qtyping.QuantGranularity.TENSORWISE,
        dtype=qtyping.TensorDataType.UINT,
    )
    op_info = qtyping.OpInfo(
        op=fc_op,
        op_name=_TFLOpName.FULLY_CONNECTED,
        subgraph_op_index=subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,
                symmetric=True,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
        ),
    )
    # Test data with negative values.
    # min = -10.0, max = 15.5
    test_data = np.array([[-10.0, -5.0, 0.0, 5.0, 15.5]]).astype(np.float32)
    quant_params = naive_min_max_quantize.get_tensor_quant_params(
        op_info=op_info,
        tensor_quant_config=activation_tensor_config,
        tensor_content=test_data,
    )
    self.assertEqual(quant_params.num_bits, 8)
    self.assertFalse(quant_params.signed)
    self.assertFalse(quant_params.symmetric)

    # Golden values calculation:
    # min = -10.0, max = 15.5
    # scale = (max - min) / 255 = 25.5 / 255 = 0.1
    # zero_point = round(-min / scale) = round(10.0 / 0.1) = 100
    # quantized = round(test_data / scale) + zero_point
    #            = [-100, -50, 0, 50, 155] + 100 = [0, 50, 100, 150, 255]
    self.assertTrue(np.allclose(quant_params.scale, 0.1, atol=1e-6))
    self.assertTrue(
        np.array_equal(
            quant_params.zero_point, np.array([[100]], dtype=np.uint8)
        )
    )
    self.assertIsNotNone(quant_params.quantized_data)
    expected_quantized = np.array([[0, 50, 100, 150, 255]], dtype=np.uint8)
    self.assertTrue(
        np.array_equal(quant_params.quantized_data, expected_quantized)
    )

  def test_calibrate_ignores_inf_min_max(self):
    """Tests that calibration ignores infinity values."""
    # Sample input/output data for the fc op.
    input_tensor_name = "sequential/flatten/Reshape"
    output_tensor_name = (
        "sequential/dense/MatMul;sequential/dense/Relu;sequential/dense/BiasAdd"
    )
    bloat16_inf = 3.39e38
    tensor_content_map = {
        input_tensor_name: np.array(
            [[-np.inf, 1.0, 5.0, np.inf, bloat16_inf]], dtype=np.float32
        ),
        output_tensor_name: np.array(
            [[6.0, 7.0, -bloat16_inf, 9.0, np.inf]], dtype=np.float32
        ),
    }
    # Read from Model Explorer.
    subgraph0 = self._test_model.subgraphs[0]
    fc_op = subgraph0.operators[3]
    op_qsvs = naive_min_max_quantize.min_max_calibrate(
        fc_op,
        self._graph_info,
        tensor_content_map,
        inputs_to_ignore=[1, 2],  # Ignore weight and bias.
        outputs_to_ignore=[],
    )

    self.assertIn(input_tensor_name, op_qsvs)
    self.assertEqual(op_qsvs[input_tensor_name]["min"], 1.0)
    self.assertEqual(op_qsvs[input_tensor_name]["max"], 5.0)

    self.assertIn(output_tensor_name, op_qsvs)
    self.assertEqual(op_qsvs[output_tensor_name]["min"], 6.0)
    self.assertEqual(op_qsvs[output_tensor_name]["max"], 9.0)


if __name__ == "__main__":
  absltest.main()
