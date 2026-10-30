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
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import mse
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils


class MseQuantizeTest(parameterized.TestCase):
  """Tests for general functions for MSE."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(test_utils.get_path_to_datafile("../../tests/models"))
        / "conv_fc_mnist.tflite",
    )
    self._test_model = tfl_flatbuffer_utils.read_model(self._test_model_path)
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._test_model.subgraphs[0].tensors,
        buffers=self._test_model.buffers,
    )
    self._tensor_name_to_qsv = {}
    subgraph0 = self._test_model.subgraphs[0]
    self._subgraph_op_index = 3
    self._fc_op = subgraph0.operators[self._subgraph_op_index]
    self._fc_op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=self._subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=None,
        ),
    )

  def test_get_tensor_quant_params_raises_error_with_unsupported_symmetry(self):
    err_msg = "Unsupported symmetry"
    test_data = np.array([[-7, 7], [4, -4], [4, -4], [7, 7]])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: err_msg in str(err)
    ):
      _ = mse.get_tensor_quant_params(
          op_info=self._fc_op_info,
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              symmetric=False,
              granularity=qtyping.QuantGranularity.CHANNELWISE,
          ),
          tensor_content=test_data,
      )

  def test_get_tensor_quant_params_raises_error_with_unsupported_granularity(
      self,
  ):
    err_msg = "Blockwise quantization is not supported"
    test_data = np.array([[-7, 7], [4, -4], [4, -4], [7, 7]])
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda err: err_msg in str(err)
    ):
      _ = mse.get_tensor_quant_params(
          op_info=self._fc_op_info,
          tensor_quant_config=qtyping.TensorQuantizationConfig(
              num_bits=4,
              symmetric=True,
              granularity=qtyping.QuantGranularity.BLOCKWISE_32,
          ),
          tensor_content=test_data,
      )

  def test_get_tensor_quant_params_succeeds_with_qsv(self):
    # Fall back to naive_min_max_quantize.py for non-weight tensors.
    tensor_quant_params = mse.get_tensor_quant_params(
        op_info=self._fc_op_info,
        tensor_quant_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            granularity=qtyping.QuantGranularity.TENSORWISE,
        ),
        tensor_qsv={
            "min": np.array([-1]),
            "max": np.array([1]),
        },
    )

    self.assertIsNone(tensor_quant_params.quantized_dimension)
    scale = tensor_quant_params.scale
    self.assertEqual(scale.shape, (1,))
    self.assertSequenceAlmostEqual(scale.flatten(), [1 / 127])

    # Zero point should be zero for symmetric quantization.
    zp = tensor_quant_params.zero_point
    self.assertEqual(np.sum(zp), 0)
    self.assertEqual(zp.shape, (1,))

  def test_get_tensor_quant_params_succeeds_with_tensorwise_granularity(self):
    test_data = np.array([
        [-1e5, 25, -50, 75, -100, 125],
        [25, -30, 50, -75, 1e5, -125],
        [50, -60, 70, -80, 90, -100],
    ])
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    fc_op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=self._subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_config,
        ),
    )
    quant_params = mse.get_tensor_quant_params(
        op_info=fc_op_info,
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )

    with self.subTest(name="CheckQuantParamsShapes"):
      self.assertEqual(quant_params.zero_point.shape, (1, 1))
      self.assertEqual(quant_params.scale.shape, (1, 1))
      self.assertIsNone(quant_params.quantized_dimension)
      self.assertIsNotNone(quant_params.quantized_data)
      self.assertTupleEqual(
          cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
      )

    with self.subTest(name="CheckQuantParamsValues"):
      self.assertTrue(np.all(quant_params.zero_point == 0))

  def test_get_tensor_quant_params_succeeds_with_channelwise_granularity(self):
    # Test that the call generates quant params that are appropriately shaped,
    # have some clipping, and correct config values without checking the
    # actual values numerically.
    test_data = np.array([
        [-1e5, 25, -50, 75, -100, 125],
        [25, -30, 50, -75, 1e5, -125],
        [50, -60, 70, -80, 90, -100],
    ])
    tensor_config = qtyping.TensorQuantizationConfig(
        num_bits=4,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
    )
    fc_op_info = qtyping.OpInfo(
        op=self._fc_op,
        op_name=qtyping.TFLOperationName.FULLY_CONNECTED,
        subgraph_op_index=self._subgraph_op_index,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=tensor_config,
        ),
    )
    quant_params = mse.get_tensor_quant_params(
        op_info=fc_op_info,
        tensor_quant_config=tensor_config,
        tensor_content=test_data,
    )

    with self.subTest(name="CheckQuantParamsShapes"):
      self.assertEqual(quant_params.zero_point.shape, (test_data.shape[0], 1))
      self.assertEqual(quant_params.scale.shape, (test_data.shape[0], 1))
      self.assertIsNotNone(quant_params.quantized_data)
      self.assertTupleEqual(
          cast(np.ndarray, quant_params.quantized_data).shape, test_data.shape
      )

    with self.subTest(name="CheckQuantParamsValues"):
      self.assertTrue(np.all(quant_params.zero_point == 0))
      self.assertEqual(quant_params.quantized_dimension, 0)


if __name__ == "__main__":
  absltest.main()
