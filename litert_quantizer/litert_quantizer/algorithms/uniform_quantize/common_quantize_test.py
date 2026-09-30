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

  def test_reshape_data_for_blockwise_raises_error_when_quantized_dim_not_divisible_by_block_size(
      self,
  ):
    tensor_data = np.ones((24, 128), dtype=np.float32)
    block_size = 256
    quantized_dim = 1
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        lambda err: (
            "Tensor quantization dimension must be divisible by block"
            " size for blockwise quantization."
        )
        in str(err),
    ):
      common_quantize._reshape_data_for_blockwise(
          tensor_data, quantized_dim, block_size
      )

  def test_reshape_data_for_blockwise_returns_correct_values(self):
    tensor_data = np.ones((24, 128), dtype=np.float32)
    block_size = 32
    quantized_dim = 1
    new_tensor_data, reduce_dim = common_quantize._reshape_data_for_blockwise(
        tensor_data, quantized_dim, block_size
    )
    self.assertEqual(new_tensor_data.shape, (24, 4, 32))
    self.assertEqual(reduce_dim, 2)

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


if __name__ == "__main__":
  absltest.main()
