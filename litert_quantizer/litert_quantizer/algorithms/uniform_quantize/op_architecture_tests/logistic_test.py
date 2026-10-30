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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer.algorithms.uniform_quantize import common_quantize
from litert_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from litert_quantizer.algorithms.uniform_quantize import octav
from litert_quantizer.algorithms.uniform_quantize.op_architecture_tests import test_utils as op_test_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = op_test_utils.OpTestInfo

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    op_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)
_DEFAULT_WEIGHT_QUANT_SETTING = op_test_utils.DEFAULT_WEIGHT_QUANT_SETTING


class LogisticTest(op_test_utils.BaseQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "single_fc_bias_logistic.tflite"
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[0]]), np.array([[1]])),
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
    )

  @parameterized.product(
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
      activation_tensor_config=(
          (_DEFAULT_ACTIVATION_QUANT_SETTING),
          (
              _TensorQuantConfig(
                  num_bits=16,
                  symmetric=True,
                  granularity=qtyping.QuantGranularity.TENSORWISE,
              )
          ),
      ),
  )
  def test_materialize_logistics_succeeds(
      self, get_tensor_quant_params_func, activation_tensor_config
  ):
    op_quant_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=activation_tensor_config,
        weight_tensor_config=_DEFAULT_WEIGHT_QUANT_SETTING,
        compute_precision=_ComputePrecision.INTEGER,  # SRQ.
    )
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 1
    op = subgraph0.operators[subgraph_op_id]
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.LOGISTIC,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=op_quant_config,
    )

    # Test settings.
    op_tensor_names = {}
    op_tensor_names["input"] = "model/dense/MatMul2"
    op_tensor_names["output"] = "StatefulPartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_softmax_and_logistic,
        get_tensor_quant_params_func,
    )


if __name__ == "__main__":
  absltest.main()
