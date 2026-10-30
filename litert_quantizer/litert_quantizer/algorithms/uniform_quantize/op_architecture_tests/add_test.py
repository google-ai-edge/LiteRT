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


class AddTest(op_test_utils.BaseQuantizeTest):

  def _custom_setup(self, test_model_file: str):
    np.random.seed(666)

    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / test_model_file
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
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
      # num_bits, symmetric
      test_case=(
          (8, False),
          (16, True),
      ),
  )
  def test_materialize_srq_add_succeeds(
      self,
      get_tensor_quant_params_func,
      test_case,
  ):
    self._custom_setup("single_add.tflite")
    activation_num_bits, activation_symmetric = test_case
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["input2"] = "serving_default_input_2:0"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names

    activation_tensor_config = _TensorQuantConfig(
        num_bits=activation_num_bits,
        symmetric=activation_symmetric,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.ADD,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_add,
        get_tensor_quant_params_func,
    )

  @parameterized.product(
      # get_tensor_quant_params_func, activation_num_bits, activation_symmetric
      test_case=(
          (naive_min_max_quantize.get_tensor_quant_params, 8, False),
          (naive_min_max_quantize.get_tensor_quant_params, 16, True),
          (octav.get_tensor_quant_params, 8, True),
          (octav.get_tensor_quant_params, 16, True),
      ),
  )
  def test_materialize_srq_add1_constant_input_succeeds(
      self,
      test_case,
  ):
    """Tests the case where one of the ADD inputs is a constant tensor."""
    get_tensor_quant_params_func, activation_num_bits, activation_symmetric = (
        test_case
    )
    self._custom_setup("single_add1_constant_input.tflite")
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["input2"] = "Const"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names

    activation_tensor_config = _TensorQuantConfig(
        num_bits=activation_num_bits,
        symmetric=activation_symmetric,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.ADD,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_add,
        get_tensor_quant_params_func,
        constant_inputs=[1],
    )


if __name__ == "__main__":
  absltest.main()
