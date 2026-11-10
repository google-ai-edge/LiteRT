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

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_TFLOpName = qtyping.TFLOperationName
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = op_test_utils.OpTestInfo
_DEFAULT_ACTIVATION_QUANT_SETTING = (
    op_test_utils.DEFAULT_ACTIVATION_QUANT_SETTING
)


class BatchMatmulTest(op_test_utils.BaseQuantizeTest):
  """Tests bmm op where both inputs are non-constant tensors."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "bmm.tflite"
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
      # get_tensor_quant_params_func, num_bits_activation, symmetric_activation
      test_case=(
          (naive_min_max_quantize.get_tensor_quant_params, 8, False),
          (naive_min_max_quantize.get_tensor_quant_params, 16, True),
          (octav.get_tensor_quant_params, 8, True),
          (octav.get_tensor_quant_params, 16, True),
      )
  )
  def test_batch_matmul_adjy_false_srq_succeeds(self, test_case):
    get_tensor_quant_params_func, num_bits_activation, symmetric_activation = (
        test_case
    )
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["input2"] = "serving_default_input_2:0"
    op_tensor_names["output"] = "model/tf.linalg.matmul/MatMul"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 2

    activation_tensor_config = _TensorQuantConfig(
        num_bits=num_bits_activation,
        symmetric=symmetric_activation,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_batch_matmul,
        get_tensor_quant_params_func,
    )

  @parameterized.product(
      # get_tensor_quant_params_func, num_bits_activation, symmetric_activation
      test_case=(
          (naive_min_max_quantize.get_tensor_quant_params, 8, False),
          (naive_min_max_quantize.get_tensor_quant_params, 16, True),
          (octav.get_tensor_quant_params, 8, True),
          (octav.get_tensor_quant_params, 16, True),
      )
  )
  def test_batch_matmul_adjy_true_srq_succeeds(self, test_case):
    get_tensor_quant_params_func, num_bits_activation, symmetric_activation = (
        test_case
    )
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 1
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["input"] = "model/tf.linalg.matmul/MatMul"
    op_tensor_names["input2"] = "serving_default_input_2:0"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 1

    activation_tensor_config = _TensorQuantConfig(
        num_bits=num_bits_activation,
        symmetric=symmetric_activation,
        granularity=qtyping.QuantGranularity.TENSORWISE,
    )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_batch_matmul,
        get_tensor_quant_params_func,
    )


class BatchMatmulConstantInputTest(op_test_utils.BaseQuantizeTest):
  """Tests bmm op where one the inputs is a constant tensor."""

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "bmm_constant_input.tflite"
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
      num_bits_weight=(4, 8),
      # get_tensor_quant_params_func, symmetric_weight
      algos=(
          (naive_min_max_quantize.get_tensor_quant_params, True),
          (naive_min_max_quantize.get_tensor_quant_params, False),
          (octav.get_tensor_quant_params, True),
      ),
      test_case=(
          # Tuple holds compute precision, whether to use srq and whether
          # to use explicit dequantize.
          (_ComputePrecision.INTEGER, True, False),
          (_ComputePrecision.INTEGER, False, False),
          (_ComputePrecision.FLOAT, False, True),
      ),
  )
  def test_batch_matmul_adjy_false_succeeds(
      self,
      num_bits_weight,
      algos,
      test_case,
  ):
    get_tensor_quant_params_func, symmetric_weight = algos
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["weight"] = "model/tf.linalg.matmul/MatMul/b"
    op_tensor_names["input"] = "serving_default_input_1:0"
    op_tensor_names["output"] = "model/tf.linalg.matmul/MatMul"
    compute_precision, is_srq, explicit_dequantize = test_case
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 2

    activation_tensor_config = None
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_batch_matmul,
        get_tensor_quant_params_func,
    )

  @parameterized.product(
      num_bits_weight=(4, 8),
      # get_tensor_quant_params_func, symmetric_weight
      algos=(
          (naive_min_max_quantize.get_tensor_quant_params, True),
          (naive_min_max_quantize.get_tensor_quant_params, False),
          (octav.get_tensor_quant_params, True),
      ),
      test_case=(
          # Tuple holds compute precision, whether to use srq and whether
          # to use explicit dequantize.
          (_ComputePrecision.INTEGER, True, False),
          (_ComputePrecision.INTEGER, False, False),
          (_ComputePrecision.FLOAT, False, True),
      ),
  )
  def test_batch_matmul_adjy_true_succeeds(
      self,
      num_bits_weight,
      algos,
      test_case,
  ):
    get_tensor_quant_params_func, symmetric_weight = algos
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 1
    op = subgraph0.operators[subgraph_op_id]
    op_tensor_names = {}
    op_tensor_names["weight"] = "arith.constant"
    op_tensor_names["input"] = "model/tf.linalg.matmul/MatMul"
    op_tensor_names["output"] = "PartitionedCall:0"
    compute_precision, is_srq, explicit_dequantize = test_case
    self._op_test_info.op_tensor_names = op_tensor_names
    self._op_test_info.quantized_dimension = 1

    activation_tensor_config = None
    # Check if SRQ.
    if compute_precision == _ComputePrecision.INTEGER and is_srq:
      activation_tensor_config = _DEFAULT_ACTIVATION_QUANT_SETTING

    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.BATCH_MATMUL,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits_weight,
                symmetric=symmetric_weight,
                granularity=qtyping.QuantGranularity.TENSORWISE,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_batch_matmul,
        get_tensor_quant_params_func,
    )


if __name__ == "__main__":
  absltest.main()
