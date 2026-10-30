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


class Conv2DTransposeTest(op_test_utils.BaseQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH)
        / "single_conv2d_transpose_bias.tflite"
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
        quantized_dimension=0,
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
    )
    self._set_op_tensor_names()

  def _set_op_tensor_names(self):
    op_tensor_names = {}
    op_tensor_names["weight"] = (
        "sequential_5/conv2d_transpose_3/conv2d_transpose"
    )
    op_tensor_names["bias"] = (
        "sequential_5/conv2d_transpose_3/BiasAdd;sequential_5/conv2d_transpose_3/conv2d_transpose;sequential_5/conv2d_transpose_3/BiasAdd/ReadVariableOp"
    )
    op_tensor_names["input"] = "serving_default_input_6:0"
    op_tensor_names["ignored_input"] = "sequential_5/conv2d_transpose_3/stack"
    op_tensor_names["output"] = "StatefulPartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names

  @parameterized.product(
      # get_tensor_quant_params_func, symmetric_weight
      algos=(
          (naive_min_max_quantize.get_tensor_quant_params, True),
          (naive_min_max_quantize.get_tensor_quant_params, False),
          (octav.get_tensor_quant_params, True),
      ),
      granularity=(
          qtyping.QuantGranularity.CHANNELWISE,
          qtyping.QuantGranularity.TENSORWISE,
      ),
      test_case=(
          # Tuple holds compute precision and whether to use srq and explicit
          # dequantize.
          (_ComputePrecision.FLOAT, True),
          (_ComputePrecision.INTEGER, False),
      ),
  )
  def test_materialize_weight_only_drq_conv2d_transpose_succeeds(
      self,
      algos,
      granularity,
      test_case,
  ):
    get_quant_params_func, symmetric_weight = algos
    compute_precision, explicit_dequantize = test_case

    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    activation_tensor_config = None
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.CONV_2D_TRANSPOSE,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,  # Only int8 is supported for now.
                symmetric=symmetric_weight,
                granularity=granularity,
            ),
            compute_precision=compute_precision,
            explicit_dequantize=explicit_dequantize,
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_conv2d_transpose,
        get_quant_params_func,
        bias_quantized_dim=None,
        input_index=2,
        weight_index=1,
        bias_index=3,
        inputs_to_ignore=[0],
    )

  @parameterized.product(
      activation_num_bits=(8, 16),
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
  )
  def test_materialize_srq_conv2d_transpose_succeeds(
      self,
      activation_num_bits,
      get_tensor_quant_params_func,
  ):
    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]

    if activation_num_bits == 8:
      activation_tensor_config = _TensorQuantConfig(
          num_bits=8,
          symmetric=False,
          granularity=qtyping.QuantGranularity.TENSORWISE,
      )
    else:
      activation_tensor_config = _TensorQuantConfig(
          num_bits=16,
          symmetric=True,
          granularity=qtyping.QuantGranularity.TENSORWISE,
      )
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.CONV_2D_TRANSPOSE,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=activation_tensor_config,
            weight_tensor_config=_TensorQuantConfig(
                num_bits=8,  # Only int8 is supported for now.
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
            ),
            compute_precision=_ComputePrecision.INTEGER,  # SRQ.
        ),
    )
    self._test_fc_bmm_conv(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_conv2d_transpose,
        get_tensor_quant_params_func,
        bias_quantized_dim=None,
        input_index=2,
        weight_index=1,
        bias_index=3,
        inputs_to_ignore=[0],
    )

  @parameterized.product(
      get_tensor_quant_params_func=(
          naive_min_max_quantize.get_tensor_quant_params,
          octav.get_tensor_quant_params,
      ),
      # min_weight_elements, whether to expect weights to be quantized.
      test_case=(
          ("weights_are_not_quantized", 1000000, False),
          ("weights_are_quantized_for_min_weight_elements_0", 0, True),
          ("weights_are_quantized_for_min_weight_elements_1", 1, True),
      ),
  )
  def test_materialize_conv2d_transpose_quantizes_weights_larger_than_min_weight_elements_for_w8_afp32(
      self, get_tensor_quant_params_func, test_case
  ):
    _, min_weight_elements, expect_weights_quantized = test_case
    self._test_materialize_fn_quantizes_weights_larger_than_min_weight_elements_for_w8_afp32(
        op_name=qtyping.TFLOperationName.CONV_2D_TRANSPOSE,
        subgraph_op_id=0,
        min_weight_elements=min_weight_elements,
        graph_info=self._graph_info,
        op_test_info=self._op_test_info,
        materialization_func=common_quantize.materialize_conv2d_transpose,
        get_tensor_quant_params_func=get_tensor_quant_params_func,
        expect_weights_quantized=expect_weights_quantized,
    )


if __name__ == "__main__":
  absltest.main()
