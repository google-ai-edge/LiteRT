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
from litert_quantizer.algorithms.uniform_quantize.op_architecture_tests import test_utils as op_test_utils
from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils

_TEST_DATA_PREFIX_PATH = test_utils.get_path_to_datafile(
    "../../../tests/models"
)
_ComputePrecision = qtyping.ComputePrecision
_TensorQuantConfig = qtyping.TensorQuantizationConfig
_QuantTransformation = qtyping.QuantTransformation
_OpTestInfo = op_test_utils.OpTestInfo


class MoeTest(op_test_utils.BaseQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "single_moe_multiaxis.tflite"
    )
    self._op_test_info = _OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={
            "input": "serving_default_args_0",
            "ignored_input2": "serving_default_args_1",
            "ignored_input3": "serving_default_args_2",
            "weight": "arith.constant",
            "weight2": "arith.constant1",
            "weight3": "arith.constant2",
            "ignored_input7": "arith.constant3",
            "output": "serving_default_output_0_output",
        },
        input_range=(np.array([[-10]]), np.array([[8]])),
        output_range=(np.array([[10]]), np.array([[88]])),
    )
    # The test model has one subgraph for now.
    self._graph_info = qtyping.GraphInfo(
        subgraph_tensors=self._op_test_info.test_model.subgraphs[0].tensors,
        buffers=self._op_test_info.test_model.buffers,
    )

  @parameterized.parameters(8, 4)
  def test_materialize_moe_succeeds(self, num_bits):
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    moe_op = subgraph0.operators[subgraph_op_id]

    op_info = qtyping.OpInfo(
        op=moe_op,
        op_name=qtyping.TFLOperationName.CUSTOM_OP,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=num_bits,
                symmetric=True,
                granularity=qtyping.QuantGranularity.CHANNELWISE,
                quantized_dimensions=[0, 1],
            ),
            compute_precision=_ComputePrecision.INTEGER,
        ),
    )

    tensor_quant_params = common_quantize.materialize_moe(
        naive_min_max_quantize.get_tensor_quant_params,
        op_info,
        self._graph_info,
        tensor_name_to_qsv={},
        tensor_quant_params_cache=common_utils.TensorQuantParamsCache(),
    )

    # Total 7 inputs + 1 output = 8 tensor transformation params.
    self.assertLen(tensor_quant_params, 8)

    # 1. Non-quantized input activation (input 0).
    self._test_tensor_transformation_params(
        self._op_test_info.op_tensor_names["input"],
        subgraph_op_id,
        is_inbounding_tensor=True,
        tensor_quant_config=None,
        transformation_params=tensor_quant_params[0],
        desired_transformations=[_QuantTransformation.NO_QUANTIZE],
    )

    # 2. Ignored inputs (inputs 1, 2, 6).
    self._test_ignored_inputs_and_outputs(
        tensor_quant_params,
        op_info,
        self._op_test_info,
        inputs_to_ignore=[1, 2, 6],
        is_op_with_weight=True,
    )

    # 3. Quantized Multi-Axis weights (inputs 3, 4, 5).
    for idx, key in [(3, "weight"), (4, "weight2"), (5, "weight3")]:
      with self.subTest(key):
        self._test_tensor_transformation_params(
            self._op_test_info.op_tensor_names[key],
            subgraph_op_id,
            is_inbounding_tensor=True,
            tensor_quant_config=op_info.op_quant_config.weight_tensor_config,
            transformation_params=tensor_quant_params[idx],
            desired_transformations=[_QuantTransformation.QUANTIZE_TENSOR],
            tensor_data=tfl_flatbuffer_utils.get_tensor_data(
                self._graph_info.subgraph_tensors[moe_op.inputs[idx]],
                self._graph_info.buffers,
            ),
            quantized_dimension=None,
        )
        weight_params = tensor_quant_params[idx].consumers[0].parameters
        self.assertEqual(weight_params.quantized_dimensions, (0, 1))

    # 4. Non-quantized output activation (output 0).
    self._test_tensor_transformation_params(
        self._op_test_info.op_tensor_names["output"],
        subgraph_op_id,
        is_inbounding_tensor=False,
        tensor_quant_config=None,
        transformation_params=tensor_quant_params[7],
        desired_transformations=[_QuantTransformation.NO_QUANTIZE],
    )


if __name__ == "__main__":
  absltest.main()
