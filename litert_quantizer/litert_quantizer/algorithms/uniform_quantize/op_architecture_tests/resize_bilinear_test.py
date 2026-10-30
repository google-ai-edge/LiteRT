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


class ResizeBilinearTest(op_test_utils.BaseQuantizeTest):

  def setUp(self):
    super().setUp()
    np.random.seed(666)
    self._test_model_path = str(
        pathlib.Path(_TEST_DATA_PREFIX_PATH) / "single_resize_bilinear.tflite"
    )
    self._op_test_info = op_test_utils.OpTestInfo(
        test_model=tfl_flatbuffer_utils.read_model(self._test_model_path),
        op_tensor_names={},
        input_range=(np.array([[-10]]), np.array([[10]])),
        output_range=(np.array([[-10]]), np.array([[10]])),
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
      activations_num_bits_and_symmetric=[
          (8, False),
          (8, True),
          (16, True),
      ],
  )
  def test_materialize_resize_bilinear_succeeds(
      self, get_tensor_quant_params_func, activations_num_bits_and_symmetric
  ):
    activation_config = test_utils.get_static_activation_quant_setting(
        *activations_num_bits_and_symmetric
    )
    op_quant_config = test_utils.get_static_op_quant_config(activation_config)

    # Read from Model Explorer.
    subgraph0 = self._op_test_info.test_model.subgraphs[0]
    subgraph_op_id = 0
    op = subgraph0.operators[subgraph_op_id]
    op_info = qtyping.OpInfo(
        op=op,
        op_name=qtyping.TFLOperationName.RESIZE_BILINEAR,
        subgraph_op_index=subgraph_op_id,
        op_quant_config=op_quant_config,
    )

    # Test settings.
    op_tensor_names = {}
    op_tensor_names["input"] = "serving_default_input:0"
    op_tensor_names["input2"] = "Const"
    op_tensor_names["output"] = "PartitionedCall:0"
    self._op_test_info.op_tensor_names = op_tensor_names
    self._test_no_weights_op(
        op_info,
        self._graph_info,
        self._op_test_info,
        common_quantize.materialize_resize_bilinear,
        get_tensor_quant_params_func,
        same_input_output_params=True,
        inputs_to_ignore=[1],  # Size tensor does not need to be quantized.
    )


if __name__ == "__main__":
  absltest.main()
