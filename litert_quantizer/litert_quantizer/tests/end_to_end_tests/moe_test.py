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

"""E2E tests for the quantizer for model with the MoE custom op."""

import pathlib

from absl.testing import absltest

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils


_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile('../models/')
_QuantAlgo = quantizer.AlgorithmName


class MoeTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = qtyping.TFLOperationName.CUSTOM_OP

  def test_quantization_accuracy_within_tolerance(self):
    model_filename = 'single_moe_multiaxis.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model_filename
    op_config = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=8,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            quantized_dimensions=[0, 1],
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
    )
    self.assert_quantization_accuracy(
        algorithm_key=_QuantAlgo.MIN_MAX_UNIFORM_QUANT,
        model_path=model_path,
        op_name=self._op_name,
        op_config=op_config,
        output_tolerance=1e-5,
        # Keeps the random top-k expert indices valid for the 2 experts.
        min_max_range=(0, 1),
    )


if __name__ == '__main__':
  absltest.main()
