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

"""E2E tests for the quantizer for model with pad."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils


_TEST_MODEL_FOLDER = test_utils.get_path_to_datafile('../models/')


class PadTest(test_utils.BaseOpTestCase):

  def setUp(self):
    super().setUp()
    self._op_name = qtyping.TFLOperationName.PAD

  @parameterized.product(
      algorithm_key=[
          quantizer.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
          quantizer.AlgorithmName.OCTAV,
      ],
      activations_num_bits_and_symmetric=[
          (8, True),
          (8, False),
          (16, True),
      ],
  )
  def test_static_quantization_accuracy_and_size_within_tolerance(
      self, algorithm_key, activations_num_bits_and_symmetric
  ):
    output_tolerance = 1e-2
    # Model size increases due to q-dq inserted around the pad op.
    expected_model_size_reduction = -40

    model_filename = 'single_pad.tflite'
    model_path = pathlib.Path(_TEST_MODEL_FOLDER) / model_filename

    activation_config = test_utils.get_static_activation_quant_setting(
        *activations_num_bits_and_symmetric
    )
    op_config = test_utils.get_static_op_quant_config(activation_config)
    self.assert_quantization_accuracy_and_size(
        algorithm_key,
        model_path,
        self._op_name,
        op_config,
        expected_model_size_reduction=expected_model_size_reduction,
        output_tolerance=output_tolerance,
    )


if __name__ == '__main__':
  absltest.main()
