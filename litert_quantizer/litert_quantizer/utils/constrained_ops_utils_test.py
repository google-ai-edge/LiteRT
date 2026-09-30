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

from absl.testing import absltest
from absl.testing import parameterized

from litert_quantizer.algorithms.utils import common_utils
from litert_quantizer.utils import constrained_ops_utils


_OpQuantConstraint = common_utils.OpQuantConstraint


class ConstrainedOpsUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="same_as_input_scale",
          constraint=_OpQuantConstraint.SAME_AS_INPUT_SCALE,
          expected_num_ops=17,
      ),
      dict(
          testcase_name="same_as_output_scale",
          constraint=_OpQuantConstraint.SAME_AS_OUTPUT_SCALE,
          expected_num_ops=7,
      ),
      dict(
          testcase_name="no_constrain",
          constraint=_OpQuantConstraint.NO_CONSTRAIN,
          expected_num_ops=23,
      ),
      dict(
          testcase_name="fixed_output_scale",
          constraint=_OpQuantConstraint.FIXED_OUTPUT_SCALE,
          expected_num_ops=3,
      ),
  )
  def test_get_constrained_op_list(self, constraint, expected_num_ops):
    constrained_ops = constrained_ops_utils.get_constrained_op_list(constraint)
    self.assertLen(constrained_ops, expected_num_ops)


if __name__ == "__main__":
  absltest.main()
