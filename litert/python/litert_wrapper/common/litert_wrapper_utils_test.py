# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for litert_wrapper_utils."""

import unittest

from litert.python.litert_wrapper.common import _litert_wrapper_utils_test_helper


class LiteRtWrapperUtilsTest(unittest.TestCase):

  def test_tensor_buffer_capsule_cleanup(self):
    capsule = (
        _litert_wrapper_utils_test_helper.make_testing_tensor_buffer_capsule()
    )
    self.assertIsNotNone(capsule)
    del capsule


if __name__ == "__main__":
  unittest.main()
