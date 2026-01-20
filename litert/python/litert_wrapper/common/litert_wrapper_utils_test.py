# Copyright 2025 Google LLC.
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

import sys
import unittest

from litert.python.litert_wrapper.common import _litert_wrapper_utils_test_helper


class LiteRtWrapperUtilsTest(unittest.TestCase):

  def test_ref_counting(self):
    model = object()
    initial_ref = sys.getrefcount(model)

    # Create capsule, passing the model
    capsule = _litert_wrapper_utils_test_helper.make_capsule(model)

    # Refcount should increase by 1 because the capsule holds a reference
    self.assertEqual(sys.getrefcount(model), initial_ref + 1)

    # Verify context is correct (optional, exposed via pycapsule but helper
    # returns object) Since we can't easily access PyCapsule APIs from pure
    # python without ctypes or more helpers, we rely on refcount check.

    # Delete capsule
    del capsule

    # Refcount should return to initial
    self.assertEqual(sys.getrefcount(model), initial_ref)

  def test_no_model(self):
    # Just verify it doesn't crash
    capsule = _litert_wrapper_utils_test_helper.make_capsule(None)
    del capsule


if __name__ == "__main__":
  unittest.main()
