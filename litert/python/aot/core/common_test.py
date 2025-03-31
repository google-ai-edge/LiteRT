# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import absltest as googletest
from litert.python.aot.core import common

_TFL_MODEL = pathlib.Path("test/testdata/add_simple.tflite")


class CommonTest(googletest.TestCase):

  def test_get_resource(self):
    resource = common.get_resource(_TFL_MODEL)
    self.assertTrue(resource.exists())
    self.assertTrue(resource.is_file())

  def test_get_resource_non_existent(self):
    with self.assertRaises(FileNotFoundError):
      common.get_resource(pathlib.Path("non_existent.tflite"))

  def test_is_tflite(self):
    resource = common.get_resource(_TFL_MODEL)
    self.assertTrue(common.is_tflite(resource))


if __name__ == "__main__":
  googletest.main()
