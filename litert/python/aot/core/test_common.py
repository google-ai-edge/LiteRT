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

"""Test utilities for LiteRt."""

import pathlib
import tempfile
from typing import Any

from absl.testing import absltest as googletest
from litert.python.aot.core import common

_TEST_DATA_DIR = pathlib.Path("test/testdata")


class TestWithTfliteModels(googletest.TestCase):
  """Test class for accessing generated tflite test models.

  Also provides a temp dir for convenience.
  """

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    super().tearDown()
    self.temp_dir.cleanup()

  def output_dir(self) -> pathlib.Path:
    return pathlib.Path(self.temp_dir.name)

  def get_model_path(self, model_name: str) -> pathlib.Path:
    model_path = _TEST_DATA_DIR / model_name
    resource_path = common.get_resource(model_path)
    self.assertTrue(common.is_tflite(resource_path))
    return resource_path

  def get_model_paths(self) -> list[pathlib.Path]:
    resource_paths = [
        common.get_resource(mp) for mp in _TEST_DATA_DIR.glob("*.tflite")
    ]
    for rp in resource_paths:
      self.assertTrue(common.is_tflite(rp))
    return resource_paths

  def get_touch_side_effect(self, filename: pathlib.Path, ret_val: Any = None):
    def side_effect(*unused_args, **unused_kwargs):
      filename.touch()
      return ret_val

    return side_effect
