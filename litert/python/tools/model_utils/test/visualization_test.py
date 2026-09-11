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
"""Minimal regression test for Model Explorer integration in model_utils."""

from absl.testing import absltest as googletest
from litert.python.tools.model_utils.model_explorer_integration import adapter
from litert.python.tools.model_utils.model_explorer_integration import visualize


class VisualizationTest(googletest.TestCase):

  def test_adapter_import(self):
    a = adapter.Adapter()
    self.assertIsNotNone(a)

  def test_visualize_import(self):
    self.assertTrue(callable(visualize.visualize))


if __name__ == '__main__':
  googletest.main()
