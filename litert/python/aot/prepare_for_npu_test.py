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

from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot import prepare_for_npu
from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import test_common
from litert.python.aot.core import types
from litert.python.aot.vendors.example import example_backend


class MockAieQuantizer(components.AieQuantizerT):

  @property
  def default_recipe(self) -> components.QuantRecipe:
    return []

  def __call__(self, *args, **kwargs):
    pass


class MockApplyPlugin(components.ApplyPluginT):

  def __call__(self, *args, **kwargs):
    pass


class MockMlirTransforms(components.MlirTransformsT):

  def __call__(self, *args, **kwargs):
    pass


class PrepareForNpuTest(test_common.TestWithTfliteModels):

  @mock.patch.object(MockMlirTransforms, "__call__")
  @mock.patch.object(MockApplyPlugin, "__call__")
  @mock.patch.object(MockAieQuantizer, "__call__")
  def test_prepare_for_example_npu(
      self, q_mck: mock.Mock, mck: mock.Mock, mlir_mck: mock.Mock
  ):
    mck.side_effect = self.get_touch_side_effect(
        self.output_dir() / "add_simple_apply_plugin.tflite", 0
    )
    q_mck.side_effect = self.get_touch_side_effect(
        self.output_dir() / "add_simple_aie_quantizer.tflite", 0
    )
    mlir_mck.side_effect = self.get_touch_side_effect(
        self.output_dir() / "add_simple_mlir_transforms.tflite", 0
    )
    model_path = self.get_model_path("add_simple.tflite")
    output_model = prepare_for_npu.prepare_for_npu(
        types.Model.create_from_path(model_path),
        self.output_dir(),
        example_backend.ExampleBackend,
        {"backend_id": "example"},
        MockApplyPlugin(),
        MockMlirTransforms(),
        MockAieQuantizer(),
    ).models[0]
    mck.assert_called_once()
    q_mck.assert_called_once()
    mlir_mck.assert_called_once()
    self.assertEqual(output_model.path.parent, self.output_dir())
    self.assertTrue(common.is_tflite(output_model.path))

  def test_prepare_for_npu_bad_config(self):
    model_path = self.get_model_path("add_simple.tflite")
    with self.assertRaises(ValueError):
      prepare_for_npu.prepare_for_npu(
          types.Model.create_from_path(model_path),
          self.output_dir(),
          example_backend.ExampleBackend,
          {"backend_id": "something_else"},
          MockApplyPlugin(),
          MockMlirTransforms(),
          MockAieQuantizer(),
      )


if __name__ == "__main__":
  googletest.main()
