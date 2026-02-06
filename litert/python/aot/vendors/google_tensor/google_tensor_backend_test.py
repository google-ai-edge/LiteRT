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

import os
from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import test_common
from litert.python.aot.core import types
from litert.python.aot.vendors.google_tensor import google_tensor_backend
from litert.python.aot.vendors.google_tensor import target as target_lib


class MockComponent:

  @property
  def component_name(self) -> str:
    return "mock_component"

  def __call__(self, *args, **kwargs):
    pass


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


class _MockSdk:
  """A mock class to be used for autospeccing the ai_edge_litert_sdk_google_tensor module."""

  def path_to_sdk_libs(self) -> str:
    """Returns the path to the SDK libraries."""
    raise NotImplementedError


class GoogleTensorBackendTest(test_common.TestWithTfliteModels):

  @property
  def basic_config(self) -> types.Config:
    return {
        "backend_id": google_tensor_backend.GoogleTensorBackend.id(),
        "output_dir": self.output_dir(),
    }

  @property
  def output_model(self) -> types.Model:
    return types.Model(self.output_dir() / "output.tflite")

  def test_unsupported_component(self):
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "GOOGLE backend does not support mock_component component.",
    ):
      model = types.Model("add_simple.tflite")
      component = MockComponent()
      backend.call_component(model, self.output_model, component)

  @mock.patch.object(common, "get_resource", autospec=True)
  @mock.patch.object(MockApplyPlugin, "__call__", autospec=True)
  @mock.patch.dict(
      os.environ,
      {"GOOGLE_TENSOR_COMPILER_LIB": "/google_tensor_compiler_libs_path"},
  )
  def test_apply_plugin(
      self, mock_apply_plugin_call: mock.Mock, mock_get_resource: mock.Mock
  ):
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    mock_get_resource.return_value = "/fake/path/to/plugin.so"
    model = types.Model("add_simple.tflite")
    output_model = self.output_model
    component = MockApplyPlugin()
    backend.call_component(model, output_model, component)
    mock_apply_plugin_call.assert_called_once()
    args, kwargs = mock_apply_plugin_call.call_args
    self.assertEqual(args[1], model)  # input_model
    self.assertEqual(args[2], output_model)  # output_model
    self.assertEqual(
        args[3], target_lib.SocManufacturer.GOOGLE
    )  # soc_manufacturer
    self.assertEqual(args[4], target_lib.SocModel.ALL)  # soc_model
    self.assertEqual(kwargs["libs"], "/fake/path/to")  # libs
    self.assertEqual(
        kwargs["sdk_libs_path"], "/google_tensor_compiler_libs_path"
    )  # sdk_libs_path

  @mock.patch.object(
      common,
      "get_resource",
      autospec=True,
      return_value="/fake/path/to/plugin.so",
  )
  @mock.patch.object(MockApplyPlugin, "__call__", autospec=True)
  def test_apply_plugin_with_sdk_import(
      self, mock_apply_plugin_call: mock.Mock, mock_get_resource: mock.Mock
  ):
    # Arrange: setup mocks.
    mock_sdk = mock.create_autospec(_MockSdk, instance=True)
    mock_sdk.path_to_sdk_libs.return_value = "/path/from/sdk"
    self.enter_context(
        mock.patch.dict(
            "sys.modules",
            {"ai_edge_litert_sdk_google_tensor": mock_sdk},
        )
    )
    # Arrange: setup test data.
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    model = types.Model("add_simple.tflite")
    output_model = self.output_model
    component = MockApplyPlugin()

    # Act.
    backend.call_component(model, output_model, component)

    # Assert.
    with self.subTest("get_resource"):
      mock_get_resource.assert_called_once()
    with self.subTest("apply_plugin_call"):
      mock_apply_plugin_call.assert_called_once()
      _, kwargs = mock_apply_plugin_call.call_args
      self.assertEqual(kwargs["sdk_libs_path"], "/path/from/sdk")

  @mock.patch.object(MockApplyPlugin, "__call__", autospec=True)
  def test_apply_plugin_with_compiler_flags(
      self, mock_apply_plugin_call: mock.Mock
  ):
    config = self.basic_config
    config["google_tensor_truncation_type"] = "bf16"
    config["Unsupported_flag"] = "unsupported_value"
    backend = google_tensor_backend.GoogleTensorBackend.create(config)
    model = types.Model("add_simple.tflite")
    output_model = self.output_model
    component = MockApplyPlugin()
    backend.call_component(model, output_model, component)
    mock_apply_plugin_call.assert_called_once()
    _, kwargs = mock_apply_plugin_call.call_args
    self.assertEqual(kwargs["google_tensor_truncation_type"], "bf16")
    self.assertNotIn("google_tensor_unsupported_flag", kwargs)

  @mock.patch.object(MockAieQuantizer, "__call__")
  def test_aie_quantizer(self, mck: mock.Mock):
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    model = types.Model("add_simple.tflite")
    component = MockAieQuantizer()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockMlirTransforms, "__call__")
  def test_mlir_transforms(self, mck: mock.Mock):
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    model = types.Model("add_simple.tflite")
    component = MockMlirTransforms()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  def test_bad_config(self):
    config = {}
    with self.assertRaisesRegex(ValueError, "Invalid backend id"):
      google_tensor_backend.GoogleTensorBackend.create(config)

  def test_specialize_final(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.TENSOR_G3.value
    backend = google_tensor_backend.GoogleTensorBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, 1)

  def test_specialize_all(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.ALL.value
    backend = google_tensor_backend.GoogleTensorBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, len(target_lib.SocModel) - 1)

  def test_target_id(self):
    backend = google_tensor_backend.GoogleTensorBackend.create(
        self.basic_config
    )
    self.assertEqual(backend.target_id_suffix, "_Google_ALL")


if __name__ == "__main__":
  googletest.main()
