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
from litert.python.aot.core import components
from litert.python.aot.core import test_common
from litert.python.aot.core import types
from litert.python.aot.vendors.qualcomm import qualcomm_backend
from litert.python.aot.vendors.qualcomm import target as target_lib


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


class QualcommBackendTest(test_common.TestWithTfliteModels):

  @property
  def basic_config(self) -> types.Config:
    return {
        "backend_id": qualcomm_backend.QualcommBackend.id(),
        "output_dir": self.output_dir(),
    }

  @property
  def output_model(self) -> types.Model:
    return types.Model(self.output_dir() / "output.tflite")

  def test_unsupported_component(self):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    with self.assertRaisesRegex(
        NotImplementedError,
        "qualcomm backend does not support mock_component component.",
    ):
      model = types.Model("add_simple.tflite")
      component = MockComponent()
      backend.call_component(model, self.output_model, component)

  @mock.patch.object(MockApplyPlugin, "__call__")
  def test_apply_plugin(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = types.Model("add_simple.tflite")
    component = MockApplyPlugin()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockAieQuantizer, "__call__")
  def test_aie_quantizer(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = types.Model("add_simple.tflite")
    component = MockAieQuantizer()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockMlirTransforms, "__call__")
  def test_mlir_transforms(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = types.Model("add_simple.tflite")
    component = MockMlirTransforms()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  def test_bad_config(self):
    config = {}
    with self.assertRaisesRegex(ValueError, "Invalid backend id"):
      qualcomm_backend.QualcommBackend.create(config)

  def test_specialize_final(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.SM8750.value
    backend = qualcomm_backend.QualcommBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, 1)

  def test_specialize_all(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.ALL.value
    backend = qualcomm_backend.QualcommBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, len(target_lib.SocModel) - 1)

  def test_target_id(self):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    self.assertEqual(backend.target_id_suffix, "_Qualcomm_ALL")


if __name__ == "__main__":
  googletest.main()
