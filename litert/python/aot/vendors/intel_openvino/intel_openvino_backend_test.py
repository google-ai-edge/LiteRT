# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

"""Unit tests for the Intel OpenVINO AOT backend."""

from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot.core import aot_types
from litert.python.aot.core import components
from litert.python.aot.core import test_common
from litert.python.aot.vendors.intel_openvino import intel_openvino_backend
from litert.python.aot.vendors.intel_openvino import target as target_lib


class MockComponent:

  @property
  def component_name(self) -> str:
    return "mock_component"

  def __call__(self, *args, **kwargs):
    pass


class MockApplyPlugin(components.ApplyPluginT):

  def __call__(self, *args, **kwargs):
    pass


class MockMlirTransforms(components.MlirTransformsT):

  def __call__(self, *args, **kwargs):
    pass


class IntelOpenVinoBackendTest(test_common.TestWithTfliteModels):

  @property
  def basic_config(self) -> aot_types.Config:
    return {
        "backend_id": intel_openvino_backend.IntelOpenVinoBackend.id(),
        "output_dir": self.output_dir(),
    }

  @property
  def output_model(self) -> aot_types.Model:
    return aot_types.Model(self.output_dir() / "output.tflite")

  def test_unsupported_component(self):
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(
        self.basic_config
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "intel_openvino backend does not support mock_component component.",
    ):
      model = aot_types.Model("add_simple.tflite")
      component = MockComponent()
      backend.call_component(model, self.output_model, component)

  @mock.patch.object(MockApplyPlugin, "__call__")
  def test_apply_plugin(self, mck: mock.Mock):
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(
        self.basic_config
    )
    model = aot_types.Model("add_simple.tflite")
    component = MockApplyPlugin()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockMlirTransforms, "__call__")
  def test_mlir_transforms(self, mck: mock.Mock):
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(
        self.basic_config
    )
    model = aot_types.Model("add_simple.tflite")
    component = MockMlirTransforms()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  def test_bad_config(self):
    config = {}
    with self.assertRaisesRegex(ValueError, "Invalid backend id"):
      intel_openvino_backend.IntelOpenVinoBackend.create(config)

  def test_specialize_final(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.LNL.value
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, 1)

  def test_specialize_all(self):
    config = self.basic_config
    config["soc_model"] = target_lib.SocModel.ALL.value
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(config)
    backends = list(backend.specialize())
    self.assertLen(backends, len(target_lib.SocModel) - 1)

  def test_target_id(self):
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(
        self.basic_config
    )
    self.assertEqual(backend.target_id_suffix, "_Intel_ALL")

  @mock.patch.object(MockApplyPlugin, "__call__", autospec=True)
  def test_apply_plugin_with_intel_openvino_flags(
      self, mock_apply_plugin_call: mock.Mock
  ):
    config = self.basic_config
    config["intel_openvino_device_type"] = "npu"
    config["intel_openvino_performance_mode"] = "latency"
    config["intel_openvino_configs_map"] = "CACHE_DIR=/tmp/ov_cache"
    config["Unsupported_flag"] = "unsupported_value"
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(config)
    model = aot_types.Model("add_simple.tflite")
    output_model = self.output_model
    component = MockApplyPlugin()
    backend.call_component(model, output_model, component)
    mock_apply_plugin_call.assert_called_once()
    _, kwargs = mock_apply_plugin_call.call_args
    self.assertEqual(kwargs["intel_openvino_device_type"], "npu")
    self.assertEqual(kwargs["intel_openvino_performance_mode"], "latency")
    self.assertEqual(
        kwargs["intel_openvino_configs_map"], "CACHE_DIR=/tmp/ov_cache"
    )
    self.assertNotIn("Unsupported_flag", kwargs)

  @mock.patch.object(MockApplyPlugin, "__call__", autospec=True)
  def test_apply_plugin_with_compilation_config_flags(
      self, mock_apply_plugin_call: mock.Mock
  ):
    config = self.basic_config
    config["compilation_config"] = {
        "intel_openvino_device_type": "auto",
        "intel_openvino_performance_mode": "throughput",
    }
    backend = intel_openvino_backend.IntelOpenVinoBackend.create(config)
    model = aot_types.Model("add_simple.tflite")
    output_model = self.output_model
    component = MockApplyPlugin()
    backend.call_component(model, output_model, component)
    mock_apply_plugin_call.assert_called_once()
    _, kwargs = mock_apply_plugin_call.call_args
    self.assertEqual(kwargs["intel_openvino_device_type"], "auto")
    self.assertEqual(kwargs["intel_openvino_performance_mode"], "throughput")


if __name__ == "__main__":
  googletest.main()
