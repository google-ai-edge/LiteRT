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

import os
import pathlib
import sys
import types
from unittest import mock

from absl.testing import absltest as googletest
from litert.python.aot.core import aot_types
from litert.python.aot.core import components
from litert.python.aot.core import test_common
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
  def basic_config(self) -> aot_types.Config:
    return {
        "backend_id": qualcomm_backend.QualcommBackend.id(),
        "output_dir": self.output_dir(),
    }

  @property
  def output_model(self) -> aot_types.Model:
    return aot_types.Model(self.output_dir() / "output.tflite")

  def test_unsupported_component(self):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    with self.assertRaisesRegex(
        NotImplementedError,
        "qualcomm backend does not support mock_component component.",
    ):
      model = aot_types.Model("add_simple.tflite")
      component = MockComponent()
      backend.call_component(model, self.output_model, component)

  @mock.patch.object(MockApplyPlugin, "__call__")
  def test_apply_plugin(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = aot_types.Model("add_simple.tflite")
    component = MockApplyPlugin()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockAieQuantizer, "__call__")
  def test_aie_quantizer(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = aot_types.Model("add_simple.tflite")
    component = MockAieQuantizer()
    backend.call_component(model, self.output_model, component)
    mck.assert_called_once()

  @mock.patch.object(MockMlirTransforms, "__call__")
  def test_mlir_transforms(self, mck: mock.Mock):
    backend = qualcomm_backend.QualcommBackend.create(self.basic_config)
    model = aot_types.Model("add_simple.tflite")
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


_SDK_MODULE = "ai_edge_litert_sdk_qualcomm"
_AARCH64_SUBDIR = "aarch64-oe-linux-gcc11.2"
_X86_SUBDIR = "x86_64-linux-clang"


class _RecordingApplyPlugin(components.ApplyPluginT):

  def __init__(self):
    self.calls = []

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))


def _fake_qnn_sdk(
    lib_dir=None, *, error=None, default_paths=None
) -> types.ModuleType:
  """Returns a fake ai_edge_litert_sdk_qualcomm module."""
  module = types.ModuleType(_SDK_MODULE)
  setattr(
      module,
      "path_to_sdk_libs",
      mock.Mock(return_value=lib_dir, side_effect=error),
  )
  setattr(
      module, "default_qualcomm_paths", mock.Mock(return_value=default_paths)
  )
  return module


class QualcommSocModelTest(googletest.TestCase):

  def test_qcs_models_exist(self):
    self.assertEqual(
        target_lib.SocModel("QCS8275"), target_lib.SocModel.QCS8275
    )
    self.assertEqual(
        target_lib.SocModel("QCS8300"), target_lib.SocModel.QCS8300
    )

  def test_qcs_target_repr_and_flatten(self):
    target = target_lib.Target(target_lib.SocModel.QCS8275)
    self.assertEqual(repr(target), "Qualcomm_QCS8275")
    flat = target.flatten()
    self.assertEqual(flat["soc_model"], "QCS8275")
    self.assertEqual(flat["soc_manufacturer"], "Qualcomm")

  def test_specialize_all_includes_qcs_models(self):
    backend = qualcomm_backend.QualcommBackend.create({
        "backend_id": qualcomm_backend.QualcommBackend.id(),
        "output_dir": self.create_tempdir().full_path,
        "soc_model": target_lib.SocModel.ALL.value,
    })
    soc_models = {b.soc_model for b in backend.specialize()}
    self.assertIn(target_lib.SocModel.QCS8275, soc_models)
    self.assertIn(target_lib.SocModel.QCS8300, soc_models)


class ResolveSdkLibsPathTest(googletest.TestCase):
  """Tests for qualcomm_backend._resolve_sdk_libs_path."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    # Hermetic environment: no SDK module and no QNN env vars, Linux host.
    self.enter_context(mock.patch.dict(os.environ))
    os.environ.pop("QAIRT_LIB_PATH", None)
    os.environ.pop("QNN_SDK_ROOT", None)
    self.enter_context(mock.patch.dict(sys.modules, {_SDK_MODULE: None}))
    self.enter_context(
        mock.patch.object(qualcomm_backend.sys, "platform", "linux")
    )
    self.mock_machine = self.enter_context(
        mock.patch.object(
            qualcomm_backend.platform, "machine", return_value="x86_64"
        )
    )
    self.qairt_dir = self._make_dir("env_qairt")
    self.sdk_dir = self._make_dir("sdk_lib")

  def _make_dir(self, name: str) -> str:
    path = self.tmp_dir / name
    path.mkdir(parents=True)
    return str(path)

  def _backend(self, **config_overrides) -> qualcomm_backend.QualcommBackend:
    config = {
        "backend_id": qualcomm_backend.QualcommBackend.id(),
        "output_dir": str(self.tmp_dir),
        "soc_model": target_lib.SocModel.QCS8275.value,
    }
    config.update(config_overrides)
    return qualcomm_backend.QualcommBackend.create(config)

  def _resolve(self, **config_overrides) -> str | None:
    return qualcomm_backend._resolve_sdk_libs_path(
        self._backend(**config_overrides)
    )

  def _resolve_with_sdk(self, sdk: types.ModuleType) -> str | None:
    with mock.patch.dict(sys.modules, {_SDK_MODULE: sdk}):
      return self._resolve()

  def _make_sdk_root(self, *subdirs: str) -> pathlib.Path:
    root = self.tmp_dir / "qairt"
    root.mkdir(exist_ok=True)
    for subdir in subdirs:
      (root / "lib" / subdir).mkdir(parents=True)
    return root

  def _set_qnn_sdk_root(self) -> str:
    """Points $QNN_SDK_ROOT at an SDK with x86_64 libs; returns the lib dir."""
    root = self._make_sdk_root(_X86_SUBDIR)
    os.environ["QNN_SDK_ROOT"] = str(root)
    return str(root / "lib" / _X86_SUBDIR)

  def test_nothing_configured_returns_none(self):
    self.assertIsNone(self._resolve())

  def test_config_wins_over_everything(self):
    os.environ["QAIRT_LIB_PATH"] = self.qairt_dir
    self._set_qnn_sdk_root()
    sdk = _fake_qnn_sdk(self.sdk_dir)
    with mock.patch.dict(sys.modules, {_SDK_MODULE: sdk}):
      self.assertEqual(self._resolve(sdk_libs_path="/cfg/lib"), "/cfg/lib")
    sdk.path_to_sdk_libs.assert_not_called()

  def test_config_pathlib_is_stringified(self):
    self.assertEqual(
        self._resolve(sdk_libs_path=pathlib.Path("/cfg/lib")), "/cfg/lib"
    )

  def test_qairt_lib_path_wins_over_sdk_package(self):
    os.environ["QAIRT_LIB_PATH"] = self.qairt_dir
    sdk = _fake_qnn_sdk(self.sdk_dir)
    self.assertEqual(self._resolve_with_sdk(sdk), self.qairt_dir)
    sdk.path_to_sdk_libs.assert_not_called()

  def test_missing_qairt_lib_path_is_ignored(self):
    os.environ["QAIRT_LIB_PATH"] = str(self.tmp_dir / "does_not_exist")
    sdk = _fake_qnn_sdk(self.sdk_dir)
    self.assertEqual(self._resolve_with_sdk(sdk), self.sdk_dir)

  def test_qairt_lib_path_wins_over_qnn_sdk_root(self):
    os.environ["QAIRT_LIB_PATH"] = self.qairt_dir
    self._set_qnn_sdk_root()
    self.assertEqual(self._resolve(), self.qairt_dir)

  def test_empty_qairt_lib_path_is_ignored(self):
    os.environ["QAIRT_LIB_PATH"] = ""
    expected = self._set_qnn_sdk_root()
    self.assertEqual(self._resolve(), expected)

  def test_sdk_package_wins_over_qnn_sdk_root(self):
    self._set_qnn_sdk_root()
    sdk = _fake_qnn_sdk(pathlib.Path(self.sdk_dir))
    self.assertEqual(self._resolve_with_sdk(sdk), self.sdk_dir)
    sdk.path_to_sdk_libs.assert_called_once_with()

  def test_missing_sdk_lib_dir_falls_through_to_qnn_sdk_root(self):
    expected = self._set_qnn_sdk_root()
    sdk = _fake_qnn_sdk(str(self.tmp_dir / "does_not_exist"))
    self.assertEqual(self._resolve_with_sdk(sdk), expected)
    sdk.path_to_sdk_libs.assert_called_once_with()

  def test_sdk_returns_none_falls_through_to_qnn_sdk_root(self):
    # e.g. the SDK package is installed but its data/ dir was not downloaded.
    expected = self._set_qnn_sdk_root()
    self.assertEqual(self._resolve_with_sdk(_fake_qnn_sdk(None)), expected)

  def test_sdk_not_implemented_falls_through_to_qnn_sdk_root(self):
    expected = self._set_qnn_sdk_root()
    sdk = _fake_qnn_sdk(error=NotImplementedError("unsupported host"))
    self.assertEqual(self._resolve_with_sdk(sdk), expected)

  def test_qnn_sdk_root_ignored_off_linux(self):
    root = self._make_sdk_root(_AARCH64_SUBDIR, _X86_SUBDIR)
    os.environ["QNN_SDK_ROOT"] = str(root)
    for plat, machine in (
        ("darwin", "arm64"),
        ("darwin", "x86_64"),
        ("win32", "AMD64"),
    ):
      with self.subTest(plat=plat, machine=machine):
        self.mock_machine.return_value = machine
        with mock.patch.object(qualcomm_backend.sys, "platform", plat):
          self.assertIsNone(self._resolve())

  def test_qnn_sdk_root_selects_host_arch_subdir(self):
    root = self._make_sdk_root(_AARCH64_SUBDIR, _X86_SUBDIR)
    os.environ["QNN_SDK_ROOT"] = str(root)
    cases = (
        ("x86_64", _X86_SUBDIR),
        ("AMD64", _X86_SUBDIR),
        ("aarch64", _AARCH64_SUBDIR),
        ("arm64", _AARCH64_SUBDIR),
    )
    for machine, subdir in cases:
      with self.subTest(machine=machine):
        self.mock_machine.return_value = machine
        self.assertEqual(self._resolve(), str(root / "lib" / subdir))

  def test_qnn_sdk_root_x86_host_does_not_use_aarch64_libs(self):
    os.environ["QNN_SDK_ROOT"] = str(self._make_sdk_root(_AARCH64_SUBDIR))
    self.mock_machine.return_value = "x86_64"
    self.assertIsNone(self._resolve())

  def test_qnn_sdk_root_aarch64_host_does_not_use_x86_libs(self):
    os.environ["QNN_SDK_ROOT"] = str(self._make_sdk_root(_X86_SUBDIR))
    self.mock_machine.return_value = "aarch64"
    self.assertIsNone(self._resolve())

  def test_qnn_sdk_root_aarch64_host_tries_older_toolchain_builds(self):
    self.mock_machine.return_value = "aarch64"
    for i, (present, expected) in enumerate((
        (
            (_AARCH64_SUBDIR, "aarch64-ubuntu-gcc9.4"),
            _AARCH64_SUBDIR,
        ),
        (
            ("aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc9.3"),
            "aarch64-ubuntu-gcc9.4",
        ),
        (("aarch64-oe-linux-gcc9.3",), "aarch64-oe-linux-gcc9.3"),
    )):
      with self.subTest(present=present):
        root = self.tmp_dir / f"qairt_{i}"
        for subdir in present:
          (root / "lib" / subdir).mkdir(parents=True)
        os.environ["QNN_SDK_ROOT"] = str(root)
        self.assertEqual(self._resolve(), str(root / "lib" / expected))

  def test_qnn_sdk_root_without_lib_dirs_returns_none(self):
    os.environ["QNN_SDK_ROOT"] = str(self._make_sdk_root())
    self.assertIsNone(self._resolve())

  def test_qnn_sdk_root_subdir_must_be_a_directory(self):
    root = self._make_sdk_root()
    (root / "lib").mkdir()
    (root / "lib" / _X86_SUBDIR).write_text("not a dir")
    os.environ["QNN_SDK_ROOT"] = str(root)
    self.assertIsNone(self._resolve())


class QualcommApplyPluginSdkLibsTest(googletest.TestCase):
  """Tests that _apply_plugin forwards the resolved sdk_libs_path."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    self.mock_resolve = self.enter_context(
        mock.patch.object(
            qualcomm_backend, "_resolve_sdk_libs_path", return_value=None
        )
    )
    self.mock_get_resource = self.enter_context(
        mock.patch.object(
            qualcomm_backend.common,
            "get_resource",
            side_effect=FileNotFoundError("no plugin"),
        )
    )
    self.enter_context(mock.patch.dict(sys.modules, {_SDK_MODULE: None}))

  def _run(self):
    backend = qualcomm_backend.QualcommBackend.create({
        "backend_id": qualcomm_backend.QualcommBackend.id(),
        "output_dir": str(self.tmp_dir),
        "soc_model": target_lib.SocModel.QCS8275.value,
    })
    component = _RecordingApplyPlugin()
    backend.call_component(
        aot_types.Model(self.tmp_dir / "in.tflite"),
        aot_types.Model(self.tmp_dir / "out.tflite"),
        component,
    )
    self.assertLen(component.calls, 1)
    args, kwargs = component.calls[0]
    self.assertEqual(args[2], target_lib.SocManufacturer.QUALCOMM)
    self.assertEqual(args[3], target_lib.SocModel.QCS8275)
    self.mock_resolve.assert_called_once_with(backend)
    return kwargs

  def test_no_plugin_no_sdk_passes_no_extra_kwargs(self):
    self.assertEqual(self._run(), {})

  def test_no_plugin_with_sdk_libs_path(self):
    self.mock_resolve.return_value = "/qnn/lib"
    self.assertEqual(self._run(), {"sdk_libs_path": "/qnn/lib"})

  def test_plugin_found_passes_libs_and_sdk_libs_path(self):
    plugin = self.tmp_dir / "plugin" / "libplugin.so"
    self.mock_get_resource.side_effect = None
    self.mock_get_resource.return_value = plugin
    self.mock_resolve.return_value = "/qnn/lib"
    self.assertEqual(
        self._run(), {"libs": str(plugin.parent), "sdk_libs_path": "/qnn/lib"}
    )

  def test_on_device_passes_sdk_dsp_skel_dir(self):
    sdk = _fake_qnn_sdk(default_paths=("/sdk/lib", "/sdk/skel"))
    with mock.patch.dict(sys.modules, {_SDK_MODULE: sdk}):
      self.assertEqual(self._run(), {"qualcomm_dsp_skel_dir": "/sdk/skel"})

  def test_off_device_or_old_sdk_passes_no_dsp_skel_dir(self):
    for sdk in (_fake_qnn_sdk(), types.ModuleType(_SDK_MODULE)):
      with self.subTest(sdk=sdk):
        self.mock_resolve.reset_mock()
        with mock.patch.dict(sys.modules, {_SDK_MODULE: sdk}):
          self.assertEqual(self._run(), {})

  def test_plugin_found_without_sdk_libs_path(self):
    plugin = self.tmp_dir / "plugin" / "libplugin.so"
    self.mock_get_resource.side_effect = None
    self.mock_get_resource.return_value = plugin
    self.assertEqual(
        self._run(), {"libs": str(plugin.parent), "sdk_libs_path": None}
    )


if __name__ == "__main__":
  googletest.main()
