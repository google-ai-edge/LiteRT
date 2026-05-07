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

"""Backend implementation for the Intel OpenVINO compiler plugin."""

import copy
import functools
import os
import pathlib
import sys
from typing import Any, Iterable

from litert.python.aot.core import aot_types
from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.vendors import import_vendor
from litert.python.aot.vendors.intel_openvino import target as target_lib

if sys.platform == "win32":
  COMPILER_PLUGIN_LIB_PATH = pathlib.Path(
      "vendors/intel_openvino/compiler/LiteRtCompilerPlugin.dll"
  )
  DISPATCH_LIB_PATH = pathlib.Path(
      "vendors/intel_openvino/dispatch/LiteRtDispatch.dll"
  )
else:
  COMPILER_PLUGIN_LIB_PATH = pathlib.Path(
      "vendors/intel_openvino/compiler/libLiteRtCompilerPlugin_IntelOpenvino.so"
  )
  DISPATCH_LIB_PATH = pathlib.Path(
      "vendors/intel_openvino/dispatch/libLiteRtDispatch_IntelOpenvino.so"
  )


def _find_lib_dir(lib_path: pathlib.Path) -> str | None:
  """Auto-discover a library directory from the wheel package."""
  try:
    resolved = common.get_resource(lib_path)
    return os.path.dirname(resolved)
  except FileNotFoundError:
    pass
  try:
    # pylint: disable=g-import-not-at-top
    from importlib import resources as _res

    root = _res.files("ai_edge_litert")
    for component in lib_path.parts:
      root = root.joinpath(component)
    p = pathlib.Path(str(root))
    if p.is_file():
      return str(p.parent)
  except (ModuleNotFoundError, TypeError):
    pass
  return None


def get_dispatch_dir() -> str | None:
  """Auto-discover the dispatch library directory from the wheel package."""
  return _find_lib_dir(DISPATCH_LIB_PATH)


def get_compiler_plugin_dir() -> str | None:
  """Auto-discover the compiler plugin directory from the wheel package."""
  return _find_lib_dir(COMPILER_PLUGIN_LIB_PATH)


def _is_intel_openvino_flag(flag: str) -> bool:
  """Returns true if the flag is an Intel OpenVINO flag."""
  return flag.startswith("intel_openvino_")


@import_vendor.register_backend
class IntelOpenVinoBackend(aot_types.Backend):
  """Backend implementation for the Intel OpenVINO compiler plugin."""

  def __init__(self, config: aot_types.Config):
    super().__init__(config)
    self._compilation_config = config.get("compilation_config", None)

  @property
  def soc_manufacturer(self) -> target_lib.SocManufacturer:
    return target_lib.SocManufacturer.INTEL

  @property
  def soc_model(self) -> target_lib.SocModel:
    return target_lib.SocModel(self.config.get("soc_model", "ALL"))

  @property
  def target(self) -> target_lib.Target:
    return target_lib.Target(self.soc_model, self.soc_manufacturer)

  @property
  def target_id(self) -> str:
    return repr(self.target)

  def specialize(self) -> Iterable["IntelOpenVinoBackend"]:
    if self.soc_model != target_lib.SocModel.ALL:
      yield self
    else:
      for soc_model in target_lib.SocModel:
        if soc_model != target_lib.SocModel.ALL:
          new_config = copy.deepcopy(self.config)
          new_config["soc_model"] = soc_model.value
          yield self.create(new_config)

  @classmethod
  def id(cls) -> str:
    # pylint: disable=protected-access
    return target_lib._INTEL_OPENVINO_BACKEND_ID

  @classmethod
  def create(cls, config: aot_types.Config) -> "IntelOpenVinoBackend":
    if config.get("backend_id", "") != cls.id():
      raise ValueError("Invalid backend id")
    return cls(config)

  def call_component(
      self,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      component: aot_types.Component,
  ) -> Any:
    return _call_component(component, self, input_model, output_model)


@functools.singledispatch
def _call_component(
    component: aot_types.Component,
    backend: IntelOpenVinoBackend,
    unused_input_model: aot_types.Model,
    unused_output_model: aot_types.Model,
):
  raise NotImplementedError(
      f"{backend.id()} backend does not support"
      f" {component.component_name} component."
  )


def _get_openvino_libs_path() -> str | None:
  """Returns the path to OpenVINO's shared library directory."""
  try:
    import openvino as _ov  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top

    ov_dir = os.path.dirname(_ov.__file__)
    libs_dir = os.path.join(ov_dir, "libs")
    if os.path.isdir(libs_dir):
      return libs_dir
    return ov_dir
  except (ImportError, TypeError):
    return None


@_call_component.register
def _apply_plugin(
    component: components.ApplyPluginT,
    backend: IntelOpenVinoBackend,
    input_model: aot_types.Model,
    output_model: aot_types.Model,
):
  """Calls the apply plugin component."""
  try:
    # If the plugin is not built from source (i.e. using ai_edge_litert wheel),
    # we find the plugin library directory from the package path.
    # Otherwise we use the default library path.
    plugin_path = common.get_resource(COMPILER_PLUGIN_LIB_PATH)
    lib_dir = os.path.dirname(plugin_path)

    sdk_libs_path = _get_openvino_libs_path()
    try:
      # pytype: disable=import-error
      import ai_edge_litert_sdk_intel  # pylint: disable=g-import-not-at-top
      # pytype: enable=import-error

      intel_sdk_libs = ai_edge_litert_sdk_intel.path_to_sdk_libs()
      if intel_sdk_libs:
        sdk_libs_path = (
            f"{intel_sdk_libs}{os.pathsep}{sdk_libs_path}"
            if sdk_libs_path
            else str(intel_sdk_libs)
        )
    except ImportError:
      pass

    extra_kwargs = {
        "libs": lib_dir,
        "sdk_libs_path": sdk_libs_path,
    }
  except FileNotFoundError:
    extra_kwargs = {}

  # Add Intel OpenVINO specific flags from the backend config.
  for flag, value in backend.config.items():
    if _is_intel_openvino_flag(flag):
      extra_kwargs[flag] = value

  for flag, value in backend.config.get("compilation_config", {}).items():
    if _is_intel_openvino_flag(flag):
      extra_kwargs[flag] = value

  return component(
      input_model,
      output_model,
      backend.soc_manufacturer,
      backend.soc_model,
      **extra_kwargs,
  )


@_call_component.register
def _mlir_transforms(
    component: components.MlirTransformsT,
    unused_backend: IntelOpenVinoBackend,
    input_model: aot_types.Model,
    output_model: aot_types.Model,
):
  return component(input_model, output_model, [])
