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

"""Backend implementation for the example compiler plugin.."""

import copy
import functools
import os
import pathlib
import platform
import sys
from typing import Iterable

from litert.python.aot.core import aot_types
from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.vendors import import_vendor
from litert.python.aot.vendors.qualcomm import target as target_lib

COMPILER_PLUGIN_LIB_PATH = pathlib.Path(
    "vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so"
)


@import_vendor.register_backend
class QualcommBackend(aot_types.Backend):
  """Backend implementation for the example compiler plugin."""

  def __init__(self, config: aot_types.Config):
    super().__init__(config)
    self._compilation_config = config.get("compilation_config", None)

  @property
  def soc_manufacturer(self) -> target_lib.SocManufacturer:
    return target_lib.SocManufacturer.QUALCOMM

  @property
  def soc_model(self) -> target_lib.SocModel:
    return target_lib.SocModel(self.config.get("soc_model", "ALL"))

  @property
  def target(self) -> target_lib.Target:
    return target_lib.Target(self.soc_model, self.soc_manufacturer)

  @property
  def target_id(self) -> str:
    return repr(self.target)

  def specialize(self) -> Iterable["QualcommBackend"]:
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
    return target_lib._QUALCOMM_BACKEND_ID  # pylint: disable=protected-access

  @classmethod
  def create(cls, config: aot_types.Config) -> "QualcommBackend":
    if config.get("backend_id", "") != cls.id():
      raise ValueError("Invalid backend id")
    return cls(config)

  @property
  def quantize_recipe(self) -> str | None:
    return self.config.get("quantize_recipe", None)

  def call_component(
      self,
      input_model: aot_types.Model,
      output_model: aot_types.Model,
      component: aot_types.Component,
  ):
    return _call_component(component, self, input_model, output_model)


@functools.singledispatch
def _call_component(
    component: aot_types.Component,
    backend: QualcommBackend,
    unused_input_model: aot_types.Model,
    unused_output_model: aot_types.Model,
):
  raise NotImplementedError(
      f"{backend.id()} backend does not support"
      f" {component.component_name} component."
  )


# Linux aarch64 QNN host library builds in QAIRT, newest toolchain first.
_QNN_HOST_LIB_SUBDIRS_AARCH64 = (
    "aarch64-oe-linux-gcc11.2",
    "aarch64-ubuntu-gcc9.4",
    "aarch64-oe-linux-gcc9.3",
)
_QNN_HOST_LIB_SUBDIR_X86_64 = "x86_64-linux-clang"


def _resolve_sdk_libs_path(backend: QualcommBackend) -> str | None:
  """Resolves the QNN host library directory used by apply_plugin_main.

  Resolution order: backend config `sdk_libs_path`, `$QAIRT_LIB_PATH`, the
  `ai_edge_litert_sdk_qualcomm` package, then
  `$QNN_SDK_ROOT/lib/<host-arch subdir>` (Linux only). Only existing
  directories are returned.

  Args:
    backend: The Qualcomm backend whose config may override the path.

  Returns:
    The QNN host library directory, or None if it cannot be resolved.
  """
  configured_path = backend.config.get("sdk_libs_path")
  if configured_path:
    return str(configured_path)
  env_lib_path = os.environ.get("QAIRT_LIB_PATH")
  if env_lib_path and os.path.isdir(env_lib_path):
    return env_lib_path
  try:
    import ai_edge_litert_sdk_qualcomm as sdk  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    sdk_libs = sdk.path_to_sdk_libs()
    if sdk_libs is not None and os.path.isdir(sdk_libs):
      return str(sdk_libs)
  except (ImportError, NotImplementedError):
    pass
  qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
  if qnn_sdk_root and sys.platform == "linux":
    subdirs = (
        _QNN_HOST_LIB_SUBDIRS_AARCH64
        if platform.machine() in ("aarch64", "arm64")
        else (_QNN_HOST_LIB_SUBDIR_X86_64,)
    )
    for subdir in subdirs:
      candidate = pathlib.Path(qnn_sdk_root) / "lib" / subdir
      if candidate.is_dir():
        return str(candidate)
  return None


def _device_dsp_skel_dir() -> str | None:
  """Returns the SDK's Hexagon skel dir when compiling on a Qualcomm device.

  On-device compilation opens a CDSP session, so apply_plugin_main also needs
  the DSP-side libraries.
  """
  try:
    import ai_edge_litert_sdk_qualcomm as sdk  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  except ImportError:
    return None
  # Older SDK releases predate on-device (Linux aarch64) support.
  default_paths = getattr(sdk, "default_qualcomm_paths", None)
  paths = default_paths() if default_paths else None
  return str(paths[1]) if paths else None


@_call_component.register
def _apply_plugin(
    component: components.ApplyPluginT,
    backend: QualcommBackend,
    input_model: aot_types.Model,
    output_model: aot_types.Model,
):
  """Calls the apply plugin component."""
  sdk_libs_path = _resolve_sdk_libs_path(backend)

  try:
    # If the plugin is not built from source (i.e. using ai_edge_litert wheel),
    # we find the plugin library directory from the package path.
    # Otherwise we use the default library path.
    plugin_path = common.get_resource(COMPILER_PLUGIN_LIB_PATH)
    lib_dir = os.path.dirname(plugin_path)
    extra_kwargs = {"libs": lib_dir, "sdk_libs_path": sdk_libs_path}
  except FileNotFoundError:
    extra_kwargs = {"sdk_libs_path": sdk_libs_path} if sdk_libs_path else {}
  dsp_skel_dir = _device_dsp_skel_dir()
  if dsp_skel_dir:
    extra_kwargs["qualcomm_dsp_skel_dir"] = dsp_skel_dir
  return component(
      input_model,
      output_model,
      backend.soc_manufacturer,
      backend.soc_model,
      **extra_kwargs,
  )


@_call_component.register
def _aie_quantizer(
    component: components.AieQuantizerT,
    backend: QualcommBackend,
    input_model: aot_types.Model,
    output_model: aot_types.Model,
):
  return component(
      input_model,
      output_model,
      quantization_recipe=backend.quantize_recipe,
  )


@_call_component.register
def _mlir_transforms(
    component: components.MlirTransformsT,
    unused_backend: QualcommBackend,
    input_model: aot_types.Model,
    output_model: aot_types.Model,
):
  return component(input_model, output_model, [])
