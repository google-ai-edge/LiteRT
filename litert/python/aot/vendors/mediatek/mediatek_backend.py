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
import itertools
import os
import pathlib
from typing import Iterable

from litert.python.aot.core import common
from litert.python.aot.core import components
from litert.python.aot.core import types
from litert.python.aot.vendors import import_vendor
from litert.python.aot.vendors.mediatek import target as target_lib

COMPILER_PLUGIN_LIB_PATH = pathlib.Path(
    "vendors/mediatek/compiler/libLiteRtCompilerPlugin_MediaTek.so"
)


@import_vendor.register_backend
class MediaTekBackend(types.Backend):
  """Backend implementation for the example compiler plugin."""

  def __init__(self, config: types.Config):
    super().__init__(config)
    self._compilation_config = config.get("compilation_config", None)

  @property
  def soc_manufacturer(self) -> target_lib.SocManufacturer:
    return target_lib.SocManufacturer.MEDIATEK

  @property
  def soc_model(self) -> target_lib.SocModel:
    return target_lib.SocModel(self.config.get("soc_model", "ALL"))

  @property
  def android_os_version(self) -> target_lib.AndroidOsVersion:
    return target_lib.AndroidOsVersion(
        self.config.get("android_os_version", "ALL")
    )

  @property
  def target(self) -> target_lib.Target:
    return target_lib.Target(
        self.soc_model, self.soc_manufacturer, self.android_os_version
    )

  @property
  def target_id(self) -> str:
    return repr(self.target)

  def specialize(self) -> Iterable["MediaTekBackend"]:
    if (
        self.soc_model != target_lib.SocModel.ALL
        and self.android_os_version != target_lib.AndroidOsVersion.ALL
    ):
      yield self
    else:
      if self.soc_model == target_lib.SocModel.ALL:
        soc_models = filter(
            lambda x: x != target_lib.SocModel.ALL, target_lib.SocModel
        )
      else:
        soc_models = [self.soc_model]
      if self.android_os_version == target_lib.AndroidOsVersion.ALL:
        android_os_versions = [
            x
            for x in target_lib.AndroidOsVersion
            if x != target_lib.AndroidOsVersion.ALL
        ]
      else:
        android_os_versions = [self.android_os_version]
      for soc_model, android_os_version in itertools.product(
          soc_models, android_os_versions
      ):
        new_config = copy.deepcopy(self.config)
        new_config["soc_model"] = soc_model.value
        new_config["android_os_version"] = android_os_version.value
        yield self.create(new_config)

  @classmethod
  def id(cls) -> str:
    return target_lib._MEDIATEK_BACKEND_ID  # pylint: disable=protected-access

  @classmethod
  def create(cls, config: types.Config) -> "MediaTekBackend":
    if config.get("backend_id", "") != cls.id():
      raise ValueError("Invalid backend id")
    return cls(config)

  @property
  def quantize_recipe(self) -> str | None:
    return self.config.get("quantize_recipe", None)

  def call_component(
      self,
      input_model: types.Model,
      output_model: types.Model,
      component: types.Component,
  ):
    return _call_component(component, self, input_model, output_model)


@functools.singledispatch
def _call_component(
    component: types.Component,
    backend: MediaTekBackend,
    unused_input_model: types.Model,
    unused_output_model: types.Model,
):
  raise NotImplementedError(
      f"{backend.id()} backend does not support"
      f" {component.component_name} component."
  )


# TODO(toribiosteven): Translate SOC | OS version to the corresponding
# MediaTek SDK version and pass to the plugin.
@_call_component.register
def _apply_plugin(
    component: components.ApplyPluginT,
    backend: MediaTekBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  """Calls the apply plugin component."""
  try:
    # If the plugin is not built from source (i.e. using ai_edge_litert wheel),
    # we find the plugin library directory from the package path.
    # Otherwise we use the default library path.
    plugin_path = common.get_resource(COMPILER_PLUGIN_LIB_PATH)
    lib_dir = os.path.dirname(plugin_path)

    try:
      # pytype: disable=import-error
      import ai_edge_litert_sdk_mediatek  # pylint: disable=g-import-not-at-top
      # pytype: enable=import-error

      # TODO(weiyiw): Translate SOC | OS version to the corresponding
      # MediaTek SDK version and pass to the plugin.
      sdk_version = "v8"
      sdk_libs_path = str(
          ai_edge_litert_sdk_mediatek.path_to_sdk_libs(sdk_version)
      )
    except ImportError:
      sdk_libs_path = None
    extra_kwargs = {"libs": lib_dir, "sdk_libs_path": sdk_libs_path}
  except FileNotFoundError:
    extra_kwargs = {}
  return component(
      input_model,
      output_model,
      backend.soc_manufacturer,
      backend.soc_model.lower(),
      **extra_kwargs,
  )


@_call_component.register
def _aie_quantizer(
    component: components.AieQuantizerT,
    backend: MediaTekBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(
      input_model,
      output_model,
      quantization_recipe=backend.quantize_recipe,
  )


@_call_component.register
def _mlir_transforms(
    component: components.MlirTransformsT,
    unused_backend: MediaTekBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(input_model, output_model, [])
