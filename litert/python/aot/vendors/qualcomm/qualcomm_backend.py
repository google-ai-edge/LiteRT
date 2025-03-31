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
from typing import Iterable, Self

from litert.python.aot.core import components
from litert.python.aot.core import types
from litert.python.aot.vendors import import_vendor
from litert.python.aot.vendors.qualcomm import target as target_lib


@import_vendor.register_backend
class QualcommBackend(types.Backend):
  """Backend implementation for the example compiler plugin."""

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
  def target_id_suffix(self) -> str:
    return "_" + repr(self.target)

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
  def create(cls, config: types.Config) -> Self:
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
    backend: QualcommBackend,
    unused_input_model: types.Model,
    unused_output_model: types.Model,
):
  raise NotImplementedError(
      f"{backend.id()} backend does not support"
      f" {component.component_name} component."
  )


@_call_component.register
def _apply_plugin(
    component: components.ApplyPluginT,
    backend: QualcommBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(
      input_model,
      output_model,
      backend.soc_manufacturer,
      backend.soc_model,
  )


@_call_component.register
def _aie_quantizer(
    component: components.AieQuantizerT,
    backend: QualcommBackend,
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
    unused_backend: QualcommBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(input_model, output_model, [])
