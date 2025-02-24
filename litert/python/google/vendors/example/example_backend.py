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

import functools
from typing import Self

from google3.third_party.odml.litert.litert.python.google.core import components
from google3.third_party.odml.litert.litert.python.google.core import types


class ExampleBackend(types.Backend):
  """Backend implementation for the example compiler plugin."""

  @classmethod
  def soc_manufacturer(cls) -> str:
    return "ExampleSocManufacturer"

  @classmethod
  def soc_model(cls) -> str:
    return "ExampleSocModel"

  @classmethod
  def id(cls) -> str:
    return "example"

  @property
  def shared_pass_names(self) -> list[str]:
    return ["example-pass"]

  @classmethod
  def create(cls, config: types.Config) -> Self:
    if config.get("backend_id", "") != cls.id():
      raise ValueError("Invalid backend id")
    return cls(config)

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
    backend: ExampleBackend,
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
    backend: ExampleBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(
      input_model,
      output_model,
      backend.soc_manufacturer(),
      backend.soc_model(),
  )


@_call_component.register
def _aie_quantizer(
    component: components.AieQuantizerT,
    unused_backend: ExampleBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(
      input_model,
      output_model,
  )


@_call_component.register
def _mlir_transforms(
    component: components.MlirTransformsT,
    backend: ExampleBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(input_model, output_model, backend.shared_pass_names)
