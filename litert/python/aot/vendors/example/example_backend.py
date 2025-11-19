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
from typing import Any

from litert.python.aot.core import components
from litert.python.aot.core import types
from litert.python.aot.vendors import import_vendor


class ExampleTarget(types.Target):
  """Compilation target for the example backend."""

  def __init__(self, soc_manufacturer: str, soc_model: str):
    self.soc_manufacturer = soc_manufacturer
    self.soc_model = soc_model

  def __hash__(self) -> int:
    return hash((self.soc_manufacturer, self.soc_model))

  def __eq__(self, other) -> bool:
    return (
        self.soc_manufacturer == other.soc_manufacturer
        and self.soc_model == other.soc_model
    )

  def __repr__(self) -> str:
    return f"{self.soc_manufacturer}_{self.soc_model}"

  def flatten(self) -> dict[str, Any]:
    return {
        "soc_manufacturer": self.soc_manufacturer,
        "soc_model": self.soc_model,
    }

  @classmethod
  def backend_id(cls) -> str:
    return "example"


# Note this is not a real target so not auto-registered unless the module is
# imported.
@import_vendor.register_backend
class ExampleBackend(types.Backend):
  """Backend implementation for the example compiler plugin."""

  def __init__(self, config: types.Config):
    super().__init__(config)
    self._compilation_config = config.get("compilation_config", None)

  @classmethod
  def target_(cls) -> ExampleTarget:
    return ExampleTarget("ExampleSocManufacturer", "ExampleSocModel")

  @property
  def target(self) -> ExampleTarget:
    return self.target_()

  @classmethod
  def soc_manufacturer(cls) -> str:
    return cls.target_().soc_manufacturer

  @classmethod
  def soc_model(cls) -> str:
    return cls.target_().soc_model

  @classmethod
  def id(cls) -> str:
    return "example"

  @property
  def target_id(self) -> str:
    return ""

  @property
  def shared_pass_names(self) -> list[str]:
    return ["example-pass"]

  @classmethod
  def create(cls, config: types.Config) -> "ExampleBackend":
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
      backend.soc_manufacturer,
      backend.soc_model,
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
