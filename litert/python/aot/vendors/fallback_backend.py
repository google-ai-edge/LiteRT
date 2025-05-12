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

"""A Fallback backend for LITERT."""

import functools
from typing import Any

from litert.python.aot.core import components
from litert.python.aot.core import types


class FallbackTarget(types.Target):
  """A virtual Compilation target."""

  def __hash__(self) -> int:
    return hash(self.backend_id())

  def __eq__(self, other: types.Target) -> bool:
    return self.backend_id() == other.backend_id()

  def __repr__(self) -> str:
    return f"{self.backend_id()}"

  @classmethod
  def backend_id(cls) -> str:
    return "fallback"

  def flatten(self) -> dict[str, Any]:
    return {"backend_id": self.backend_id()}


class FallbackBackend(types.Backend):
  """Fallback backend for LITERT."""

  @property
  def target(self) -> FallbackTarget:
    return FallbackTarget()

  @property
  def target_id(self) -> str:
    return repr(self.target)

  @classmethod
  def id(cls) -> str:
    return "fallback"

  @classmethod
  def create(cls, config: types.Config) -> "FallbackBackend":
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
    backend: FallbackBackend,
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
    backend: FallbackBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  """A no-op component that just copies the input model to the output model."""
  del component, backend
  if input_model.in_memory:
    output_model.set_bytes(input_model.model_bytes)
  else:
    output_model.set_path(input_model.path)


@_call_component.register
def _aie_quantizer(
    component: components.AieQuantizerT,
    backend: FallbackBackend,
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
    unused_backend: FallbackBackend,
    input_model: types.Model,
    output_model: types.Model,
):
  return component(input_model, output_model, [])
