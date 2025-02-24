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

"""Basic types used in the LiteRt AOT flow."""

import abc
import pathlib
from typing import Any, Mapping, Protocol, Self, Type, TypeAlias


# An intermediate result of a component.
# NOTE: Consider variant<bytes, filepath> approach to replace pathlib.Path for
# intermediate results.
Model: TypeAlias = pathlib.Path


class Component(Protocol):
  """An arbitrary module in the AOT flow that inputs and outputs a Model.

  For example quantizer, graph rewriter, compiler plugin etc.
  """

  @property
  def component_name(self) -> str:
    ...

  def __call__(self, input_model: Model, output_model: Model, *args, **kwargs):
    ...


# A user provided configuration. This will contain all the information needed
# to select the proper backend and run components (e.g. quant recipe,
# backend id etc). Backends will validate and resolve configurations and are
# ultimately responsible deciding how to configure the components.
# NOTE: Consider a typed config approach (proto, data class, etc.)
Config: TypeAlias = Mapping[str, Any]


class Backend(metaclass=abc.ABCMeta):
  """A backend pertaining to a particular SoC vendor.

  Mainly responsible for resolving configurations and managing vendor specific
  resources (e.g. .so etc).
  """

  # NOTE: Only initialize through "create".
  def __init__(self, config: Config):
    self._config = config

  @classmethod
  @abc.abstractmethod
  def create(cls, config: Config) -> Self:
    pass

  @classmethod
  @abc.abstractmethod
  def id(cls) -> str:
    pass

  @property
  def config(self) -> Config:
    return self._config

  @abc.abstractmethod
  def call_component(
      self, input_model: Model, output_model: Model, component: Component
  ):
    pass


BackendT: TypeAlias = Type[Backend]
