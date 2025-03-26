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

"""Interfaces for specific components used in the LiteRt AOT flow."""

import abc
from typing import Any, TypeAlias

from ai_edge_litert.aot.core import types

QuantRecipe: TypeAlias = list[dict[str, Any]] | str


class AieQuantizerT(metaclass=abc.ABCMeta):
  """Interface for AIE quantizer components."""

  @property
  def component_name(self) -> str:
    return "aie_quantizer"

  @abc.abstractmethod
  def __call__(
      self,
      input_model: types.Model,
      output_model: types.Model,
      quantization_recipe: QuantRecipe | None = None,
      *args,
      **kwargs,
  ):
    pass


class ApplyPluginT(metaclass=abc.ABCMeta):
  """Interface for apply plugin components."""

  @property
  def default_err(self) -> str:
    # NOTE: Capture stderr from underlying binary.
    return "none"

  @property
  def component_name(self) -> str:
    return "apply_plugin"

  @abc.abstractmethod
  def __call__(
      self,
      input_model: types.Model,
      output_model: types.Model,
      soc_manufacturer: str,
      soc_model: str,
      *args,
      **kwargs,
  ):
    pass


class MlirTransformsT(metaclass=abc.ABCMeta):
  """Interface for MLIR transforms components."""

  @property
  def component_name(self) -> str:
    return "mlir_transforms"

  @abc.abstractmethod
  def __call__(
      self, input_model: types.Model, output_model: types.Model, *args, **kwargs
  ):
    pass
