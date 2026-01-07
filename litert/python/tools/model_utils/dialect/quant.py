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
"""Quant dialect definitions."""

import abc
import collections
from litert.python.mlir import ir
from litert.python.mlir.dialects import quant as ir_quant_d
from litert.python.tools.model_utils import core


class QuantizedTypeBase(collections.UserString[str], abc.ABC):
  """Base class for quantized types.

  This class inherits from collections.UserString[str] so that it can be
  compatible with the element_type attribute of RankedTensorType.
  """

  def __init__(self):
    # pylint: disable=super-init-not-called
    # Trick: avoid calling super().__init__() to avoid creating a UserString
    # object with an empty string. QuantizedTypeBase.data is supposed to be
    # a getter overridden by the subclasses.
    pass

  @property
  def data(self) -> str:
    ...

  def to_mlir(self) -> ir.Type:
    """Converts the quantized type to an MLIR IR type."""
    return ir.Type.parse(self.data)


@core.register_mlir_transform(ir_quant_d.UniformQuantizedType)
class UniformQuantizedType(QuantizedTypeBase):
  """MLIR UniformQuantizedType (per-layer/per-tensor quantization)."""

  def __init__(
      self,
      storage_type: str,
      expressed_type: str,
      scale: float,
      zero_point: int,
      storage_type_min: int | None = None,
      storage_type_max: int | None = None,
  ):
    super().__init__()
    self.storage_type = storage_type
    self.expressed_type = expressed_type
    self.scale = scale
    self.zero_point = zero_point
    self.storage_type_min = storage_type_min
    self.storage_type_max = storage_type_max

  @classmethod
  def from_mlir(cls, ir_type: ir_quant_d.UniformQuantizedType):
    return cls(
        storage_type=str(ir_type.storage_type),
        expressed_type=str(ir_type.expressed_type),
        scale=ir_type.scale,
        zero_point=ir_type.zero_point,
        storage_type_min=ir_type.storage_type_min,
        storage_type_max=ir_type.storage_type_max,
    )

  @property
  def data(self) -> str:
    """MLIR serialization of the UniformQuantized type.

    `!quant.uniform` `<`
      storedType (`<` storageMin `:` storageMax `>`)? `:`
      expressedType `,`
      scale (`:` zeroPoint)?
    `>`
    """
    type_str = "!quant.uniform<"

    type_str += str(self.storage_type)
    if self.storage_type_min is not None or self.storage_type_max is not None:
      if self.storage_type_min is None or self.storage_type_max is None:
        raise ValueError(
            "Both storage_type_min and storage_type_max must be specified."
        )
      type_str += f"<{self.storage_type_min}:{self.storage_type_max}>"

    type_str += ":"
    type_str += str(self.expressed_type)

    type_str += ","
    type_str += str(self.scale)
    if self.zero_point is not None:
      type_str += ":"
      type_str += str(self.zero_point)
    type_str += ">"
    return type_str


@core.register_mlir_transform(ir_quant_d.UniformQuantizedPerAxisType)
class UniformQuantizedPerAxisType(QuantizedTypeBase):
  """MLIR UniformQuantizedPerAxisType (per-axis/per-channel quantization)."""

  def __init__(
      self,
      storage_type: str,
      expressed_type: str,
      scales: list[float],
      zero_points: list[int],
      quantized_dimension: int,
      storage_type_min: int | None = None,
      storage_type_max: int | None = None,
  ):
    super().__init__()
    self.storage_type = storage_type
    self.expressed_type = expressed_type
    self.scales = scales
    self.zero_points = zero_points
    self.quantized_dimension = quantized_dimension
    self.storage_type_min = storage_type_min
    self.storage_type_max = storage_type_max

  @classmethod
  def from_mlir(cls, ir_type: ir_quant_d.UniformQuantizedPerAxisType):
    return cls(
        storage_type=str(ir_type.storage_type),
        expressed_type=str(ir_type.expressed_type),
        scales=list(ir_type.scales),
        zero_points=list(ir_type.zero_points),
        quantized_dimension=ir_type.quantized_dimension,
        storage_type_min=ir_type.storage_type_min,
        storage_type_max=ir_type.storage_type_max,
    )

  @property
  def data(self) -> str:
    """MLIR serialization of the UniformQuantizedPerAxis type.

    `!quant.uniform` `<`
      storedType (`<` storageMin `:` storageMax `>`)? `:`
      expressedType `:`
      channelAxis `,`
      `{`
        scale0 (`:` zeroPoint0)? `,`
        scale1 (`:` zeroPoint1)? ...
      '}'
    `>`
    """
    type_str = "!quant.uniform<"
    type_str += str(self.storage_type)
    if self.storage_type_min is not None or self.storage_type_max is not None:
      if self.storage_type_min is None or self.storage_type_max is None:
        raise ValueError(
            "Both storage_type_min and storage_type_max must be specified."
        )
      type_str += f"<{self.storage_type_min}:{self.storage_type_max}>"

    type_str += ":"
    type_str += str(self.expressed_type)

    type_str += ":"
    type_str += str(self.quantized_dimension)

    type_str += ", {"
    if self.zero_points and len(self.scales) != len(self.zero_points):
      raise ValueError("The number of scales and zero_points must be the same.")

    for i, scale in enumerate(self.scales):
      type_str += f"{scale}"
      if self.zero_points:
        type_str += f":{self.zero_points[i]}"
      if i < len(self.scales) - 1:
        type_str += ","
    type_str += "}>"
    return type_str


# Alias for quant types.
UniformQuantizedPerLayerType = UniformQuantizedType
UniformQuantizedPerChannelType = UniformQuantizedPerAxisType
