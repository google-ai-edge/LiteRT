# Copyright 2025 Google LLC.
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
"""Attribute definition for tfl const_bytes."""

from typing import Union

from litert.python.mlir import ir
import xdsl
from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

Printer = xdsl.printer


# TODO: Implement core.register_mlir_transform
@irdl.irdl_attr_definition
class ConstBytesAttr(core.MlirAttributeBase, irdl.Data[bytes]):
  """A string attribute representation of compiled bytes."""

  name = "bytes"

  def __init__(self, value: Union[bytes, "ConstBytesAttr"]):
    while isinstance(value, ConstBytesAttr):
      value = value.data
    assert isinstance(value, bytes)
    super().__init__(value)

  @classmethod
  def from_mlir(cls, attr):
    raise NotImplementedError()

  def _hex(self):
    data = self.data.hex()
    return "0x" + data

  def to_mlir(self):
    return ir.Attribute.parse(f'#tfl<const_bytes: "{self._hex()}">')

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(self._hex()[:100])
