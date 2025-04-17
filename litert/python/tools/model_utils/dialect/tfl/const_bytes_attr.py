"""Attribute definition for tfl const_bytes."""

from typing import Union

from mlir import ir
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
