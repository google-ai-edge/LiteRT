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
"""Base classes for ModelUtils dialects."""

import abc
import functools
from typing import Any, Sequence

from litert.python.mlir import ir
import xdsl.irdl
import xdsl.printer

from . import _diagnostic


__all__ = [
    "overload_cls_attrs",
    "register_mlir_transform",
    "mlir_transforms",
    "MlirOpBase",
    "MlirAttributeBase",
    "MlirTypeBase",
]


def overload_cls_attrs(cls):
  new_attrs = {}
  if hasattr(cls, "overload_cls_attrs"):
    new_attrs = {**new_attrs, **cls.overload_cls_attrs()}

  new_attrs = {**cls.__dict__, **new_attrs}
  if "overload_cls_attrs" in new_attrs:
    new_attrs.pop("overload_cls_attrs")
  return type.__new__(type(cls), cls.__name__, cls.__mro__, new_attrs)


setattr(ir.Location, "__deepcopy__", lambda self, memo: self)


class MlirOpBase(xdsl.irdl.IRDLOperation):
  """Base class for all ModelUtils operations."""

  def __init__(self, *args, location: ir.Location | None = None, **kwargs):
    super().__init__(*args, **kwargs)
    if not hasattr(self, "location") or location is not None:
      self.location = location

  @classmethod
  @functools.wraps(xdsl.irdl.IRDLOperation.build)
  def build(
      cls: "MlirOpBase",
      *,
      operands: Sequence[Any] | None = None,
      result_types: Sequence[Any] | None = None,
      unflatten_variadics: bool = True,
      **kwargs
  ):
    """IRDLOperation.build wrapper that unflattens variadic operands and results."""
    if not unflatten_variadics:
      return super().build(
          operands=operands,
          result_types=result_types,
          **kwargs,
      )

    def _is_variadic_prepared(objs):
      """Checks user has prepared the inputs for variadic defs."""
      if not isinstance(objs, (list, tuple)):
        # If it is not a list or tuple, pass the objects as is.
        return True
      for obj in objs:
        if isinstance(obj, (list, tuple)):
          # If any of the objects are lists or tuples, the caller may have
          # prepared the inputs for variadic defs.
          return True
      return False

    op_def = cls.get_irdl_definition()

    if _is_variadic_prepared(operands):
      variadic_prepared_operands = operands
    else:
      variadic_prepared_operands = []
      for i, (_, operand_def) in enumerate(op_def.operands):
        if isinstance(operand_def, xdsl.irdl.VarOperandDef):
          # Assumes the rest of the operands belong to the first var operand.
          variadic_prepared_operands.append(operands[i:])
          break
        else:
          if i >= len(operands):
            break
          variadic_prepared_operands.append(operands[i])

    if _is_variadic_prepared(result_types):
      variadic_prepared_result_types = result_types
    else:
      variadic_prepared_result_types = []
      for i, (_, result_def) in enumerate(op_def.results):
        if isinstance(result_def, xdsl.irdl.VarResultDef):
          # Assumes the rest of the result types belong the first var result.
          variadic_prepared_result_types.append(result_types[i:])
          break
        else:
          if i >= len(result_types):
            break
          variadic_prepared_result_types.append(result_types[i])

    return super().build(
        operands=variadic_prepared_operands,
        result_types=variadic_prepared_result_types,
        **kwargs,
    )

  @classmethod
  def parse(cls, *args, **kwargs):
    raise NotImplementedError("MlirOp does not implement xDSL parse method.")

  def emit_error(
      self,
      message: str,
      exception_type,
      underlying_error,
  ):
    diag = _diagnostic.OpFocusDiagnostic()
    diag.add_message(self, message)
    diag.raise_exception(message, self, exception_type, underlying_error)


class MlirAttributeBase(abc.ABC):
  """Base class for all ModelUtils attributes."""

  @classmethod
  def parse_parameter(cls, parser):
    raise NotImplementedError()

  def print_parameter(self, printer: xdsl.printer.Printer) -> None:
    content = str(self.to_mlir())
    if len(content) > 200:
      content = content[:200] + "..."
    printer.print_string(content)

  @classmethod
  def from_mlir(cls, mlir_attribute: ir.Attribute):
    raise NotImplementedError()

  def to_mlir(self):
    raise NotImplementedError()


class MlirTypeBase:
  """Base class for all ModelUtils types."""

  @classmethod
  def from_mlir(cls, mlir_type: ir.Type):
    raise NotImplementedError()

  def to_mlir(self):
    raise NotImplementedError()

  @classmethod
  def parse_parameter(cls, parser):
    raise NotImplementedError()

  def print_parameter(self, printer: xdsl.printer.Printer) -> None:
    content = str(self.to_mlir())
    if len(content) > 200:
      content = content[:200] + "..."
    printer.print_string(content)


mlir_transforms = {}


def register_mlir_transform(key: str | ir.Attribute | ir.Type | None = None):
  """Register transformation from an MLIR type to a ModelUtils object."""

  def reg(cls: MlirOpBase | MlirAttributeBase | MlirTypeBase):
    nonlocal key
    if isinstance(cls, MlirOpBase):
      key = key if key is not None else cls.name
    if isinstance(cls, MlirAttributeBase):
      assert isinstance(key, ir.Attribute)
    if isinstance(cls, MlirTypeBase):
      assert isinstance(key, ir.Type)

    mlir_transforms[key] = cls
    return cls

  return reg
