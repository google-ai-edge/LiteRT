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
"""MLIR builtin dialect definitions."""

import abc
from typing import Any, Literal, Sequence, Union, cast, final

from litert.python.mlir import ir
import numpy as np
import xdsl
from xdsl import irdl

from litert.python.tools.model_utils import core

SSAValue = irdl.SSAValue
Printer = xdsl.printer.Printer


def attribute_from_mlir(ir_attr: ir.Attribute):
  """Builds an ModelUtils attribute from an ir.Attribute.

  This function looks up the attribute class from the ModelUtils transforms
  registry. If the registry is not found, it creates a mlir.MlirAttribute.

  Args:
    ir_attr: The ir.Attribute object to build the ModelUtils attribute from.

  Returns:
    The ModelUtils attribute object.
  """
  attr_cls = core.mlir_transforms.get(type(ir_attr))
  if attr_cls is not None:
    return attr_cls.from_mlir(ir_attr)

  return MlirAttribute(ir_attr)


def type_from_mlir(ir_type: ir.Type):
  """Builds an ModelUtils type from an ir.Type.

  This function looks up the type class from the ModelUtils transforms
  registry. If the registry is not found, it creates a mlir.MlirType.

  Args:
    ir_type: The ir.Type object to build the ModelUtils type from.

  Returns:
    The ModelUtils type object.
  """
  type_cls = core.mlir_transforms.get(ir_type) or core.mlir_transforms.get(
      type(ir_type)
  )
  if type_cls is not None:
    return type_cls.from_mlir(ir_type)

  return MlirType(ir_type)


@final
@irdl.irdl_op_definition
class MlirOp(core.MlirOpBase):
  """Common class for MLIR ops which do not register MLIR transforms."""

  name = "__dummy__"

  _var_operand = irdl.var_operand_def()
  _var_result = irdl.var_result_def()
  _var_region = irdl.var_region_def()

  def __init__(
      self,
      name: str,
      operands,
      result_types,
      attributes=None,
      regions=None,
      location: ir.Location | None = None,
  ):
    super().__init__(
        operands=[operands],
        result_types=[result_types],
        attributes=attributes,
        regions=[regions],
    )
    self.name = name

  @classmethod
  def build(
      cls,
      *,
      operands=None,
      result_types=None,
      attributes=None,
      successors=None,
      regions=None,
  ):
    # Notmalize vardict operands and results to a list.
    operands = core.utils.tree_flatten(operands)
    result_types = core.utils.tree_flatten(result_types)
    regions = core.utils.tree_flatten(regions)
    return super().build(
        operands=[operands],
        result_types=[result_types],
        attributes=attributes,
        successors=successors,
        regions=[regions],
    )


@final
@irdl.irdl_attr_definition
class MlirAttribute(core.MlirAttributeBase, irdl.Data[Any]):
  """Common class for MLIR attributes which do not register MLIR transforms."""

  name = "#"

  def __init__(self, data):
    while isinstance(data, MlirAttribute):
      data = data.data
    super().__init__(data)

  def update(self, mlir_attr_str: str):
    self.data = self.data.type.parse(mlir_attr_str)

  def to_mlir(self):
    return self.data

  def __deepcopy__(self, memo):
    return MlirAttribute(self.data)


@final
@irdl.irdl_attr_definition
class MlirType(
    core.MlirTypeBase,
    irdl.Generic[irdl.AttributeCovT],
    xdsl.ir.ParametrizedAttribute,
    xdsl.ir.TypeAttribute,
):
  """Common class for MLIR types which do not register MLIR transforms."""

  name = "!"
  _irty: irdl.ParameterDef[MlirAttribute]

  def __init__(self, ir_type: ir.Type):
    super().__init__([MlirAttribute(ir_type)])

  @property
  def mlir_type(self):
    return self._irty.data

  @mlir_type.setter
  def mlir_type(self, mlir_type: ir.Type):
    self._irty.data = mlir_type

  def to_mlir(self):
    return self.mlir_type

  def __deepcopy__(self, memo):
    memo[self] = MlirType(self._irty)
    return memo[self]


@irdl.irdl_attr_definition
class _UnknownType(
    core.MlirTypeBase,
    irdl.Generic[irdl.AttributeCovT],
    xdsl.ir.ParametrizedAttribute,
    xdsl.ir.TypeAttribute,
):
  name = "unknown"


UNKNOWN_TYPE = _UnknownType()

# ============== builtins ================


@core.register_mlir_transform(ir.StringAttr)
@irdl.irdl_attr_definition
class StringAttr(core.MlirAttributeBase, irdl.Data[str]):
  """MLIR builtin StringAttr."""

  name = "string"

  def __init__(self, value: Union[str, "StringAttr"]):
    while isinstance(value, StringAttr):
      value = value.data
    assert isinstance(value, str)
    super().__init__(value)

  @classmethod
  def from_mlir(cls, attr: ir.StringAttr):
    try:
      return cls(attr.value)
    except RuntimeError:
      # MLIR StringAttr can contain raw bytes, which may cause RuntimeError
      # when converting to Python string.
      # We wrap the entire Attribute in MlirAttribute to avoid the error.
      return MlirAttribute(attr)

  def to_mlir(self):
    return ir.StringAttr.get(self.data)

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(f'"{self.data}"')

  @classmethod
  def op_attribute_accessor(cls, attribute_name: str, doc=None):
    def fget(self: irdl.IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: irdl.IRDLOperation, value: irdl.Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, str):
        self.attributes[attribute_name] = cls(value)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: irdl.IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.BoolAttr)
@irdl.irdl_attr_definition
class BoolAttr(core.MlirAttributeBase, irdl.Data[bool]):
  """MLIR builtin BoolAttr."""

  name = "bool"

  def __init__(self, value: Union[bool, "BoolAttr"]):
    while isinstance(value, BoolAttr):
      value = value.data
    assert isinstance(value, bool)
    super().__init__(value)

  @classmethod
  def from_mlir(cls, attr: ir.BoolAttr):
    return BoolAttr(attr.value)

  def to_mlir(self):
    return ir.BoolAttr.get(self.data)

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(str(self.data).lower())

  @classmethod
  def op_attribute_accessor(cls, attribute_name: str, doc=None):
    def fget(self: irdl.IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: irdl.IRDLOperation, value: irdl.Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, bool):
        self.attributes[attribute_name] = cls(value)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: irdl.IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.IntegerAttr)
@irdl.irdl_attr_definition
class IntAttr(core.MlirAttributeBase, irdl.Data[int]):
  """MLIR builtin IntegerAttr."""

  name = "int"

  def __init__(
      self,
      value: Union[int, "IntAttr"],
      width: int = 32,
      sign: Literal["signed", "signless", "unsiged"] = "signless",
      *,
      _type: str | ir.Type | None = None,
  ):
    if isinstance(value, IntAttr):
      super().__init__(value.data)
      self._type = value._type
      return

    if _type is None:
      if sign == "signless":
        _type = "i" + str(width)
      elif sign == "signed":
        _type = "si" + str(width)
      elif sign == "unsigned":
        _type = "ui" + str(width)
      else:
        raise ValueError(f"Unsupported integer type sign: {sign}")

    super().__init__(value)
    self._type = str(_type)

  @classmethod
  def from_mlir(cls, attr: ir.IntegerAttr):
    return cls(attr.value, _type=attr.type)

  def to_mlir(self):
    return ir.IntegerAttr.get(ir.Type.parse(self._type), self.data)

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(f"{self.data} : {self._type}")

  def __int__(self):
    return self.data

  @classmethod
  def op_attribute_accessor(
      cls, attribute_name: str, *args, doc=None, **kwargs
  ):
    def fget(self: irdl.IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: irdl.IRDLOperation, value: irdl.Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, int):
        self.attributes[attribute_name] = cls(value, *args, **kwargs)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: irdl.IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


IntegerAttr = IntAttr


@core.register_mlir_transform(ir.FloatAttr)
@irdl.irdl_attr_definition
class FloatAttr(core.MlirAttributeBase, irdl.Data[float]):
  """MLIR builtin FloatAttr."""

  name = "float"

  def __init__(
      self,
      value: Union[float, "FloatAttr"],
      width: int = 32,
      *,
      _type: str | ir.Type | None = None,
  ):
    if isinstance(value, FloatAttr):
      super().__init__(value.data)
      self._type = value._type
      return

    if _type is None:
      _type = "f" + str(width)

    super().__init__(value)
    self._type = str(_type)

  def __float__(self):
    return self.data

  @classmethod
  def from_mlir(cls, attr: ir.FloatAttr):
    return cls(attr.value, _type=attr.type)

  def to_mlir(self):
    return ir.FloatAttr.get(ir.Type.parse(self._type), self.data)

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(f"{self.data} : {self._type}")

  @classmethod
  def op_attribute_accessor(
      cls, attribute_name: str, *args, doc=None, **kwargs
  ):
    def fget(self: irdl.IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: irdl.IRDLOperation, value: irdl.Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, float):
        self.attributes[attribute_name] = cls(value, *args, **kwargs)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: irdl.IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.ArrayAttr)
@irdl.irdl_attr_definition
class ArrayAttr(
    core.MlirAttributeBase, irdl.Data[list[core.MlirAttributeBase]]
):
  """MLIR builtin ArrayAttr."""

  name = ""

  def __init__(self, value: Union[list[core.MlirAttributeBase], "ArrayAttr"]):
    if isinstance(value, ArrayAttr):
      super().__init__(list(value.data))
      return
    super().__init__(list(value))

  @classmethod
  def from_mlir(cls, array_attr: ir.ArrayAttr):
    data = []
    for attr in array_attr:
      data.append(attribute_from_mlir(attr))
    return cls(data)

  def to_mlir(self):
    return ir.ArrayAttr.get([attr.to_mlir() for attr in self.data])

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string("[")
    for i, attr in enumerate(self.data):
      if i > 0:
        printer.print_string(", ")
      printer.print(attr)
    printer.print_string("]")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    return self.data[index]

  def __setitem__(self, index: int, value: core.MlirAttributeBase):
    self.data[index] = value

  def __delitem__(self, index: int):
    del self.data[index]

  def __iter__(self):
    return iter(self.data)

  def __contains__(self, value: core.MlirAttributeBase):
    return value in self.data

  def append(self, value: core.MlirAttributeBase):
    self.data.append(value)

  def pop(self, index: int = -1) -> core.MlirAttributeBase:
    return self.data.pop(index)

  def insert(self, index: int, value: core.MlirAttributeBase):
    self.data.insert(index, value)


@core.register_mlir_transform(ir.DictAttr)
@irdl.irdl_attr_definition
class DictAttr(
    core.MlirAttributeBase, irdl.Data[dict[str, core.MlirAttributeBase]]
):
  """MLIR builtin DictionaryAttr."""

  name = ""

  def __init__(
      self,
      value: Union[dict[str, core.MlirAttributeBase], "DictAttr"],
  ):
    if isinstance(value, DictAttr):
      super().__init__(dict(value.data))
      return

    super().__init__(value)

  @classmethod
  def from_mlir(cls, dict_attr: ir.DictAttr):
    names = core.pybind.get_dictionary_attr_names(dict_attr)
    data = {}
    for name in names:
      attr = dict_attr[name]
      data[name] = attribute_from_mlir(attr)
    return cls(data)

  def to_mlir(self):
    return ir.DictAttr.get(
        {name: attr.to_mlir() for name, attr in self.data.items()}
    )

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string("{")
    for i, (name, attr) in enumerate(self.data.items()):
      if i > 0:
        printer.print_string(", ")
      printer.print_string(f"{name} = ")
      printer.print(attr)
    printer.print_string("}")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, key: str):
    return self.data[key]

  def __iter__(self):
    return iter(self.data)

  def __setitem__(self, key: str, value: core.MlirAttributeBase):
    self.data[key] = value

  def __delitem__(self, key: str):
    del self.data[key]

  def __contains__(self, key: str):
    return key in self.data

  def keys(self) -> list[str]:
    return list(self.data.keys())

  def items(self) -> list[tuple[str, core.MlirAttributeBase]]:
    return list(self.data.items())

  def get(self, key: str, default: core.MlirAttributeBase | None = None):
    return self.data.get(key, default)

  def values(self) -> list[core.MlirAttributeBase]:
    return list(self.data.values())


@core.register_mlir_transform(ir.DenseIntElementsAttr)
@core.register_mlir_transform(ir.DenseFPElementsAttr)
@core.register_mlir_transform(ir.DenseElementsAttr)
@irdl.irdl_attr_definition
class DenseElementsAttr(
    core.MlirAttributeBase, irdl.Data[ir.DenseElementsAttr]
):
  """MLIR builtin DenseElementsAttr.

  This class is also used to represent other variants of DenseElementsAttr, like
  DenseIntElementsAttr and DenseFPElementsAttr.
  """

  name = "dense"

  def __init__(
      self,
      data: (
          ir.DenseElementsAttr
          | np.ndarray
          | np.generic
          | list[int | float]
          | tuple[int | float, ...]
      ),
  ):
    while isinstance(data, DenseElementsAttr):
      data = data.data

    if not isinstance(data, ir.DenseElementsAttr):
      # !!WARNING!!
      # TFL friendly: use 32-bit values as defualt attribute and data type
      # Use DenseElementsAttr(np.array(..., dtype=np.xx64)) if 64-bit data
      # is desired.
      if isinstance(data, (int, float, list, tuple)):
        data = np.array(data)
        dtype_32bit = data.dtype.name.replace("64", "32")
        data = data.astype(np.dtype(dtype_32bit))

      data = self._numpy_to_attr(data)
    super().__init__(data)

  def __deepcopy__(self, memo):
    return DenseElementsAttr(self.data)

  @classmethod
  def from_mlir(cls, attr: ir.DenseElementsAttr):
    return DenseElementsAttr(attr)

  def to_mlir(self):
    return self.data

  def _numpy_to_attr(self, x: np.ndarray | np.generic):
    element_type = core.utils.dtype_to_ir_type(x.dtype)
    shape = x.shape
    if x.dtype == np.bool_:
      x = np.packbits(x, bitorder="little")  # type: ignore
    x = np.ascontiguousarray(x)
    attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)  # type: ignore
    return attr

  def _raw_bytes(self, offset: int = 0, size: int = -1) -> bytes:
    return core.pybind.get_dense_elements_attr_bytes(
        self.to_mlir(), offset, size
    )

  def _dtype(self) -> np.dtype:
    tensor_ty = self.data.type
    if not hasattr(tensor_ty, "element_type"):
      raise ValueError(f"element_type is not presented in: {tensor_ty}")
    return core.utils.ir_type_to_dtype(tensor_ty.element_type)

  def is_splat(self) -> bool:
    return self.data.is_splat

  def _raw_bytes_as_ndarray(
      self, byte_offset: int = 0, byte_size: int = -1
  ) -> np.ndarray:
    """Creates a copy of the attribute's data as a NumPy array."""
    tensor_ty = self.data.type
    if not isinstance(tensor_ty, ir.RankedTensorType):
      raise ValueError(
          "Only DenseElementsAttr in RankedTensorType is supported to convert"
          " to NumPy array."
      )

    dtype = core.utils.ir_type_to_dtype(tensor_ty.element_type)
    shape = tensor_ty.shape

    raw_data = self._raw_bytes(offset=byte_offset, size=byte_size)

    if dtype == np.dtype(np.bool_):
      numel = int(np.prod(shape))
      numel = min(byte_size, numel) if byte_size > 0 else numel
      numel = numel if not self.is_splat() else 1
      flat_arr = np.frombuffer(raw_data, dtype=np.uint8)
      flat_arr = np.unpackbits(flat_arr, bitorder="little")
      flat_arr = flat_arr[:numel].astype(dtype)
    else:
      flat_arr = np.frombuffer(raw_data, dtype=dtype)
    return flat_arr

  def _shape(self) -> list[int] | None:
    if not isinstance(self.data.type, ir.RankedTensorType):
      return None
    return list(self.data.type.shape)

  def numpy(self) -> np.ndarray:
    """Creates a copy of the attribute's data as a NumPy array."""
    flat_arr = self._raw_bytes_as_ndarray()
    if self.is_splat():
      flat_arr = np.repeat(flat_arr, np.prod(self._shape()))
    arr = flat_arr.reshape(self._shape())
    return arr

  def print_parameter(self, printer: Printer) -> None:
    np_array2string_threshold = 16

    data: np.ndarray
    if (
        not self._shape()
        or np.prod(self._shape()) < np_array2string_threshold * 2
    ):
      data = self.numpy()
    elif self.is_splat():
      data = np.repeat(
          self._raw_bytes_as_ndarray(),
          min(np.prod(self._shape()), np_array2string_threshold * 2),
      )
    else:
      # Assume all dtype size can divide 128.
      # Get at least 16 elements on both ends of the flattened array.
      # Overlapped and duplicated elements don't matter because the print
      # elements threshold is 16.
      byte_size = 128 * np_array2string_threshold
      prefix_arr = self._raw_bytes_as_ndarray(0, byte_size)
      suffix_arr = self._raw_bytes_as_ndarray(-byte_size, byte_size)
      data = np.concatenate([prefix_arr, suffix_arr], axis=0)

    data_str = np.array2string(
        data.flatten(),
        threshold=np_array2string_threshold,
        max_line_width=np.inf,
        separator=", ",
    )
    printer.print_string(data_str)


@core.register_mlir_transform(ir.RankedTensorType)
@irdl.irdl_attr_definition
class RankedTensorType(
    core.MlirTypeBase,
    irdl.Generic[irdl.AttributeCovT],
    xdsl.ir.ParametrizedAttribute,
    xdsl.ir.TypeAttribute,
):
  """MLIR builtin RankedTensorType."""

  name = "tensor"
  _shape: irdl.ParameterDef[DenseElementsAttr]
  _element_type: irdl.ParameterDef[StringAttr]

  def __init__(
      self,
      shape: np.ndarray | Sequence[int] | DenseElementsAttr,
      element_type: str | np.dtype | ir.Type | StringAttr,
  ):
    if not isinstance(element_type, StringAttr):
      if isinstance(element_type, np.dtype):
        element_type = core.utils.dtype_to_ir_type(element_type)
      element_type = StringAttr(str(element_type))

    if not isinstance(shape, DenseElementsAttr):
      if isinstance(shape, np.ndarray):
        shape = shape.flatten().tolist()
      # Assume negative dimensions are dynamic.
      # ??? Dynamic dim may be 0 when converting from flatbuffer, why?
      shape = [-1 if d < 0 else d for d in shape]
      shape = DenseElementsAttr(shape)

    super().__init__([shape, element_type])

  def __deepcopy__(self, memo):
    return RankedTensorType(self._shape, self._element_type)

  def clone(
      self,
      shape: np.ndarray | Sequence[int] | DenseElementsAttr | None = None,
      element_type: str | np.dtype | ir.Type | StringAttr | None = None,
  ):
    if shape is None:
      shape = self._shape
    if element_type is None:
      element_type = self._element_type
    return RankedTensorType(shape, element_type)

  @property
  def shape(self) -> list[int]:
    return self._shape.numpy().flatten().tolist()

  @property
  def rank(self) -> int:
    return len(self.shape)

  @property
  def element_type(self):
    return self._element_type.data

  @property
  def elty(self):
    return self.element_type

  @classmethod
  def from_mlir(cls, attr: ir.RankedTensorType):
    return RankedTensorType(attr.shape, str(attr.element_type))

  def _inner_str(self, abbreviated_type=False):
    shape = [str(d) if d >= 0 else "?" for d in self.shape]
    shape_str = "x".join(shape)
    type_str = str(self.element_type)
    if abbreviated_type:
      type_str = type_str[:100] + ("..." if len(type_str) > 100 else "")

    if not shape:
      return type_str
    return f"{shape_str}x{type_str}"

  def to_mlir(self):
    return ir.RankedTensorType.parse(f"tensor<{self._inner_str()}>")

  def print_parameters(self, printer: Printer) -> None:
    # for xDSL internal printer.
    printer.print_string(f"<{self._inner_str(abbreviated_type=True)}>")


@core.register_mlir_transform("builtin.module")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class ModuleOp(core.MlirOpBase):
  """MLIR builtin ModuleOp."""

  name = "builtin.module"

  sym_name = irdl.opt_attr_def(StringAttr)
  body = irdl.region_def("single_block")

  def __init__(
      self,
      ops: list[irdl.Operation] | irdl.Region,
      attributes: irdl.Mapping[str, irdl.Attribute] | None = None,
  ):
    if attributes is None:
      attributes = {}
    if isinstance(ops, irdl.Region):
      region = ops
    else:
      region = irdl.Region(irdl.Block(ops))
    super().__init__(regions=[region], attributes=attributes)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        syn_name=StringAttr.op_attribute_accessor("syn_name"),
    )

  @property
  def ops(self) -> list[core.MlirOpBase]:
    return list(self.body.ops)

  def replace_by(self, new_module: core.MlirOpBase):
    """Replace the module with the new module."""
    assert new_module.name == self.name
    self.attributes = new_module.attributes
    self.regions = new_module.regions

  def cleanup(self):
    """Run CSE and canonicalization passes to clean up the module."""
    # pylint: disable=g-import-not-at-top
    from litert.python.tools.model_utils import passes
    # pylint: enable=g-import-not-at-top

    passes.MlirPass("builtin.module(cse,canonicalize,cse)")(self)

  def print(self, printer: Printer) -> None:
    if self.attributes:
      printer.print_op_attributes(self.attributes)

    if not self.body.block.ops:
      # Do not print the entry block if the region has an empty block
      printer.print(" {\n")
      printer.print("}")
    else:
      printer.print(" ", self.body)


class SSARankedTensorValue(SSAValue, abc.ABC):

  @property
  def type(self) -> RankedTensorType:
    return super().type

  def __instancecheck__(self, instance):
    return isinstance(instance, SSAValue) and isinstance(
        instance.type, RankedTensorType
    )
