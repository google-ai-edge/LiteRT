import abc
from typing import Literal, Sequence, cast, final
from jax import tree_util
from mlir import ir
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *
from xdsl.printer import Printer
from .. import core


@final
@irdl_op_definition
class MlirOp(core.MlirOpBase):
  name = "__dummy__"

  _var_operand = var_operand_def()
  _var_result = var_result_def()
  _var_region = var_region_def()

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
    operands, _ = tree_util.tree_flatten(operands)
    result_types, _ = tree_util.tree_flatten(result_types)
    regions, _ = tree_util.tree_flatten(regions)
    return super().build(
        operands=[operands],
        result_types=[result_types],
        attributes=attributes,
        successors=successors,
        regions=[regions],
    )


@final
@irdl_attr_definition
class MlirAttribute(core.MlirAttributeBase, Data[Any]):
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
@irdl_attr_definition
class MlirType(
    core.MlirTypeBase,
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
):
  name = "!"
  _irty: ParameterDef[MlirAttribute]

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


@irdl_attr_definition
class _UnknownType(
    core.MlirTypeBase,
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
):
  name = "unknown"


UNKNOWN_TYPE = _UnknownType()

# ============== builtins ================


@core.register_mlir_transform(ir.StringAttr)
@irdl_attr_definition
class StringAttr(core.MlirAttributeBase, Data[str]):
  name = "string"

  def __init__(self, value: Union[str, "StringAttr"]):
    while isinstance(value, StringAttr):
      value = value.data
    assert isinstance(value, str)
    super().__init__(value)

  @classmethod
  def from_mlir(cls, attr: ir.StringAttr):
    return StringAttr(attr.value)

  def to_mlir(self):
    return ir.StringAttr.get(self.data)

  def print_parameter(self, printer: Printer) -> None:
    printer.print_string(f'"{self.data}"')

  @classmethod
  def op_attribute_accessor(cls, attribute_name: str, doc=None):
    def fget(self: IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: IRDLOperation, value: Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, str):
        self.attributes[attribute_name] = cls(value)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.BoolAttr)
@irdl_attr_definition
class BoolAttr(core.MlirAttributeBase, Data[bool]):
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
    def fget(self: IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: IRDLOperation, value: Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, bool):
        self.attributes[attribute_name] = cls(value)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.IntegerAttr)
@irdl_attr_definition
class IntAttr(core.MlirAttributeBase, Data[int]):
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
    def fget(self: IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: IRDLOperation, value: Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, int):
        self.attributes[attribute_name] = cls(value, *args, **kwargs)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


IntegerAttr = IntAttr


@core.register_mlir_transform(ir.FloatAttr)
@irdl_attr_definition
class FloatAttr(core.MlirAttributeBase, Data[float]):
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
    def fget(self: IRDLOperation):
      return cast(cls, self.attributes[attribute_name]).data

    def fset(self: IRDLOperation, value: Attribute | str | None):
      if isinstance(value, cls):
        self.attributes[attribute_name] = value
      elif isinstance(value, float):
        self.attributes[attribute_name] = cls(value, *args, **kwargs)
      elif value is None:
        self.attributes.pop(attribute_name)
      else:
        raise ValueError(f"Unsupported attribute value: {value}")

    def fdel(self: IRDLOperation):
      self.attributes.pop(attribute_name)

    return property(fget, fset, fdel, doc)


@core.register_mlir_transform(ir.DenseIntElementsAttr)
@core.register_mlir_transform(ir.DenseFPElementsAttr)
@core.register_mlir_transform(ir.DenseElementsAttr)
@irdl_attr_definition
class DenseElementsAttr(core.MlirAttributeBase, Data[ir.DenseElementsAttr]):
  name = "dense"

  def __init__(
      self,
      data: ir.DenseElementsAttr | np.ndarray | np.generic | list | tuple,
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

  def numpy(self):
    return core.pybind.get_elements_attr_buffer(self.data)

  def print_parameter(self, printer: Printer) -> None:
    data_str = np.array2string(
        self.numpy().flatten(),
        threshold=16,
        max_line_width=np.inf,
        separator=", ",
    )
    printer.print_string(data_str)


@core.register_mlir_transform(ir.RankedTensorType)
@irdl_attr_definition
class RankedTensorType(
    core.MlirTypeBase,
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
):
  name = "tensor"
  _shape: ParameterDef[DenseElementsAttr]
  _element_type: ParameterDef[StringAttr]

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
@irdl_op_definition
class ModuleOp(core.MlirOpBase):
  name = "builtin.module"

  sym_name = opt_attr_def(StringAttr)
  body = region_def("single_block")

  def __init__(
      self,
      ops: list[Operation] | Region,
      attributes: Mapping[str, Attribute] | None = None,
  ):
    if attributes is None:
      attributes = {}
    if isinstance(ops, Region):
      region = ops
    else:
      region = Region(Block(ops))
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
    from litert.python.google.tools.model_utils import passes

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

  def __instancecheck__(cls, instance):
    return isinstance(instance, SSAValue) and isinstance(
        instance.type, RankedTensorType
    )
