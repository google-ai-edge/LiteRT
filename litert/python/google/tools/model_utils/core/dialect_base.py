import abc

from mlir import ir
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

  def __init__(self, *args, location: ir.Location | None = None, **kwargs):
    super().__init__(*args, **kwargs)
    if not hasattr(self, "location") or location is not None:
      self.location = location

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
