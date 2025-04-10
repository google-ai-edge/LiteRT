from mlir import ir
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.broadcast_to")
@core.overload_cls_attrs
@irdl_op_definition
class BroadcastToOp(core.MlirOpBase):
  name = "tfl.broadcast_to"

  input = operand_def()
  _shape = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      shape: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      location: ir.Location | None = None,
  ):
    input = SSAValue.get(input)
    shape = SSAValue.get(shape)

    if result_type is None:
      result_type = self._infer_result_type(input, shape)

    super().__init__(
        operands=[input, shape],
        result_types=[result_type],
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      shape: SSAValue | core.MlirOpBase,
  ):
    input = SSAValue.get(input)
    shape = SSAValue.get(shape)
    if (
        not isinstance(shape.owner, core.MlirOpBase)
        or not isinstance(input.type, mlir.RankedTensorType)
        or shape.owner.name != ConstantOp.name
    ):
      raise ValueError(
          "result_type must be specified when shape is not from a const op"
      )
    return mlir.RankedTensorType(
        shape.owner.numpy().tolist(), input.type.element_type
    )

  @property
  def shape(self):
    owner = self._shape.owner
    if not isinstance(owner, Operation) or owner.name != ConstantOp.name:
      return self._shape
    return owner.numpy().tolist()


def broadcast_to(input, shape, *args, **kwargs):
  if isinstance(shape, (list, tuple, np.ndarray)):
    shape = ConstantOp(shape)
  return BroadcastToOp(input, shape, *args, **kwargs).output
