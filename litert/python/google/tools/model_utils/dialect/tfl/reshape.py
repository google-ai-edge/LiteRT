from mlir import ir
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _const

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.reshape")
@irdl_op_definition
class ReshapeOp(core.MlirOpBase):
  name = "tfl.reshape"

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
      if (
          not isinstance(shape.owner, core.MlirOpBase)
          or not isinstance(input.type, mlir.RankedTensorType)
          or shape.owner.name != ConstantOp.name
      ):
        raise ValueError(
            "result_type must be specified when shape is not from a const op"
        )
      result_type = mlir.RankedTensorType(
          shape.owner.numpy().astype(np.int32),
          input.type.element_type,
      )

    super().__init__(
        operands=[input, shape],
        result_types=[result_type],
        location=location,
    )

  @property
  def shape(self):
    owner = self._shape.owner
    if not isinstance(owner, Operation) or owner.name != ConstantOp.name:
      return self._shape
    return owner.numpy().tolist()


def reshape(input, shape, *args, **kwargs):
  if not isinstance(shape, (SSAValue, core.MlirOpBase)):
    shape = ConstantOp(shape)
  return ReshapeOp(input, shape, *args, **kwargs).output
