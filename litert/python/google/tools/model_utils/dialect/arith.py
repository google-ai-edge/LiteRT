from mlir import ir
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir


@core.register_mlir_transform("arith.constant")
@irdl_op_definition
class ConstantOp(core.MlirOpBase):
  name = "arith.constant"

  value = attr_def(mlir.DenseElementsAttr)
  output = result_def()

  def __init__(
      self,
      value: mlir.DenseElementsAttr | np.ndarray | list | tuple,
      location: ir.Location | None = None,
  ):
    if not isinstance(value, mlir.DenseElementsAttr):
      value = mlir.DenseElementsAttr(value)

    super().__init__(
        result_types=[mlir.RankedTensorType.from_mlir(value.data.type)],
        attributes={"value": value},
        location=location,
    )

  def numpy(self):
    return self.value.numpy()


def constant(*args, **kwargs):
  return ConstantOp(*args, **kwargs).output


def const(*args, **kwargs):
  return ConstantOp(*args, **kwargs).output
