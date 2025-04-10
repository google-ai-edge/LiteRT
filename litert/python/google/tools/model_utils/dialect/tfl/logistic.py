from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.logistic")
@core.overload_cls_attrs
@irdl_op_definition
class LogisticOp(core.MlirOpBase):
  name = "tfl.logistic"

  x = operand_def()
  y = result_def()

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    x = SSAValue.get(x)

    result_types = [result_type or self._infer_result_type(x)]

    super().__init__(
        operands=[x],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      x: SSAValue | core.MlirOpBase,
  ):
    x_type = _utils.get_tensor_type(x)

    return mlir.RankedTensorType(x_type.shape, x_type.element_type)


def logistic(
    x: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  return LogisticOp(x, result_type=result_type, location=location).y
