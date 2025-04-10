import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.fill")
@core.overload_cls_attrs
@irdl_op_definition
class FillOp(core.MlirOpBase):
  name = "tfl.fill"

  dims = operand_def()
  input = operand_def()
  result = result_def()

  def __init__(
      self,
      dims: SSAValue | core.MlirOpBase,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    dims = SSAValue.get(dims)
    input = SSAValue.get(input)
    result_types = [result_type or self._infer_result_type(dims, input)]

    super().__init__(
        operands=[dims, input],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      dims: SSAValue | core.MlirOpBase,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)
    dims_array = dims.owner.numpy()
    dims_list = [int(i) for i in dims_array.flatten()]

    return mlir.RankedTensorType(dims_list, input_type.element_type)


def fill(
    dims: SSAValue | core.MlirOpBase | np.ndarray | list[int] | tuple[int, ...],
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  if not isinstance(dims, (SSAValue, core.MlirOpBase)):
    dims = ConstantOp(np.array(dims, dtype=np.int32))
  return FillOp(dims, input, result_type=result_type, location=location).output
