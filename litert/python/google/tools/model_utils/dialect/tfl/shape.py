from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.shape")
@core.overload_cls_attrs
@irdl_op_definition
class ShapeOp(core.MlirOpBase):
  """Shape operator

  Returns the shape of a tensor.
  """

  name = "tfl.shape"

  input = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)

    result_types = [result_type or self._infer_result_type(input)]
    super().__init__(
        operands=[input],
        result_types=result_types,
        attributes={},
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)
    if not isinstance(input_type, mlir.RankedTensorType):
      raise ValueError("Input must be a ranked tensor.")

    rank = len(input_type.shape)
    return mlir.RankedTensorType([rank], "i32")


def shape(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Shape operator

  Returns the shape of a tensor.
  """
  return ShapeOp(input, result_type=result_type, location=location).output
