from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.log_softmax")
@core.overload_cls_attrs
@irdl_op_definition
class LogSoftmaxOp(core.MlirOpBase):
  """Log softmax operator

  Computes element-wise log softmax activations with the following formula

  input - log(reduce_sum(exp(input), dim))
  """

  name = "tfl.log_softmax"

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
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)

    return mlir.RankedTensorType(input_type.shape, input_type.element_type)


def log_softmax(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Log softmax operator

  Computes element-wise log softmax activations with the following formula

  input - log(reduce_sum(exp(input), dim))
  """
  return LogSoftmaxOp(input, result_type=result_type, location=location).output
