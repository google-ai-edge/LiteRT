"""tfl.abs operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.abs")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class AbsOp(core.MlirOpBase):
  """Absolute value operator.

  Given a tensor x, this operation returns a tensor containing the absolute
  value of each element in x. For example, if x is an input element and y is
  an output element, this operation computes y = abs(x).
  """

  name = "tfl.abs"

  x = irdl.operand_def()
  y = irdl.result_def()

  # No attributes defined in the spec.

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
        attributes={},
    )

  def _infer_result_type(
      self,
      x: SSAValue | core.MlirOpBase,
  ):
    # The 'SameOperandsAndResultShape' trait implies the result type is the
    # same as the input type.
    return _utils.get_tensor_type(x)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(AbsOp)
def abs(*args, **kwargs):
  """Builder function for the tfl.abs operation."""
  return AbsOp(*args, **kwargs).y
