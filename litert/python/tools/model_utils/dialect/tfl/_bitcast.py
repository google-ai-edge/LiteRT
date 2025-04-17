"""tfl.bitcast operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.bitcast")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class BitcastOp(core.MlirOpBase):
  """Bitcast operator.

  Bitcasts a tensor from one type to another.
  """

  name = "tfl.bitcast"

  input = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    result_types = [result_type]
    super().__init__(
        operands=[input],
        result_types=result_types,
        location=location,
        attributes={},
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(BitcastOp)
def bitcast(*args, **kwargs):
  return BitcastOp(*args, **kwargs).output
