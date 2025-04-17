"""tfl.bitwise_xor operation definition."""

import jax.numpy as jnp
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.bitwise_xor")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class BitwiseXorOp(core.MlirOpBase):
  """Bitwise Xor operator.

  Elementwise computes the bitwise XOR of lhs and rhs.
  """

  name = "tfl.bitwise_xor"

  lhs = irdl.operand_def()
  rhs = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    lhs = SSAValue.get(lhs)
    rhs = SSAValue.get(rhs)
    result_types = [result_type or self._infer_result_type(lhs, rhs)]
    super().__init__(
        operands=[lhs, rhs],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
  ):
    lty = _utils.get_tensor_type(lhs)
    rty = _utils.get_tensor_type(rhs)

    # SameOperandsAndResultElementType trait implies element types must match.
    if lty.element_type != rty.element_type:
      raise ValueError(
          "Element types of lhs and rhs do not match:"
          f" {lty.element_type} != {rty.element_type}"
      )

    # ResultsBroadcastableShape trait implies output shape is broadcasted shape.
    try:
      output_shape = jnp.broadcast_shapes(lty.shape, rty.shape)
    except ValueError as e:
      raise ValueError(
          f"Shapes are not broadcastable: {lty.shape} vs {rty.shape}"
      ) from e

    # Result element type is the same as operands.
    return mlir.RankedTensorType(output_shape, lty.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(BitwiseXorOp)
def bitwise_xor(*args, **kwargs):
  return BitwiseXorOp(*args, **kwargs).output
