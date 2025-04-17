import jax.numpy as jnp
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.sub")
@core.overload_cls_attrs
@irdl_op_definition
class SubOp(core.MlirOpBase):
  """Subtraction operator

  Element-wise subtraction operation.
  """

  name = "tfl.sub"

  lhs = operand_def()
  rhs = operand_def()
  fused_activation_function = opt_attr_def(mlir.StringAttr)
  output = result_def()

  def __init__(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      location=None,
  ):
    lhs = SSAValue.get(lhs)
    rhs = SSAValue.get(rhs)
    fused_activation_function = _utils.to_str(fused_activation_function)
    result_types = [
        result_type
        or self._infer_result_type(lhs, rhs, fused_activation_function)
    ]
    super().__init__(
        operands=[lhs, rhs],
        result_types=result_types,
        location=location,
        attributes={
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
        },
    )

  def _infer_result_type(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
      fused_activation_function: str | mlir.StringAttr,
  ):
    lty = _utils.get_tensor_type(lhs)
    rty = _utils.get_tensor_type(rhs)
    if lty.element_type != rty.element_type:
      raise ValueError(
          f"Element types of lhs and rhs do not match: {lty.element_type} !="
          f" {rty.element_type}"
      )
    return mlir.RankedTensorType(
        jnp.broadcast_shapes(lty.shape, rty.shape), lty.element_type
    )

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        )
    )


def sub(
    lhs: SSAValue | core.MlirOpBase,
    rhs: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    fused_activation_function: str | mlir.StringAttr = "NONE",
    location=None,
):
  """Subtraction operator

  Element-wise subtraction operation.
  """
  return SubOp(
      lhs,
      rhs,
      result_type=result_type,
      fused_activation_function=fused_activation_function,
      location=location,
  ).output
