"""tfl.add operation definition."""

import jax.numpy as jnp
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.add")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class AddOp(core.MlirOpBase):
  name = "tfl.add"

  lhs = irdl.operand_def()
  rhs = irdl.operand_def()
  fused_activation_function = irdl.opt_attr_def(mlir.StringAttr)
  output = irdl.result_def()

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
    result_types = [result_type or self._infer_result_type(lhs, rhs)]
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


@_utils.op_builder_wraps(AddOp)
def add(*args, **kwargs):
  return AddOp(*args, **kwargs).output
