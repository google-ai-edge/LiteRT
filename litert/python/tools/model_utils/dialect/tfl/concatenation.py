from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.concatenation")
@core.overload_cls_attrs
@irdl_op_definition
class ConcatenationOp(core.MlirOpBase):
  name = "tfl.concatenation"

  values = var_operand_def()
  output = result_def()
  axis = opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = opt_attr_def(mlir.StringAttr)

  def __init__(
      self,
      values: list[SSAValue | core.MlirOpBase],
      *,
      axis: int | mlir.IntegerAttr,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      result_type: core.MlirTypeBase | None = None,
      location=None,
  ):
    values = [SSAValue.get(v) for v in values]
    axis = _utils.to_int(axis)
    fused_activation_function = _utils.to_str(fused_activation_function)

    if result_type is None:
      result_type = self._infer_result_type(
          values, axis, fused_activation_function
      )

    super().__init__(
        operands=[values],
        result_types=[result_type],
        attributes={
            "axis": mlir.IntegerAttr(axis),
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
        },
        location=location,
    )

  def _infer_result_type(
      self,
      values: list[SSAValue | core.MlirOpBase],
      axis: int | mlir.IntegerAttr,
      fused_activation_function: str | mlir.StringAttr,
  ):
    if not values:
      raise ValueError(
          "Cannot infer result type for concatenation with no input values."
      )

    axis = _utils.to_int(axis)
    value_types = [_utils.get_tensor_type(v) for v in values]

    element_type = value_types[0].element_type
    output_shape = list(value_types[0].shape)
    output_shape[axis] = 0
    for ty in value_types:
      if not isinstance(ty, mlir.RankedTensorType):
        raise ValueError(
            "Cannot infer result type when input values are not all"
            " RankedTensorType."
        )
      if ty.element_type != element_type:
        raise ValueError(
            "Cannot infer result type when input values have different element"
            " types."
        )
      input_shape = ty.shape
      if len(input_shape) != len(output_shape):
        raise ValueError(
            "Cannot infer result type when input values have different ranks."
        )
      for i, dim in enumerate(input_shape):
        if i != axis and dim != output_shape[i]:
          raise ValueError(
              "Cannot infer result type when input values have incompatible"
              " shapes along non-concatenation axes."
          )
      output_shape[axis] += input_shape[axis]
    return mlir.RankedTensorType(tuple(output_shape), element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        )
    )

  @property
  def concatenation_axis(self):
    return self.axis.value.item()


def concatenation(values: list, axis: int, *args, **kwargs):
  return ConcatenationOp(values, axis=axis, *args, **kwargs).output
