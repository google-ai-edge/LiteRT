from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir


@core.register_mlir_transform("tfl.conv_2d")
@core.overload_cls_attrs
@irdl_op_definition
class Conv2DOp(core.MlirOpBase):
  name = "tfl.conv_2d"

  input = operand_def()
  filter = operand_def()
  bias = operand_def()
  output = result_def()

  dilation_h_factor = opt_attr_def(mlir.IntegerAttr)
  dilation_w_factor = opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = opt_attr_def(mlir.StringAttr)
  padding = opt_attr_def(mlir.StringAttr)
  stride_h = opt_attr_def(mlir.IntegerAttr)
  stride_w = opt_attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      filter: SSAValue | core.MlirOpBase,
      bias: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase,
      *,
      dilation_h_factor: int | mlir.IntegerAttr = 1,
      dilation_w_factor: int | mlir.IntegerAttr = 1,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      padding: str | mlir.StringAttr = "SAME",
      stride_h: int | mlir.IntegerAttr = 1,
      stride_w: int | mlir.IntegerAttr = 1,
      location=None,
  ):
    input = SSAValue.get(input)
    filter = SSAValue.get(filter)
    bias = SSAValue.get(bias)

    super().__init__(
        operands=[input, filter, bias],
        result_types=[result_type],
        attributes={
            "dilation_h_factor": mlir.IntegerAttr(dilation_h_factor),
            "dilation_w_factor": mlir.IntegerAttr(dilation_w_factor),
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
            "padding": mlir.StringAttr(padding),
            "stride_h": mlir.IntegerAttr(stride_h),
            "stride_w": mlir.IntegerAttr(stride_w),
        },
        location=location,
    )

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        dilation_h_factor=mlir.IntegerAttr.op_attribute_accessor(
            "dilation_h_factor"
        ),
        dilation_w_factor=mlir.IntegerAttr.op_attribute_accessor(
            "dilation_w_factor"
        ),
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        ),
        padding=mlir.StringAttr.op_attribute_accessor("padding"),
        stride_h=mlir.IntegerAttr.op_attribute_accessor("stride_h"),
        stride_w=mlir.IntegerAttr.op_attribute_accessor("stride_w"),
    )


def conv_2d(*args, **kwargs):
  return Conv2DOp(*args, **kwargs).output
