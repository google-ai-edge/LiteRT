from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir


@core.register_mlir_transform("tfl.depthwise_conv_2d")
@core.overload_cls_attrs
@irdl_op_definition
class DepthwiseConv2DOp(core.MlirOpBase):
  name = "tfl.depthwise_conv_2d"

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
  depth_multiplier = opt_attr_def(mlir.IntegerAttr)

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
      depth_multiplier: int | mlir.IntegerAttr = 1,
      location=None,
  ):
    operands = map(SSAValue.get, [input, filter, bias])

    super().__init__(
        operands=operands,
        result_types=[result_type],
        location=location,
    )
    self.dilation_h_factor = dilation_h_factor
    self.dilation_w_factor = dilation_w_factor
    self.fused_activation_function = fused_activation_function
    self.padding = padding
    self.stride_h = stride_h
    self.stride_w = stride_w
    self.depth_multiplier = depth_multiplier

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
        depth_multiplier=mlir.IntegerAttr.op_attribute_accessor(
            "depth_multiplier"
        ),
    )


def depthwise_conv_2d(*args, **kwargs):
  return DepthwiseConv2DOp(*args, **kwargs).output
