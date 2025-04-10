from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.average_pool_2d")
@core.overload_cls_attrs
@irdl_op_definition
class AveragePool2DOp(core.MlirOpBase):
  """_Average_pool2d operator

  Performs average-pooling operation on input.
  """

  name = "tfl.average_pool_2d"

  input = operand_def()
  filter_height = opt_attr_def(mlir.IntegerAttr)
  filter_width = opt_attr_def(mlir.IntegerAttr)
  padding = opt_attr_def(mlir.StringAttr)
  stride_h = opt_attr_def(mlir.IntegerAttr)
  stride_w = opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = opt_attr_def(mlir.StringAttr)
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      filter_height: int | mlir.IntegerAttr,
      filter_width: int | mlir.IntegerAttr,
      padding: str | mlir.StringAttr,
      stride_h: int | mlir.IntegerAttr,
      stride_w: int | mlir.IntegerAttr,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      location=None,
  ):
    input = SSAValue.get(input)
    filter_height = mlir.IntegerAttr(filter_height)
    filter_width = mlir.IntegerAttr(filter_width)
    padding = mlir.StringAttr(padding)
    stride_h = mlir.IntegerAttr(stride_h)
    stride_w = mlir.IntegerAttr(stride_w)
    fused_activation_function = _utils.to_str(fused_activation_function)

    result_types = [
        result_type
        or self._infer_result_type(
            input,
            filter_height,
            filter_width,
            padding,
            stride_h,
            stride_w,
            fused_activation_function,
        )
    ]

    super().__init__(
        operands=[input],
        result_types=result_types,
        attributes={
            "filter_height": filter_height,
            "filter_width": filter_width,
            "padding": padding,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
        },
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      filter_height: int | mlir.IntegerAttr,
      filter_width: int | mlir.IntegerAttr,
      padding: str | mlir.StringAttr,
      stride_h: int | mlir.IntegerAttr,
      stride_w: int | mlir.IntegerAttr,
      fused_activation_function: str | mlir.StringAttr,
  ):
    input_type = _utils.get_tensor_type(input)
    input_shape = input_type.shape

    filter_height = _utils.to_int(filter_height)
    filter_width = _utils.to_int(filter_width)
    padding = _utils.to_str(padding)
    stride_h = _utils.to_int(stride_h)
    stride_w = _utils.to_int(stride_w)

    if len(input_shape) != 4:
      raise ValueError("Input must be a 4D tensor.")

    batch_size, input_height, input_width, channels = input_shape

    if padding == "SAME":
      output_height = (input_height + stride_h - 1) // stride_h
      output_width = (input_width + stride_w - 1) // stride_w
    elif padding == "VALID":
      output_height = (input_height - filter_height + stride_h) // stride_h
      output_width = (input_width - filter_width + stride_w) // stride_w
    else:
      raise ValueError(f"Invalid padding type: {padding}")

    return mlir.RankedTensorType(
        [batch_size, output_height, output_width, channels],
        input_type.element_type,
    )

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        filter_height=mlir.IntegerAttr.op_attribute_accessor("filter_height"),
        filter_width=mlir.IntegerAttr.op_attribute_accessor("filter_width"),
        padding=mlir.StringAttr.op_attribute_accessor("padding"),
        stride_h=mlir.IntegerAttr.op_attribute_accessor("stride_h"),
        stride_w=mlir.IntegerAttr.op_attribute_accessor("stride_w"),
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        ),
    )


def average_pool_2d(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    filter_height: int | mlir.IntegerAttr,
    filter_width: int | mlir.IntegerAttr,
    padding: str | mlir.StringAttr,
    stride_h: int | mlir.IntegerAttr,
    stride_w: int | mlir.IntegerAttr,
    fused_activation_function: str | mlir.StringAttr = "NONE",
    location=None,
):
  """_Average_pool2d operator

  Performs average-pooling operation on input.
  """
  return AveragePool2DOp(
      input,
      result_type=result_type,
      filter_height=filter_height,
      filter_width=filter_width,
      padding=padding,
      stride_h=stride_h,
      stride_w=stride_w,
      fused_activation_function=fused_activation_function,
      location=location,
  ).output
