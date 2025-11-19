# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.max_pool_2d")
@core.overload_cls_attrs
@irdl_op_definition
class MaxPool2DOp(core.MlirOpBase):
  """Max Pool 2D op

  Performs max pool 2D on input.

  Inputs:
    inputs[0]: required: the input tensor
  """

  name = "tfl.max_pool_2d"

  input = operand_def()
  padding = opt_attr_def(mlir.StringAttr)
  stride_w = opt_attr_def(mlir.IntegerAttr)
  stride_h = opt_attr_def(mlir.IntegerAttr)
  filter_width = opt_attr_def(mlir.IntegerAttr)
  filter_height = opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = opt_attr_def(mlir.StringAttr)
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      padding: str | mlir.StringAttr,
      stride_w: int | mlir.IntegerAttr,
      stride_h: int | mlir.IntegerAttr,
      filter_width: int | mlir.IntegerAttr,
      filter_height: int | mlir.IntegerAttr,
      fused_activation_function: str | mlir.StringAttr = "NONE",
      location=None,
  ):
    input = SSAValue.get(input)
    padding = mlir.StringAttr(padding)
    stride_w = mlir.IntegerAttr(stride_w)
    stride_h = mlir.IntegerAttr(stride_h)
    filter_width = mlir.IntegerAttr(filter_width)
    filter_height = mlir.IntegerAttr(filter_height)
    fused_activation_function = _utils.to_str(fused_activation_function)

    result_types = [
        result_type
        or self._infer_result_type(
            input, padding, stride_w, stride_h, filter_width, filter_height
        )
    ]

    super().__init__(
        operands=[input],
        result_types=result_types,
        attributes={
            "padding": padding,
            "stride_w": stride_w,
            "stride_h": stride_h,
            "filter_width": filter_width,
            "filter_height": filter_height,
            "fused_activation_function": mlir.StringAttr(
                fused_activation_function
            ),
        },
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      padding: str | mlir.StringAttr,
      stride_w: int | mlir.IntegerAttr,
      stride_h: int | mlir.IntegerAttr,
      filter_width: int | mlir.IntegerAttr,
      filter_height: int | mlir.IntegerAttr,
  ):
    input_type = _utils.get_tensor_type(input)
    input_shape = input_type.shape

    padding = _utils.to_str(padding)
    stride_w = _utils.to_int(stride_w)
    stride_h = _utils.to_int(stride_h)
    filter_width = _utils.to_int(filter_width)
    filter_height = _utils.to_int(filter_height)

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
        padding=mlir.StringAttr.op_attribute_accessor("padding"),
        stride_w=mlir.IntegerAttr.op_attribute_accessor("stride_w"),
        stride_h=mlir.IntegerAttr.op_attribute_accessor("stride_h"),
        filter_width=mlir.IntegerAttr.op_attribute_accessor("filter_width"),
        filter_height=mlir.IntegerAttr.op_attribute_accessor("filter_height"),
        fused_activation_function=mlir.StringAttr.op_attribute_accessor(
            "fused_activation_function"
        ),
    )


def max_pool_2d(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    padding: str | mlir.StringAttr,
    stride_w: int | mlir.IntegerAttr,
    stride_h: int | mlir.IntegerAttr,
    filter_width: int | mlir.IntegerAttr,
    filter_height: int | mlir.IntegerAttr,
    fused_activation_function: str | mlir.StringAttr = "NONE",
    location=None,
):
  """Max Pool 2D op

  Performs max pool 2D on input.

  Inputs:
    inputs[0]: required: the input tensor
  """
  return MaxPool2DOp(
      input,
      result_type=result_type,
      padding=padding,
      stride_w=stride_w,
      stride_h=stride_h,
      filter_width=filter_width,
      filter_height=filter_height,
      fused_activation_function=fused_activation_function,
      location=location,
  ).output
