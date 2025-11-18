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
"""tfl.average_pool_2d operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.average_pool_2d")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class AveragePool2DOp(core.MlirOpBase):
  """Average_pool2d operator.

  Performs average-pooling operation on input.
  """

  name = "tfl.average_pool_2d"

  input = irdl.operand_def()
  filter_height = irdl.opt_attr_def(mlir.IntegerAttr)
  filter_width = irdl.opt_attr_def(mlir.IntegerAttr)
  padding = irdl.opt_attr_def(mlir.StringAttr)
  stride_h = irdl.opt_attr_def(mlir.IntegerAttr)
  stride_w = irdl.opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = irdl.opt_attr_def(mlir.StringAttr)
  output = irdl.result_def()

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


# pylint: disable=missing-function-docstring
@_utils.op_builder_wraps(AveragePool2DOp)
def average_pool_2d(*args, **kwargs):
  return AveragePool2DOp(*args, **kwargs).output
