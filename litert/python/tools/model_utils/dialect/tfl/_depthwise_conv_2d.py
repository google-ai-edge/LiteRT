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
"""tfl.depthwise_conv_2d operation definition."""

import math

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.depthwise_conv_2d")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class DepthwiseConv2DOp(core.MlirOpBase):
  """Depthwise convolution 2D operator."""

  name = "tfl.depthwise_conv_2d"

  input = irdl.operand_def()
  filter = irdl.operand_def()
  bias = irdl.operand_def()
  output = irdl.result_def()

  dilation_h_factor = irdl.opt_attr_def(mlir.IntegerAttr)
  dilation_w_factor = irdl.opt_attr_def(mlir.IntegerAttr)
  fused_activation_function = irdl.opt_attr_def(mlir.StringAttr)
  padding = irdl.opt_attr_def(mlir.StringAttr)
  stride_h = irdl.opt_attr_def(mlir.IntegerAttr)
  stride_w = irdl.opt_attr_def(mlir.IntegerAttr)
  depth_multiplier = irdl.opt_attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      filter: SSAValue | core.MlirOpBase,
      bias: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
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
    input = SSAValue.get(input)
    filter = SSAValue.get(filter)
    bias = SSAValue.get(bias)

    dilation_h_factor = mlir.IntegerAttr(dilation_h_factor)
    dilation_w_factor = mlir.IntegerAttr(dilation_w_factor)
    fused_activation_function = mlir.StringAttr(fused_activation_function)
    padding = mlir.StringAttr(padding)
    stride_h = mlir.IntegerAttr(stride_h)
    stride_w = mlir.IntegerAttr(stride_w)
    depth_multiplier = mlir.IntegerAttr(depth_multiplier)

    if result_type is None:
      result_type = self._infer_result_type(
          input,
          filter,
          dilation_h_factor,
          dilation_w_factor,
          padding,
          stride_h,
          stride_w,
      )

    super().__init__(
        operands=[input, filter, bias],
        result_types=[result_type],
        attributes={
            "dilation_h_factor": dilation_h_factor,
            "dilation_w_factor": dilation_w_factor,
            "fused_activation_function": fused_activation_function,
            "padding": padding,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "depth_multiplier": depth_multiplier,
        },
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      filter: SSAValue | core.MlirOpBase,
      dilation_h_factor: mlir.IntegerAttr,
      dilation_w_factor: mlir.IntegerAttr,
      padding: mlir.StringAttr,
      stride_h: mlir.IntegerAttr,
      stride_w: mlir.IntegerAttr,
  ):
    input_type = _utils.get_tensor_type(input)
    filter_type = _utils.get_tensor_type(filter)

    if len(input_type.shape) != 4 or len(filter_type.shape) != 4:
      raise ValueError("Input and filter must be 4D tensors.")

    # TFLite DepthwiseConv2D Input:
    #   [batch, height, width, in_channels]
    # TFLite DepthwiseConv2D Filter:
    #   [1, kernel_height, kernel_width, out_channels]
    # Note: out_channels = in_channels * depth_multiplier
    batch, in_h, in_w, _ = input_type.shape
    _, k_h, k_w, out_channels = filter_type.shape

    dh = _utils.to_int(dilation_h_factor)
    dw = _utils.to_int(dilation_w_factor)
    sh = _utils.to_int(stride_h)
    sw = _utils.to_int(stride_w)
    pad = _utils.to_str(padding)

    # Calculate effective filter size with dilation
    eff_k_h = (k_h - 1) * dh + 1
    eff_k_w = (k_w - 1) * dw + 1

    if pad == "SAME":
      out_h = math.ceil(in_h / sh)
      out_w = math.ceil(in_w / sw)
    elif pad == "VALID":
      out_h = math.ceil((in_h - eff_k_h + 1) / sh)
      out_w = math.ceil((in_w - eff_k_w + 1) / sw)
    else:
      raise ValueError(f"Unsupported padding type: {pad}")

    # Result shape: [batch, out_height, out_width, out_channels]
    return mlir.RankedTensorType(
        [batch, out_h, out_w, out_channels], input_type.element_type
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
        depth_multiplier=mlir.IntegerAttr.op_attribute_accessor(
            "depth_multiplier"
        ),
    )


def depthwise_conv_2d(*args, **kwargs):
  return DepthwiseConv2DOp(*args, **kwargs).output
