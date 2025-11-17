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
"""tfl.conv_2d operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.conv_2d")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class Conv2DOp(core.MlirOpBase):
  """tfl.conv_2d operation definition."""

  name = "tfl.conv_2d"

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


@_utils.op_builder_wraps(Conv2DOp)
def conv_2d(*args, **kwargs):
  return Conv2DOp(*args, **kwargs).output
