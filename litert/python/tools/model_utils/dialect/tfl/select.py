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


@core.register_mlir_transform("tfl.select")
@core.overload_cls_attrs
@irdl_op_definition
class SelectOp(core.MlirOpBase):
  """Select operator

  Select values of 'x' if the corresponding value of 'condition' is true or the
  value of 'y' if false. There are valid condition input sizes:

  Either the same shape (in which case the select is elementwise), or
  condition must be Rank 1 and match over the first dimension.
  """

  name = "tfl.select"

  condition = operand_def()
  x = operand_def()
  y = operand_def()
  output = result_def()

  def __init__(
      self,
      condition: SSAValue | core.MlirOpBase,
      x: SSAValue | core.MlirOpBase,
      y: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    condition = SSAValue.get(condition)
    x = SSAValue.get(x)
    y = SSAValue.get(y)

    result_types = [result_type or self._infer_result_type(condition, x, y)]

    super().__init__(
        operands=[condition, x, y],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      condition: SSAValue | core.MlirOpBase,
      x: SSAValue | core.MlirOpBase,
      y: SSAValue | core.MlirOpBase,
  ):
    cond_type = _utils.get_tensor_type(condition)
    x_type = _utils.get_tensor_type(x)
    y_type = _utils.get_tensor_type(y)

    if x_type.element_type != y_type.element_type:
      raise ValueError("x and y must have the same element type.")

    if cond_type.shape == x_type.shape:
      return mlir.RankedTensorType(x_type.shape, x_type.element_type)

    if len(cond_type.shape) == 1 and cond_type.shape[0] == x_type.shape[0]:
      return mlir.RankedTensorType(x_type.shape, x_type.element_type)

    raise ValueError(
        f"Invalid condition shape: {cond_type.shape}. Valid shapes are either"
        f" the same as x ({x_type.shape}) or Rank 1 with the same first"
        " dimension."
    )


def select(
    condition: SSAValue | core.MlirOpBase,
    x: SSAValue | core.MlirOpBase,
    y: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Select operator

  Select values of 'x' if the corresponding value of 'condition' is true or the
  value of 'y' if false. There are valid condition input sizes:

  Either the same shape (in which case the select is elementwise), or
  condition must be Rank 1 and match over the first dimension.
  """
  return SelectOp(
      condition, x, y, result_type=result_type, location=location
  ).output
