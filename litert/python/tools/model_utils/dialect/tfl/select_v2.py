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
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.select_v2")
@core.overload_cls_attrs
@irdl_op_definition
class SelectV2Op(core.MlirOpBase):
  """SelectV2 operator

  Select values of 'x' if the corresponding value of 'condition' is true or the
  value of 'y' if false. There are valid condition input sizes:

  Either the same shape (in which case the select is elementwise), or
  Broadcastable shapes between 'condition', 'x' and 'y'.
  """

  name = "tfl.select_v2"

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

    try:
      broadcasted_shape = np.broadcast_shapes(
          cond_type.shape, x_type.shape, y_type.shape
      )
    except ValueError:
      raise ValueError(
          "Invalid shapes for select_v2. Shapes of condition"
          f" ({cond_type.shape}), x ({x_type.shape}), and y ({y_type.shape})"
          " are not broadcastable."
      )

    return mlir.RankedTensorType(broadcasted_shape, x_type.element_type)


def select_v2(
    condition: SSAValue | core.MlirOpBase,
    x: SSAValue | core.MlirOpBase,
    y: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """SelectV2 operator

  Select values of 'x' if the corresponding value of 'condition' is true or the
  value of 'y' if false. There are valid condition input sizes:

  Either the same shape (in which case the select is elementwise), or
  Broadcastable shapes between 'condition', 'x' and 'y'.
  """
  return SelectV2Op(
      condition, x, y, result_type=result_type, location=location
  ).output
