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


@core.register_mlir_transform("tfl.maximum")
@core.overload_cls_attrs
@irdl_op_definition
class MaximumOp(core.MlirOpBase):
  """Max operator

  Element-wise max operation.
  """

  name = "tfl.maximum"

  lhs = operand_def()
  rhs = operand_def()
  max = result_def()

  def __init__(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    lhs = SSAValue.get(lhs)
    rhs = SSAValue.get(rhs)

    result_types = [result_type or self._infer_result_type(lhs, rhs)]

    super().__init__(
        operands=[lhs, rhs],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
  ):
    lhs_type = _utils.get_tensor_type(lhs)
    rhs_type = _utils.get_tensor_type(rhs)

    if lhs_type.element_type != rhs_type.element_type:
      raise ValueError("Operands must have the same element type.")

    return mlir.RankedTensorType(
        np.broadcast_shapes(lhs_type.shape, rhs_type.shape),
        lhs_type.element_type,
    )


def maximum(
    lhs: SSAValue | core.MlirOpBase,
    rhs: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Max operator

  Element-wise max operation.
  """
  return MaximumOp(lhs, rhs, result_type=result_type, location=location).max
