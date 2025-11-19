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
"""tfl.maximum operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.maximum")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class MaximumOp(core.MlirOpBase):
  """Max operator: element-wise max operation."""

  name = "tfl.maximum"

  lhs = irdl.operand_def()
  rhs = irdl.operand_def()
  max = irdl.result_def()

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
      raise ValueError(
          f"Element types of lhs and rhs do not match: {lhs_type.element_type}"
          f" != {rhs_type.element_type}"
      )

    return mlir.RankedTensorType(
        np.broadcast_shapes(lhs_type.shape, rhs_type.shape),
        lhs_type.element_type,
    )

  @classmethod
  def overload_cls_attrs(cls):
    return {}


def maximum(
    lhs: SSAValue | core.MlirOpBase,
    rhs: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Max operator: element-wise max operation."""
  if not isinstance(rhs, (SSAValue, core.MlirOpBase)):
    rhs = ConstantOp(rhs)
  return MaximumOp(lhs, rhs, result_type=result_type, location=location).max
