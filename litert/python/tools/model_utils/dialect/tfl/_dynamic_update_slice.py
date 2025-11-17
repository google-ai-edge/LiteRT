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
"""tfl.dynamic_update_slice operation definition."""

from typing import Sequence

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.dynamic_update_slice")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class DynamicUpdateSliceOp(core.MlirOpBase):
  """DynamicUpdateSlice.

  DynamicUpdateSlice op that have the same semantics with XLA
  DynamicUpdateSlice. Generates a result which is the value of the input
  array operand, with a slice update overwritten at start_indices.

  See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
  """

  name = "tfl.dynamic_update_slice"

  operand = irdl.operand_def()
  update = irdl.operand_def()
  start_indices = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      operand: SSAValue | core.MlirOpBase,
      update: SSAValue | core.MlirOpBase,
      start_indices: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    operand_val = SSAValue.get(operand)
    update_val = SSAValue.get(update)
    start_indices_val = SSAValue.get(start_indices)

    result_types = [
        result_type or self._infer_result_type(operand_val, update_val)
    ]

    super().__init__(
        operands=[operand_val, update_val, start_indices_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      operand_val: SSAValue | core.MlirOpBase,
      update_val: SSAValue | core.MlirOpBase,
  ):
    operand_type = _utils.get_tensor_type(operand_val)
    update_type = _utils.get_tensor_type(update_val)
    if operand_type.element_type != update_type.element_type:
      raise ValueError(
          "Element type of 'operand' and 'update' must match:"
          f" {operand_type.element_type} != {update_type.element_type}"
      )
    return operand_type

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(DynamicUpdateSliceOp)
def dynamic_update_slice(
    operand: SSAValue | core.MlirOpBase,
    update: SSAValue | core.MlirOpBase,
    start_indices: Sequence[int] | np.ndarray | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  # Convert start_indices if needed
  if isinstance(start_indices, (list, tuple, np.ndarray)):
    # Determine dtype (i32 or i64). Default to i32.
    # The actual required type depends on the model/backend constraints,
    # but i32 is common.
    indices_dtype = np.int32
    start_indices_arr = np.array(start_indices, dtype=indices_dtype)
    indices_op = ConstantOp(start_indices_arr)
    start_indices_ssa = indices_op.output
  elif isinstance(start_indices, core.MlirOpBase):
    start_indices_ssa = SSAValue.get(start_indices)
  elif isinstance(start_indices, SSAValue):
    start_indices_ssa = start_indices
  else:
    raise TypeError(
        f"Unsupported type for 'start_indices': {type(start_indices)}"
    )

  operand_val = SSAValue.get(operand)
  update_val = SSAValue.get(update)

  return DynamicUpdateSliceOp(
      operand_val, update_val, start_indices_ssa, *args, **kwargs
  ).output
