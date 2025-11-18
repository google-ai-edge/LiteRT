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
"""tfl.range operation definition."""

import math

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

# pylint: disable=redefined-builtin

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.range")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class RangeOp(core.MlirOpBase):
  """Range operator.

  Returns a 1D tensor defined by a sequence from start to limit with a given
  delta.
  """

  name = "tfl.range"

  start = irdl.operand_def()
  limit = irdl.operand_def()
  delta = irdl.operand_def()
  result = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      start: SSAValue | core.MlirOpBase,
      limit: SSAValue | core.MlirOpBase,
      delta: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    start_val = SSAValue.get(start)
    limit_val = SSAValue.get(limit)
    delta_val = SSAValue.get(delta)

    result_types = [
        result_type or self._infer_result_type(start_val, limit_val, delta_val)
    ]
    super().__init__(
        operands=[start_val, limit_val, delta_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      start_val: SSAValue | core.MlirOpBase,
      limit_val: SSAValue | core.MlirOpBase,
      delta_val: SSAValue | core.MlirOpBase,
  ):
    start_type = _utils.get_tensor_type(start_val)
    limit_type = _utils.get_tensor_type(limit_val)
    delta_type = _utils.get_tensor_type(delta_val)

    # All operands must have the same element type.
    if not (
        start_type.element_type
        == limit_type.element_type
        == delta_type.element_type
    ):
      raise ValueError(
          "Operands for tfl.range must all have the same element type. Got:"
          f" start={start_type.element_type}, limit={limit_type.element_type},"
          f" delta={delta_type.element_type}"
      )
    elem_type = start_type.element_type

    # All inputs must be constants to infer shape.
    start_ssa = SSAValue.get(start_val)
    limit_ssa = SSAValue.get(limit_val)
    delta_ssa = SSAValue.get(delta_val)

    if not (
        isinstance(start_ssa.owner, ConstantOp)
        and isinstance(limit_ssa.owner, ConstantOp)
        and isinstance(delta_ssa.owner, ConstantOp)
    ):
      raise ValueError(
          "Cannot infer result shape: all operands of tfl.range must be"
          " constants."
      )

    start_v = start_ssa.owner.numpy().item()
    limit_v = limit_ssa.owner.numpy().item()
    delta_v = delta_ssa.owner.numpy().item()

    # Calculate the length of the resulting 1D tensor.
    # Formula from TF docs: ceil((limit - start) / delta)
    if delta_v == 0:
      raise ValueError("'delta' for tfl.range cannot be zero.")

    if (limit_v > start_v and delta_v < 0) or (
        limit_v < start_v and delta_v > 0
    ):
      length = 0
    else:
      # Use float division for safety, then ceiling.
      length = int(math.ceil((limit_v - start_v) / float(delta_v)))
      if length < 0:
        length = 0

    output_shape = [length]
    return mlir.RankedTensorType(output_shape, elem_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(RangeOp)
def range(
    start: int | float | np.generic | SSAValue | core.MlirOpBase,
    limit: int | float | np.generic | SSAValue | core.MlirOpBase,
    delta: int | float | np.generic | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  # Normalize each operand
  def _normalize_scalar_operand(op):
    if isinstance(op, (int, float, np.generic)):
      # Create 0D scalar tensor constant
      data = np.array(op)
      data = _utils.np_64bit_to_32bit(data)
      return ConstantOp(data).output
    elif isinstance(op, core.MlirOpBase):
      return SSAValue.get(op)
    elif isinstance(op, SSAValue):
      return op
    else:
      raise TypeError(f"Unsupported type for tfl.range operand: {type(op)}")

  start_ssa = _normalize_scalar_operand(start)
  limit_ssa = _normalize_scalar_operand(limit)
  delta_ssa = _normalize_scalar_operand(delta)

  return RangeOp(start_ssa, limit_ssa, delta_ssa, *args, **kwargs).result
