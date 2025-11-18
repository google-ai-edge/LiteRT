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
"""tfl.expand_dims operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

# pylint: disable=redefined-builtin

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.expand_dims")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class ExpandDimsOp(core.MlirOpBase):
  """Inserts a dimension of 1 into a tensor's shape.

  Given a tensor input, this operation inserts a dimension of 1 at the
  dimension index axis of input's shape. The dimension index axis starts at
  zero; if you specify a negative number for axis it is counted backward from
  the end.
  """

  name = "tfl.expand_dims"

  input = irdl.operand_def()
  _dim = irdl.operand_def()  # Renamed internal operand
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      dim: SSAValue | core.MlirOpBase,  # User-facing argument name
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input_val = SSAValue.get(input)
    dim_val = SSAValue.get(dim)  # Corresponds to _dim operand

    result_types = [result_type or self._infer_result_type(input_val, dim_val)]
    super().__init__(
        operands=[input_val, dim_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      input_val: SSAValue | core.MlirOpBase,
      dim_val: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input_val)
    input_shape = list(input_type.shape)
    input_rank = len(input_shape)

    dim_ssa = SSAValue.get(dim_val)
    if not isinstance(dim_ssa.owner, ConstantOp):
      raise ValueError(
          "Cannot infer result shape: 'dim' operand must be a constant."
      )

    axis = dim_ssa.owner.numpy().item()

    # Calculate output rank *before* normalization for validation
    output_rank = input_rank + 1

    # Validate axis according to spec: -1-input.dims() <= dim <= input.dims()
    # This is equivalent to -output_rank <= axis <= input_rank in 0-based index
    if not (-output_rank <= axis <= input_rank):
      raise ValueError(
          f"Axis {axis} is out of bounds for input tensor of rank {input_rank}"
          f" (valid range [-{output_rank}, {input_rank}])."
      )

    # Normalize negative axis to be positive for insertion
    if axis < 0:
      axis += output_rank  # Use output_rank for normalization

    output_shape = input_shape[:axis] + [1] + input_shape[axis:]

    # Result element type is the same as the input element type.
    return mlir.RankedTensorType(output_shape, input_type.element_type)

  @property
  def axis(self) -> int | SSAValue:
    """Returns the axis value if 'dim' is constant, otherwise the SSAValue."""
    dim_ssa = SSAValue.get(self._dim)
    if isinstance(dim_ssa.owner, ConstantOp):
      return dim_ssa.owner.numpy().item()
    return self._dim

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(ExpandDimsOp)
def expand_dims(
    input: SSAValue | core.MlirOpBase,
    dim: int | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  input_val = SSAValue.get(input)  # Keep input as SSAValue

  # Convert dim if needed
  if isinstance(dim, list):
    if len(dim) != 1:
      raise ValueError(f"List 'dim' must have length 1, got {len(dim)}")
    dim = dim[0]

  if isinstance(dim, int):
    # Spec requires dim to be tensor<i32> or tensor<i64>. Default to i32.
    dim_op = ConstantOp(np.array(dim, dtype=np.int32))
    dim_ssa = dim_op.output
  elif isinstance(dim, core.MlirOpBase):
    dim_ssa = SSAValue.get(dim)
  elif isinstance(dim, SSAValue):
    dim_ssa = dim
  else:
    raise TypeError(f"Unsupported type for 'dim': {type(dim)}")

  return ExpandDimsOp(input_val, dim_ssa, *args, **kwargs).output
