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
"""tfl.fill operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

# pylint: disable=redefined-builtin

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.fill")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class FillOp(core.MlirOpBase):
  """Fill the tensor with given value."""

  name = "tfl.fill"

  dims = irdl.operand_def()  # Shape vector
  input = irdl.operand_def()  # Scalar value to fill with
  result = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      dims: SSAValue | core.MlirOpBase,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    dims_val = SSAValue.get(dims)
    input_val = SSAValue.get(input)

    result_types = [result_type or self._infer_result_type(dims_val, input_val)]
    super().__init__(
        operands=[dims_val, input_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      dims_val: SSAValue | core.MlirOpBase,
      input_val: SSAValue | core.MlirOpBase,
  ):
    dims_ssa = SSAValue.get(dims_val)
    input_type = _utils.get_tensor_type(input_val)

    # Dims must be a constant vector to infer shape.
    if not isinstance(dims_ssa.owner, ConstantOp):
      raise ValueError(
          "Cannot infer result shape: 'dims' operand must be a constant."
      )

    output_shape = dims_ssa.owner.numpy().tolist()

    # Result element type is the same as the input element type.
    return mlir.RankedTensorType(output_shape, input_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(FillOp)
def fill(
    dims: list[int] | tuple[int, ...] | np.ndarray | SSAValue | core.MlirOpBase,
    input: np.generic | int | float | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  # Convert dims if needed
  if isinstance(dims, (list, tuple, np.ndarray)):
    dims_arr = np.array(
        dims, dtype=np.int32
    )  # Default to i32 for shape tensors
    # Do not check ndim as requested
    dims_op = ConstantOp(dims_arr)
    dims_ssa = dims_op.output
  elif isinstance(dims, core.MlirOpBase):
    dims_ssa = SSAValue.get(dims)
  elif isinstance(dims, SSAValue):
    dims_ssa = dims
  else:
    raise TypeError(f"Unsupported type for 'dims': {type(dims)}")

  # Convert input scalar if needed
  if isinstance(input, (np.generic, int, float)):
    if isinstance(input, int):
      input_arr = np.array(input, dtype=np.int32)
    elif isinstance(input, float):
      input_arr = np.array(input, dtype=np.float32)
    else:
      input_arr = np.array(input)
    input_op = ConstantOp(input_arr)
    input_ssa = input_op.output
  elif isinstance(input, core.MlirOpBase):
    input_ssa = SSAValue.get(input)
  elif isinstance(input, SSAValue):
    input_ssa = input
  else:
    raise TypeError(f"Unsupported type for 'input': {type(input)}")

  return FillOp(dims_ssa, input_ssa, *args, **kwargs).result
