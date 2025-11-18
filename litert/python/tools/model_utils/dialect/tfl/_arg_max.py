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
"""tfl.arg_max operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.arg_max")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class ArgMaxOp(core.MlirOpBase):
  """ArgMax operator.

  Returns the index with the largest value across dimensions of a tensor.
  """

  name = "tfl.arg_max"

  input = irdl.operand_def()
  dim = irdl.operand_def()
  output = irdl.result_def()

  # The spec mentions 'output_type' as a derived attribute. This usually means
  # it determines the result type rather than being stored as an explicit MLIR
  # attribute on the operation itself. We handle it via the result_type
  # inference.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      dim: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    dim = SSAValue.get(dim)

    result_types = [result_type or self._infer_result_type(input, dim)]

    super().__init__(
        operands=[input, dim],
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
    input_shape = input_type.shape

    dim_ssa = SSAValue.get(dim_val)
    if not isinstance(dim_ssa.owner, ConstantOp):
      raise ValueError(
          "Cannot infer result shape: 'dim' operand must be a constant."
      )

    axis_val = dim_ssa.owner.numpy()
    if axis_val.ndim != 0:
      raise ValueError(f"'dim' must be a scalar tensor, but got {axis_val}")
    axis = int(axis_val)

    input_rank = len(input_shape)
    if axis < -input_rank or axis >= input_rank:
      raise ValueError(
          f"Axis {axis} is out of bounds for tensor with rank {input_rank}"
      )

    # Normalize negative axis
    if axis < 0:
      axis += input_rank

    output_shape = list(input_shape)
    del output_shape[axis]

    return mlir.RankedTensorType(output_shape, "i32")

  @classmethod
  def overload_cls_attrs(cls):
    # No explicit MLIR attributes to overload for this op.
    return {}

  @property
  def axis(self) -> int | SSAValue:
    """Returns the axis value if 'dim' is constant, otherwise the SSAValue."""
    dim_ssa = SSAValue.get(self.dim)
    if isinstance(dim_ssa.owner, ConstantOp):
      return int(dim_ssa.owner.numpy().item())
    return self.dim


# pylint: disable=missing-function-docstring
@_utils.op_builder_wraps(ArgMaxOp)
def arg_max(
    input: SSAValue | core.MlirOpBase,
    dim: int | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  if isinstance(dim, int):
    dim_op = ConstantOp(np.array(dim, dtype=np.int32))
    dim = dim_op.output
  elif isinstance(dim, core.MlirOpBase):
    dim = SSAValue.get(dim)
  elif not isinstance(dim, SSAValue):
    raise TypeError(f"Unsupported type for 'dim': {type(dim)}")

  return ArgMaxOp(input, dim, *args, **kwargs).output
