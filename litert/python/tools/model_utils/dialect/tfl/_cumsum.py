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
"""tfl.cumsum operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.cumsum")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CumsumOp(core.MlirOpBase):
  """Cumsum operator.

  Compute the cumulative sum of the tensor x along axis.
  """

  name = "tfl.cumsum"

  input = irdl.operand_def()
  _axis = irdl.operand_def()  # Renamed from axis
  output = irdl.result_def()

  exclusive = irdl.opt_attr_def(mlir.BoolAttr)
  reverse = irdl.opt_attr_def(mlir.BoolAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      axis: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      exclusive: bool | mlir.BoolAttr = False,
      reverse: bool | mlir.BoolAttr = False,
      location=None,
  ):
    input_val = SSAValue.get(input)
    axis_val = SSAValue.get(axis)
    exclusive_attr = mlir.BoolAttr(exclusive)
    reverse_attr = mlir.BoolAttr(reverse)

    result_types = [result_type or self._infer_result_type(input_val)]

    super().__init__(
        operands=[input_val, axis_val],
        result_types=result_types,
        attributes={
            "exclusive": exclusive_attr,
            "reverse": reverse_attr,
        },
        location=location,
    )

  def _infer_result_type(self, input_val: SSAValue | core.MlirOpBase):
    # Cumsum output has the same shape and element type as the input.
    input_type = _utils.get_tensor_type(input_val)
    return input_type

  @property
  def axis(self) -> int | SSAValue:
    """Returns the axis value if '_axis' is constant, otherwise the SSAValue."""
    axis_ssa = SSAValue.get(self._axis)
    if isinstance(axis_ssa.owner, ConstantOp):
      return axis_ssa.owner.numpy().item()
    return self._axis

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        exclusive=mlir.BoolAttr.op_attribute_accessor("exclusive"),
        reverse=mlir.BoolAttr.op_attribute_accessor("reverse"),
    )


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(CumsumOp)
def cumsum(
    input: SSAValue | core.MlirOpBase,
    axis: int | SSAValue | core.MlirOpBase,
    *args,
    **kwargs,
):
  if isinstance(axis, int):
    # Spec requires axis to be a tensor<i32> (scalar).
    axis_op = ConstantOp(np.array(axis, dtype=np.int32))
    axis_ssa = axis_op.output
  elif isinstance(axis, core.MlirOpBase):
    axis_ssa = SSAValue.get(axis)
  elif isinstance(axis, SSAValue):
    axis_ssa = axis
  else:
    raise TypeError(f"Unsupported type for 'axis': {type(axis)}")

  return CumsumOp(input, axis_ssa, *args, **kwargs).output
