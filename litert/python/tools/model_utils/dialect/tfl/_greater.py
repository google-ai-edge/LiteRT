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
"""tfl.greater operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.greater")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class GreaterOp(core.MlirOpBase):
  """Greater operator.

  Element-wise greater operation.
  """

  name = "tfl.greater"

  lhs = irdl.operand_def()
  rhs = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      lhs: SSAValue | core.MlirOpBase,
      rhs: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    lhs_val = SSAValue.get(lhs)
    rhs_val = SSAValue.get(rhs)
    result_types = [result_type or self._infer_result_type(lhs_val, rhs_val)]
    super().__init__(
        operands=[lhs_val, rhs_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      lhs_val: SSAValue | core.MlirOpBase,
      rhs_val: SSAValue | core.MlirOpBase,
  ):
    lhs_type = _utils.get_tensor_type(lhs_val)
    rhs_type = _utils.get_tensor_type(rhs_val)

    # ResultsBroadcastableShape trait implies output shape is broadcasted shape.
    output_shape = np.broadcast_shapes(lhs_type.shape, rhs_type.shape)

    # Result element type is always boolean (i1) according to the spec.
    return mlir.RankedTensorType(output_shape, "i1")

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(GreaterOp)
def greater(*args, **kwargs):
  return GreaterOp(*args, **kwargs).output
