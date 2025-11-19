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
"""tfl.broadcast_to operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.broadcast_to")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class BroadcastToOp(core.MlirOpBase):
  """Broadcast to operator."""

  name = "tfl.broadcast_to"

  input = irdl.operand_def()
  _shape = irdl.operand_def()
  output = irdl.result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      shape: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      location=None,
  ):
    input = SSAValue.get(input)
    shape = SSAValue.get(shape)

    if result_type is None:
      result_type = self._infer_result_type(input, shape)

    super().__init__(
        operands=[input, shape],
        result_types=[result_type],
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      shape: SSAValue | core.MlirOpBase,
  ):
    input = SSAValue.get(input)
    shape = SSAValue.get(shape)
    if not isinstance(shape.owner, ConstantOp) or not isinstance(
        input.type, mlir.RankedTensorType
    ):
      raise ValueError(
          "result_type must be specified when shape is not from a const op"
      )
    return mlir.RankedTensorType(
        shape.owner.numpy().tolist(), input.type.element_type
    )

  @property
  def shape(self):
    owner = self._shape.owner
    if not isinstance(owner, ConstantOp):
      return self._shape
    return owner.numpy().tolist()


@_utils.op_builder_wraps(BroadcastToOp)
def broadcast_to(input, shape, *args, **kwargs):
  if isinstance(shape, (list, tuple, np.ndarray)):
    shape = ConstantOp(shape)
  return BroadcastToOp(input, shape, *args, **kwargs).output
