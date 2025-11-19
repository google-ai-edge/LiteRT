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
"""tfl.broadcast_args operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.broadcast_args")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class BroadcastArgsOp(core.MlirOpBase):
  """Return the shape of s0 op s1 with broadcast.

  Given s0 and s1, tensors that represent shapes, compute r0, the broadcasted
  shape. s0, s1 and r0 are all integer vectors.
  """

  name = "tfl.broadcast_args"

  s0 = irdl.operand_def()
  s1 = irdl.operand_def()
  r0 = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      s0: SSAValue | core.MlirOpBase,
      s1: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    s0 = SSAValue.get(s0)
    s1 = SSAValue.get(s1)
    result_types = [result_type or self._infer_result_type(s0, s1)]
    super().__init__(
        operands=[s0, s1],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      s0_val: SSAValue | core.MlirOpBase,
      s1_val: SSAValue | core.MlirOpBase,
  ):
    s0_type = _utils.get_tensor_type(s0_val)
    s1_type = _utils.get_tensor_type(s1_val)
    if not (s0_type.rank == s1_type.rank == 1):
      raise ValueError("s0 and s1 must be rank 1.")

    return mlir.RankedTensorType(
        [max(s0_type.shape[0], s1_type.shape[0])],
        s0_type.element_type,
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(BroadcastArgsOp)
def broadcast_args(*args, **kwargs):
  return BroadcastArgsOp(*args, **kwargs).r0
