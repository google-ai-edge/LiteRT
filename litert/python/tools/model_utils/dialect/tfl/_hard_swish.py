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
"""tfl.hard_swish operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue

# pylint: disable=redefined-builtin


@core.register_mlir_transform("tfl.hard_swish")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class HardSwishOp(core.MlirOpBase):
  """Hardswish activation function.

  Computes hard-swish activation function f(x) -> (x * relu6(x+3))/6
  element-wise.
  """

  name = "tfl.hard_swish"

  input = irdl.operand_def()
  output = irdl.result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)

    result_types = [result_type or self._infer_result_type(input)]

    super().__init__(
        operands=[input],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)
    return mlir.RankedTensorType(input_type.shape, input_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return {}


def hard_swish(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Hardswish activation function.

  Computes hard-swish activation function f(x) -> (x * relu6(x+3))/6
  element-wise.
  """
  return HardSwishOp(input, result_type=result_type, location=location).output
