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
"""tfl.elu operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.elu")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class EluOp(core.MlirOpBase):
  """Exponential Linear Unit operator.

  Computes the exponential linear f(x) -> exp(x) - 1 for x < 0, x for x >= 0.
  element-wise.
  """

  name = "tfl.elu"

  x = irdl.operand_def()
  y = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    x_val = SSAValue.get(x)
    result_types = [result_type or self._infer_result_type(x_val)]
    super().__init__(
        operands=[x_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      x_val: SSAValue | core.MlirOpBase,
  ):
    # The 'SameOperandsAndResultShape' trait implies the result type has the
    # same shape and element type as the input.
    input_type = _utils.get_tensor_type(x_val)
    return input_type

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(EluOp)
def elu(*args, **kwargs):
  return EluOp(*args, **kwargs).y
