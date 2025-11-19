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
"""tfl.relu operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.relu")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class ReluOp(core.MlirOpBase):
  """Relu operator.

  Element-wise Relu operator x -> max(0, x)
  """

  name = "tfl.relu"

  x = irdl.operand_def()
  y = irdl.result_def()

  def __init__(
      self,
      x: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    x = SSAValue.get(x)

    result_types = [result_type or self._infer_result_type(x)]

    super().__init__(
        operands=[x],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      x: SSAValue | core.MlirOpBase,
  ):
    x_type = _utils.get_tensor_type(x)
    return mlir.RankedTensorType(x_type.shape, x_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return {}


def relu(
    x: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Relu operator.

  Element-wise Relu operator x -> max(0, x)
  """
  return ReluOp(x, result_type=result_type, location=location).y
