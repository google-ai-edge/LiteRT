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
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils


@core.register_mlir_transform("tfl.log")
@core.overload_cls_attrs
@irdl_op_definition
class LogOp(core.MlirOpBase):
  """Natural logarithm operator

  Performs element-wise natural logarithm operation on input.
  """

  name = "tfl.log"

  x = operand_def()
  y = result_def()

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


def log(
    x: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Natural logarithm operator

  Performs element-wise natural logarithm operation on input.
  """
  return LogOp(x, result_type=result_type, location=location).y
