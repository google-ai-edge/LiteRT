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


@core.register_mlir_transform("tfl.shape")
@core.overload_cls_attrs
@irdl_op_definition
class ShapeOp(core.MlirOpBase):
  """Shape operator

  Returns the shape of a tensor.
  """

  name = "tfl.shape"

  input = operand_def()
  output = result_def()

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
        attributes={},
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)
    if not isinstance(input_type, mlir.RankedTensorType):
      raise ValueError("Input must be a ranked tensor.")

    rank = len(input_type.shape)
    return mlir.RankedTensorType([rank], "i32")


def shape(
    input: SSAValue | core.MlirOpBase,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Shape operator

  Returns the shape of a tensor.
  """
  return ShapeOp(input, result_type=result_type, location=location).output
