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
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.tile")
@core.overload_cls_attrs
@irdl_op_definition
class TileOp(core.MlirOpBase):
  name = "tfl.tile"

  input = operand_def()
  multiples = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      multiples: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    multiples = SSAValue.get(multiples)

    result_types = [result_type or self._infer_result_type(input, multiples)]

    super().__init__(
        operands=[input, multiples],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      multiples: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)

    multiples_array = multiples.owner.numpy()
    multiples_list = [int(i) for i in multiples_array.flatten()]

    if len(multiples_list) != len(input_type.shape):
      raise ValueError(
          f"The length of multiples ({len(multiples_list)}) must match the rank"
          f" of input ({len(input_type.shape)})."
      )

    output_shape = [
        input_type.shape[i] * multiples_list[i]
        for i in range(len(input_type.shape))
    ]

    return mlir.RankedTensorType(output_shape, input_type.element_type)


def tile(
    input: SSAValue | core.MlirOpBase,
    multiples: (
        SSAValue
        | core.MlirOpBase
        | mlir.DenseElementsAttr
        | np.ndarray
        | list[int]
        | tuple[int, ...]
    ),
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  if not isinstance(multiples, (SSAValue, core.MlirOpBase)):
    multiples = ConstantOp(list(map(int, multiples)))
  return TileOp(
      input, multiples, result_type=result_type, location=location
  ).output
