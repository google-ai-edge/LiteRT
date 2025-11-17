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


@core.register_mlir_transform("tfl.transpose")
@core.overload_cls_attrs
@irdl_op_definition
class TransposeOp(core.MlirOpBase):
  name = "tfl.transpose"

  input = operand_def()
  perm = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      perm: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    perm = SSAValue.get(perm)

    result_types = [result_type or self._infer_result_type(input, perm)]

    super().__init__(
        operands=[input, perm],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      perm: SSAValue | core.MlirOpBase,
  ):
    input_type = _utils.get_tensor_type(input)
    perm_array = perm.owner.numpy()
    perm_list = [int(i) for i in perm_array.flatten()]
    input_shape = input_type.shape
    if len(perm_list) != len(input_shape):
      raise ValueError(
          f"The length of perm ({len(perm_list)}) must match the rank of input"
          f" ({len(input_shape)})."
      )
    if sorted(perm_list) != list(range(len(input_shape))):
      raise ValueError(
          f"Invalid permutation: {perm_list} for input rank {len(input_shape)}."
      )

    output_shape = [input_shape[i] for i in perm_list]
    return mlir.RankedTensorType(output_shape, input_type.element_type)


def transpose(
    input: SSAValue | core.MlirOpBase,
    perm: SSAValue | core.MlirOpBase | list[int] | tuple[int, ...] | np.ndarray,
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  if not isinstance(perm, (SSAValue, core.MlirOpBase)):
    perm = ConstantOp(perm)
  return TransposeOp(
      input, perm, result_type=result_type, location=location
  ).output
