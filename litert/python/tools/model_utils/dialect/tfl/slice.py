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


@core.register_mlir_transform("tfl.slice")
@core.overload_cls_attrs
@irdl_op_definition
class SliceOp(core.MlirOpBase):
  """Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in 'begin'.

  begin is zero-based; size is one-based. If size[i] is -1, all remaining
  elements in dimension i are included in the slice. In other words, this
  is equivalent to setting:
    size[i] = input.dim_size(i) - begin[i]

  Requirements:
    0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n)
  """

  name = "tfl.slice"

  input = operand_def()
  begin = operand_def()
  size = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      begin: SSAValue | core.MlirOpBase,
      size: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    begin = SSAValue.get(begin)
    size = SSAValue.get(size)

    result_types = [result_type or self._infer_result_type(input, begin, size)]

    super().__init__(
        operands=[input, begin, size],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      begin: SSAValue | core.MlirOpBase,
      size: SSAValue | core.MlirOpBase,
  ):
    input = SSAValue.get(input)
    begin = SSAValue.get(begin)
    size = SSAValue.get(size)

    input_type = _utils.get_tensor_type(input)
    input_shape = input_type.shape

    if not hasattr(begin.owner, "numpy") or not hasattr(size.owner, "numpy"):
      raise NotImplementedError(
          "Cannot infer result type when begin or size is not constant."
      )

    begin_array = begin.owner.numpy()
    begin_list = [int(i) for i in begin_array.flatten()]
    size_array = size.owner.numpy()
    size_list = [int(i) for i in size_array.flatten()]

    if len(begin_list) != len(size_list) or len(begin_list) != len(input_shape):
      raise ValueError(
          f"Length mismatch: begin={begin_list}, size={size_list},"
          f" input_shape={input_shape}"
      )

    output_shape = []
    for i in range(len(input_shape)):
      if size_list[i] == -1:
        output_shape.append(input_shape[i] - begin_list[i])
      else:
        output_shape.append(size_list[i])

    for i in range(len(input_shape)):
      if not (
          0
          <= begin_list[i]
          <= begin_list[i] + output_shape[i]
          <= input_shape[i]
      ):
        raise ValueError(
            f"Slice parameters out of bounds: begin={begin_list},"
            f" size={size_list}, input_shape={input_shape}"
        )

    return mlir.RankedTensorType(output_shape, input_type.element_type)


def slice(
    input: SSAValue | core.MlirOpBase,
    begin: (
        SSAValue | core.MlirOpBase | np.ndarray | list[int] | tuple[int, ...]
    ),
    size: SSAValue | core.MlirOpBase | np.ndarray | list[int] | tuple[int, ...],
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in 'begin'.

  begin is zero-based; size is one-based. If size[i] is -1, all remaining
  elements in dimension i are included in the slice. In other words, this
  is equivalent to setting:
    size[i] = input.dim_size(i) - begin[i]

  Requirements:
    0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n)
  """
  if not isinstance(begin, (SSAValue, core.MlirOpBase)):
    begin = ConstantOp(np.array(begin, dtype=np.int32))
  if not isinstance(size, (SSAValue, core.MlirOpBase)):
    size = ConstantOp(np.array(size, dtype=np.int32))
  return SliceOp(
      input, begin, size, result_type=result_type, location=location
  ).output
