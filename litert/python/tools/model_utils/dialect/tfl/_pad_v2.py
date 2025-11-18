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
"""Tfl.padv2 op definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

# pylint: disable=redefined-builtin
SSAValue = irdl.SSAValue
ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.padv2")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class PadV2Op(core.MlirOpBase):
  """Pads a tensor with given constant values.

  This operation pads a input according to the `paddings` and `constant_values`.
  `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the rank of
  `input`. For each dimension `D` of `input`, `paddings[D, 0]` indicates how
  many zeros to add before the contents of input in that dimension, and
  `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  in that dimension.
  `constant_values` is a scalar tensor of the same type as input that indicates
  the value to use for padding input.

  The padded size of each dimension `D` of the output is:
    `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`.
  """

  name = "tfl.padv2"

  input = irdl.operand_def()
  paddings = irdl.operand_def()
  constant_values = irdl.operand_def()
  output = irdl.result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      paddings: SSAValue | core.MlirOpBase,
      constant_values: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)
    paddings = SSAValue.get(paddings)
    constant_values = SSAValue.get(constant_values)

    result_types = [result_type or self._infer_result_type(input, paddings)]

    super().__init__(
        operands=[input, paddings, constant_values],
        result_types=result_types,
        location=location,
    )

  def _infer_result_type(self, input: SSAValue, paddings: SSAValue,):
    input_type = _utils.get_tensor_type(input)
    if not hasattr(paddings.owner, "numpy"):
      raise NotImplementedError(
          "Cannot infer result type when paddings is not constant."
      )
    paddings_array = paddings.owner.numpy()
    if paddings_array.shape[0] != len(input_type.shape):
      raise ValueError(
          "Paddings must have shape [n, 2], where n is input rank:"
          f" paddings_shape={paddings_array.shape},"
          f" input_rank={len(input_type.shape)}"
      )

    output_shape = [
        dim_shape + dim_paddings[0] + dim_paddings[1]
        for dim_shape, dim_paddings in zip(input_type.shape, paddings_array)
    ]
    return mlir.RankedTensorType(output_shape, input_type.element_type)


def padv2(
    input: SSAValue | core.MlirOpBase,
    paddings: (
        SSAValue | core.MlirOpBase | np.ndarray | list[int] | tuple[int, ...]
    ),
    constant_values: (
        SSAValue | core.MlirOpBase | np.ndarray | float | int | bool
    ),
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Pads a tensor with given constant values."""

  if not isinstance(paddings, (SSAValue, core.MlirOpBase)):
    paddings = ConstantOp(np.array(paddings, dtype=np.int32))
  if not isinstance(constant_values, (SSAValue, core.MlirOpBase)):
    if isinstance(constant_values, float):
      constant_values = np.array(constant_values, dtype=np.float32)
    elif isinstance(constant_values, int):
      constant_values = np.array(constant_values, dtype=np.int32)
    elif isinstance(constant_values, bool):
      constant_values = np.array(constant_values, dtype=np.bool)
    elif not isinstance(constant_values, np.array):
      raise ValueError(
          f"Unsupported type for constant_values: {type(constant_values)}"
      )
    constant_values = ConstantOp(constant_values)

  return PadV2Op(
      input,
      paddings,
      constant_values,
      result_type=result_type,
      location=location,
  ).output
