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


@core.register_mlir_transform("tfl.mean")
@core.overload_cls_attrs
@irdl_op_definition
class MeanOp(core.MlirOpBase):
  name = "tfl.mean"

  input = operand_def()
  axis = operand_def()
  output = result_def()

  keep_dims = opt_attr_def(mlir.BoolAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      axis: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      keep_dims: bool | mlir.BoolAttr = False,
      location=None,
  ):
    input = SSAValue.get(input)
    axis = SSAValue.get(axis)
    keep_dims = mlir.BoolAttr(keep_dims)

    result_types = [
        result_type or self._infer_result_type(input, axis, keep_dims)
    ]

    super().__init__(
        operands=[input, axis],
        result_types=result_types,
        attributes={"keep_dims": keep_dims},
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      axis: SSAValue | core.MlirOpBase,
      keep_dims: bool | mlir.BoolAttr,
  ):
    input_type = _utils.get_tensor_type(input)
    keep_dims = _utils.to_bool(keep_dims)

    axes_array = axis.owner.numpy()
    axes = [int(i) for i in axes_array.flatten()]

    input_shape = list(input_type.shape)
    rank = len(input_shape)
    normalized_axes = set([ax if ax >= 0 else ax + rank for ax in axes])

    if any(ax >= rank for ax in normalized_axes):
      raise ValueError(
          f"Axes {axes} are out of bounds for input with rank {rank}."
      )

    output_shape = []
    if keep_dims:
      output_shape = [
          1 if i in normalized_axes else input_shape[i] for i in range(rank)
      ]
    else:
      output_shape = [
          input_shape[i] for i in range(rank) if i not in normalized_axes
      ]

    return mlir.RankedTensorType(output_shape, input_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        keep_dims=mlir.BoolAttr.op_attribute_accessor("keep_dims"),
    )


def mean(input, axis, *args, **kwargs):
  if not isinstance(axis, (SSAValue, core.MlirOpBase)):
    axis = ConstantOp(axis)
  return MeanOp(input, axis, *args, **kwargs).output
