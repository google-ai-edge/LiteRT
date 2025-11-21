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
"""tfl.split operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

SSAValue = irdl.SSAValue
ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.split")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class SplitOp(core.MlirOpBase):
  """Splits a tensor into num_split tensors along one dimension.

  Splits the value tensor along split_dim into a number of sub-tensors with same
  shape as the original one, except for split_dim. Same as tf.Split.
  """

  name = "tfl.split"

  split_dim = irdl.operand_def()
  value = irdl.operand_def()
  outputs = irdl.var_result_def()

  num_splits = irdl.attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      split_dim: SSAValue | core.MlirOpBase,
      value: SSAValue | core.MlirOpBase,
      num_splits: int | mlir.IntegerAttr,
      result_types: list[core.MlirTypeBase] | None = None,
      *,
      location=None,
  ):
    split_dim = SSAValue.get(split_dim)
    value = SSAValue.get(value)
    num_splits = mlir.IntegerAttr(num_splits)

    if result_types is None:
      result_types = self._infer_result_type(split_dim, value, num_splits)

    super().__init__(
        operands=[split_dim, value],
        result_types=result_types,
        attributes={"num_splits": num_splits},
        location=location,
    )

  def _infer_result_type(
      self,
      split_dim: SSAValue,
      value: SSAValue,
      num_splits: mlir.IntegerAttr,
  ):
    value_type = _utils.get_tensor_type(value)
    num_splits_val = _utils.to_int(num_splits)

    if not hasattr(split_dim.owner, "numpy"):
      raise NotImplementedError(
          "Cannot infer result type when split_dim is not constant."
      )

    dim_array = split_dim.owner.numpy()
    dim = int(dim_array.flatten()[0])

    shape = list(value_type.shape)
    rank = len(shape)

    # Handle negative axis
    if dim < 0:
      dim += rank

    if not (0 <= dim < rank):
      raise ValueError(f"Split dimension {dim} out of bounds for rank {rank}.")

    if shape[dim] % num_splits_val != 0:
      raise ValueError(
          f"Dimension {dim} with size {shape[dim]} is not divisible by"
          f" num_splits {num_splits_val}."
      )

    new_shape = list(shape)
    new_shape[dim] //= num_splits_val

    return [
        mlir.RankedTensorType(new_shape, value_type.element_type)
        for _ in range(num_splits_val)
    ]

  @classmethod
  def overload_cls_attrs(cls):
    return dict(num_splits=mlir.IntegerAttr.op_attribute_accessor("num_splits"))


def split(
    split_dim: SSAValue | core.MlirOpBase | int | np.generic,
    value: SSAValue | core.MlirOpBase,
    num_splits: int | mlir.IntegerAttr,
    result_types: list[core.MlirTypeBase] | None = None,
    *,
    location=None,
):
  """Splits a tensor into num_split tensors along one dimension.

  Splits the value tensor along split_dim into a number of sub-tensors with same
  shape as the original one, except for split_dim. Same as tf.Split.
  """
  if not isinstance(split_dim, (SSAValue, core.MlirOpBase)):
    split_dim = ConstantOp(np.array(split_dim, dtype=np.int32))

  return SplitOp(
      split_dim,
      value,
      num_splits,
      result_types=result_types,
      location=location,
  ).outputs
