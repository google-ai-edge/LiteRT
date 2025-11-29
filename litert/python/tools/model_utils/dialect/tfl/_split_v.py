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
"""tfl.split_v operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const
from . import _utils

SSAValue = irdl.SSAValue
ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.split_v")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class SplitVOp(core.MlirOpBase):
  """Splits a tensor into num_split tensors along one dimension.

  Splits the value tensor along split_dim into a number of sub-tensors with same
  shape as the original one, except for split_dim. The grouping of the resultant
  sub-tensors is decided by size-splits. Same as tf.SplitV.
  """

  name = "tfl.split_v"

  value = irdl.operand_def()
  size_splits = irdl.operand_def()
  split_dim = irdl.operand_def()
  outputs = irdl.var_result_def()

  num_splits = irdl.attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      value: SSAValue | core.MlirOpBase,
      size_splits: SSAValue | core.MlirOpBase,
      split_dim: SSAValue | core.MlirOpBase,
      num_splits: int | mlir.IntegerAttr,
      result_types: list[core.MlirTypeBase] | None = None,
      *,
      location=None,
  ):
    value = SSAValue.get(value)
    size_splits = SSAValue.get(size_splits)
    split_dim = SSAValue.get(split_dim)
    num_splits = mlir.IntegerAttr(num_splits)

    if result_types is None:
      result_types = self._infer_result_type(
          value, size_splits, split_dim, num_splits
      )

    super().__init__(
        operands=[value, size_splits, split_dim],
        result_types=[result_types],
        attributes={"num_splits": num_splits},
        location=location,
    )

  def _infer_result_type(
      self,
      value: SSAValue | core.MlirOpBase,
      size_splits: SSAValue | core.MlirOpBase,
      split_dim: SSAValue | core.MlirOpBase,
      num_splits: mlir.IntegerAttr | None,
  ):
    value = SSAValue.get(value)
    size_splits = SSAValue.get(size_splits)
    split_dim = SSAValue.get(split_dim)

    value_type = _utils.get_tensor_type(value)

    if not hasattr(split_dim.owner, "numpy") or not hasattr(
        size_splits.owner, "numpy"
    ):
      raise NotImplementedError(
          "Cannot infer result type when split_dim or size_splits is not"
          " constant."
      )

    dim_arr = split_dim.owner.numpy()
    dim = int(dim_arr.flatten()[0])

    sizes_arr = size_splits.owner.numpy()
    sizes = [int(x) for x in sizes_arr.flatten()]

    num_splits_val = _utils.to_int(num_splits)
    if len(sizes) != num_splits_val:
      raise ValueError("Length of size_splits does not match num_splits.")

    input_shape = list(value_type.shape)
    rank = len(input_shape)
    if dim < 0:
      dim += rank

    if not (0 <= dim < rank):
      raise ValueError(f"Split dimension {dim} out of bounds.")

    total_size = input_shape[dim]
    computed_sizes = []
    minus_one_idx = -1
    current_sum = 0

    for i, s in enumerate(sizes):
      if s == -1:
        if minus_one_idx != -1:
          raise ValueError("size_splits can contain at most one -1.")
        minus_one_idx = i
        computed_sizes.append(-1)  # Placeholder
      else:
        current_sum += s
        computed_sizes.append(s)

    if minus_one_idx != -1:
      remaining = total_size - current_sum
      if remaining < 0:
        raise ValueError("Sum of explicit sizes exceeds dimension size.")
      computed_sizes[minus_one_idx] = remaining
    elif current_sum != total_size:
      raise ValueError(
          f"Sum of sizes {current_sum} does not match dimension size"
          f" {total_size}."
      )

    result_types = []
    for s in computed_sizes:
      new_shape = list(input_shape)
      new_shape[dim] = s
      result_types.append(
          mlir.RankedTensorType(new_shape, value_type.element_type)
      )

    return result_types

  @classmethod
  def overload_cls_attrs(cls):
    return dict(num_splits=mlir.IntegerAttr.op_attribute_accessor("num_splits"))


def split_v(
    value: SSAValue | core.MlirOpBase,
    size_splits: SSAValue | core.MlirOpBase | np.ndarray | list[int],
    split_dim: SSAValue | core.MlirOpBase | int | np.generic,
    num_splits: int | mlir.IntegerAttr = None,
    result_types: list[core.MlirTypeBase] | None = None,
    *,
    location=None,
):
  """Splits a tensor into num_split tensors along one dimension.

  Splits the value tensor along split_dim into a number of sub-tensors with same
  shape as the original one, except for split_dim. The grouping of the resultant
  sub-tensors is decided by size-splits. Same as tf.SplitV.
  """
  if not isinstance(size_splits, (SSAValue, core.MlirOpBase)):
    size_splits_arr = np.array(size_splits, dtype=np.int32).flatten()
    size_splits = ConstantOp(size_splits_arr)
    num_splits = size_splits_arr.size

  if not isinstance(split_dim, (SSAValue, core.MlirOpBase)):
    split_dim = ConstantOp(np.array(split_dim, dtype=np.int32))

  return SplitVOp(
      value,
      size_splits,
      split_dim,
      num_splits,
      result_types=result_types,
      location=location,
  ).outputs
