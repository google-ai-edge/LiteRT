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

from . import _const

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.split")
@irdl_op_definition
class SplitOp(core.MlirOpBase):
  name = "tfl.split"

  split_dim = operand_def()
  value = operand_def()
  outputs = var_result_def()
  num_splits = opt_attr_def(mlir.IntegerAttr)

  def _infer_result_type(
      self,
      split_dim: SSAValue,
      value: SSAValue,
      num_splits: int,
  ):
    if (
        not isinstance(value.type, mlir.RankedTensorType)
        or not isinstance(split_dim.owner, core.MlirOpBase)
        or split_dim.owner.name != ConstantOp.name
    ):
      raise ValueError(
          "Result type cannot be inferred: value must be a RankedTensorType and"
          " split_dim must be from a const op."
      )

    input_shape = value.type.shape
    split_dim_value = split_dim.owner.numpy().item()

    if split_dim_value >= len(input_shape):
      raise ValueError(
          f"Split dimension {split_dim_value} is out of range for input with"
          f" rank {len(input_shape)}."
      )

    split_size = input_shape[split_dim_value] // num_splits
    remaining = input_shape[split_dim_value] % num_splits
    result_types = []
    for i in range(num_splits):
      output_shape = list(input_shape)
      output_shape[split_dim_value] = split_size + (1 if i < remaining else 0)
      result_types.append(
          mlir.RankedTensorType(tuple(output_shape), value.type.element_type)
      )
    return result_types

  def __init__(
      self,
      split_dim: SSAValue | core.MlirOpBase,
      value: SSAValue | core.MlirOpBase,
      *,
      num_splits: int | mlir.IntegerAttr,
      result_types: list[core.MlirTypeBase] | None = None,
      location=None,
  ):
    split_dim = SSAValue.get(split_dim)
    value = SSAValue.get(value)
    if result_types is None:
      result_types = self._infer_result_type(split_dim, value, num_splits)

    super().__init__(
        operands=[split_dim, value],
        result_types=[result_types],
        attributes={"num_splits": mlir.IntegerAttr(num_splits)},
        location=location,
    )

  @property
  def split_dimension(self) -> int:
    owner = self.split_dim.owner
    if not isinstance(owner, Operation) or owner.name != ConstantOp.name:
      return self.split_dim
    return owner.numpy().item()

  @property
  def dim(self) -> int:
    return self.split_dimension


def split(split_dim, value, num_splits: int, *args, **kwargs):
  if not isinstance(split_dim, (SSAValue, core.MlirOpBase)):
    split_dim = ConstantOp(int(split_dim))
  return SplitOp(
      split_dim, value, num_splits=num_splits, *args, **kwargs
  ).outputs
