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
"""tfl.gather operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.gather")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class GatherOp(core.MlirOpBase):
  """Gather operator.

  Gather slices from params axis axis according to indices.
  """

  name = "tfl.gather"

  params = irdl.operand_def()
  indices = irdl.operand_def()
  output = irdl.result_def()

  axis = irdl.attr_def(mlir.IntegerAttr)
  batch_dims = irdl.attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      params: SSAValue | core.MlirOpBase,
      indices: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      axis: int | np.generic | mlir.IntegerAttr,
      batch_dims: int | np.generic | mlir.IntegerAttr = 0,  # Default often 0
      location=None,
  ):
    params_val = SSAValue.get(params)
    indices_val = SSAValue.get(indices)

    # Normalize attributes
    axis_int = _utils.to_int(axis)
    batch_dims_int = _utils.to_int(batch_dims)
    axis_attr = mlir.IntegerAttr(axis_int, 32)  # Explicitly use i32 for TFLite
    batch_dims_attr = mlir.IntegerAttr(batch_dims_int, 32)  # Explicitly use i32

    result_types = [
        result_type
        or self._infer_result_type(
            params_val, indices_val, axis_attr, batch_dims_attr
        )
    ]

    super().__init__(
        operands=[params_val, indices_val],
        result_types=result_types,
        attributes={
            "axis": axis_attr,
            "batch_dims": batch_dims_attr,
        },
        location=location,
    )

  def _infer_result_type(
      self,
      params_val: SSAValue | core.MlirOpBase,
      indices_val: SSAValue | core.MlirOpBase,
      axis_attr: mlir.IntegerAttr,
      batch_dims_attr: mlir.IntegerAttr,
  ):
    params_type = _utils.get_tensor_type(params_val)
    indices_type = _utils.get_tensor_type(indices_val)
    params_shape = list(params_type.shape)
    indices_shape = list(indices_type.shape)
    params_rank = len(params_shape)
    indices_rank = len(indices_shape)

    axis_val = _utils.to_int(axis_attr)
    batch_dims_val = _utils.to_int(batch_dims_attr)

    # Normalize axis
    if axis_val < 0:
      axis_val += params_rank

    # Basic validation (optional, based on general request)
    if not (0 <= axis_val < params_rank):
      raise ValueError(
          f"Axis {axis_attr.value} out of bounds for params rank {params_rank}"
      )
    if not (0 <= batch_dims_val <= min(params_rank, indices_rank)):
      raise ValueError(f"batch_dims {batch_dims_attr.value} out of bounds")
    if not (batch_dims_val <= axis_val):
      raise ValueError(
          f"batch_dims ({batch_dims_attr.value}) must be <= axis"
          f" ({axis_attr.value})"
      )

    output_shape = (
        params_shape[:axis_val]
        + indices_shape[batch_dims_val:]
        + params_shape[axis_val + 1 :]
    )

    # Result element type is the same as the params element type.
    return mlir.RankedTensorType(output_shape, params_type.element_type)

  @property
  def axis_val(self) -> int:
    """Returns the integer value of the axis attribute."""
    return _utils.to_int(self.axis)

  @property
  def batch_dims_val(self) -> int:
    """Returns the integer value of the batch_dims attribute."""
    return _utils.to_int(self.batch_dims)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        axis=mlir.IntegerAttr.op_attribute_accessor("axis"),
        batch_dims=mlir.IntegerAttr.op_attribute_accessor("batch_dims"),
    )


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(GatherOp)
def gather(*args, **kwargs):
  return GatherOp(*args, **kwargs).output
