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
"""tfl.pack operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.pack")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class PackOp(core.MlirOpBase):
  """Packs a list of tensors into one tensor.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of tensors of shape `(A, B, C)`, and an `axis` of `0`, the
  output tensor will have the shape `(N, A, B, C)`, where `N` is the number of
  tensors in the list.
  """

  name = "tfl.pack"

  values = irdl.var_operand_def()
  output = irdl.result_def()
  axis = irdl.opt_attr_def(mlir.IntegerAttr)
  values_count = irdl.opt_attr_def(mlir.IntegerAttr)

  def __init__(
      self,
      values: list[SSAValue | core.MlirOpBase],
      *,
      axis: int | mlir.IntegerAttr = 0,
      result_type: core.MlirTypeBase | None = None,
      location=None,
  ):
    values = [SSAValue.get(v) for v in values]
    axis = _utils.to_int(axis)

    if result_type is None:
      result_type = self._infer_result_type(values, axis)

    super().__init__(
        operands=[values],
        result_types=[result_type],
        attributes={
            "axis": mlir.IntegerAttr(axis),
            "values_count": mlir.IntegerAttr(len(values)),
        },
        location=location,
    )

  def _infer_result_type(
      self,
      values: list[SSAValue | core.MlirOpBase],
      axis: int,
  ):
    if not values:
      raise ValueError(
          "Cannot infer result type for pack with no input values."
      )

    value_types = [_utils.get_tensor_type(v) for v in values]
    first_type = value_types[0]
    element_type = first_type.element_type
    shape = first_type.shape

    for ty in value_types[1:]:
      if ty.element_type != element_type:
        raise ValueError("All values must have the same element type.")
      if ty.shape != shape:
        raise ValueError("All values must have the same shape.")

    rank = len(shape)
    if not -rank - 1 <= axis <= rank:
      raise ValueError(f"Axis {axis} is out of bounds for rank {rank}")
    if axis < 0:
      axis += rank + 1

    output_shape = list(shape)
    output_shape.insert(axis, len(values))

    return mlir.RankedTensorType(tuple(output_shape), element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        axis=mlir.IntegerAttr.op_attribute_accessor("axis"),
        values_count=mlir.IntegerAttr.op_attribute_accessor("values_count"),
    )


@_utils.op_builder_wraps(PackOp)
def pack(*args, **kwargs):
  """Builder function for the tfl.pack operation."""
  return PackOp(*args, **kwargs).output
