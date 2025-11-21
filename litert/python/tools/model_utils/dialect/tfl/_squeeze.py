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
"""tfl.squeeze operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue

# pylint: disable=redefined-builtin


@core.register_mlir_transform("tfl.squeeze")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class SqueezeOp(core.MlirOpBase):
  """Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor input, this operation returns a tensor of the same type with
  all   dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  squeeze_dims.
  """

  name = "tfl.squeeze"

  input = irdl.operand_def()
  output = irdl.result_def()
  squeeze_dims = irdl.opt_attr_def(mlir.ArrayAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      squeeze_dims: list[int] | tuple[int, ...] = (),
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    input = SSAValue.get(input)

    if isinstance(squeeze_dims, (list, tuple)):
      squeeze_dims = mlir.ArrayAttr(
          [mlir.IntAttr(d, width=64) for d in squeeze_dims]
      )

    result_types = [result_type or self._infer_result_type(input, squeeze_dims)]

    super().__init__(
        operands=[input],
        result_types=result_types,
        attributes={"squeeze_dims": squeeze_dims},
        location=location,
    )

  def _infer_result_type(
      self,
      input: SSAValue | core.MlirOpBase,
      squeeze_dims: mlir.ArrayAttr,
  ):
    input_type = _utils.get_tensor_type(input)
    input_shape = input_type.shape

    if not isinstance(squeeze_dims, mlir.ArrayAttr):
      raise ValueError("squeeze_dims must be an ArrayAttr of IntAttr.")

    dims_to_squeeze = []
    for d_attr in squeeze_dims.data:
      if not isinstance(d_attr, mlir.IntAttr):
        raise ValueError("squeeze_dims must be an ArrayAttr of IntAttr.")
      dims_to_squeeze.append(d_attr.data)

    new_shape = []
    rank = len(input_shape)

    # If squeeze_dims is empty, squeeze all dimensions with size 1
    if not dims_to_squeeze:
      for s in input_shape:
        if s != 1:
          new_shape.append(s)
    else:
      # Normalize negative indices
      normalized_dims = set()
      for d in dims_to_squeeze:
        if d < 0:
          d += rank
        normalized_dims.add(d)

      for i, s in enumerate(input_shape):
        if i in normalized_dims:
          # In valid TFLite, the dimension squeezed must be 1.
          # We can assert this or check bounds, but for shape inference
          # we assume the operation is valid if possible or raise if impossible.
          if s != 1 and s != -1:  # -1 is dynamic
            raise ValueError(
                f"Cannot squeeze dimension {i} with size {s}. "
                "Only dimensions with size 1 can be squeezed."
            )
        else:
          new_shape.append(s)

    return mlir.RankedTensorType(new_shape, input_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    return {}

  @property
  def dims(self) -> list[int]:
    return self._squeeze_dims.numpy().flatten().tolist()


def squeeze(
    input: SSAValue | core.MlirOpBase,
    squeeze_dims: list[int] | tuple[int, ...] | mlir.ArrayAttr = (),
    result_type: core.MlirTypeBase | None = None,
    *,
    location=None,
):
  """Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor input, this operation returns a tensor of the same type with
  all
  dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  squeeze_dims.
  """
  return SqueezeOp(
      input,
      result_type=result_type,
      squeeze_dims=squeeze_dims,
      location=location,
  ).output
