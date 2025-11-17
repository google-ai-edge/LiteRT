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
"""tfl.atan2 operation definition."""

import numpy as np
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.atan2")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class Atan2Op(core.MlirOpBase):
  """Atan2 operation.

  The "atan2" operation computes the arctangent of y/x element-wise,
  respecting signs of the arguments.
  """

  name = "tfl.atan2"

  y = irdl.operand_def()
  x = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      y: SSAValue | core.MlirOpBase,
      x: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      location=None,
  ):
    y = SSAValue.get(y)
    x = SSAValue.get(x)
    result_types = [result_type or self._infer_result_type(y, x)]
    super().__init__(
        operands=[y, x],
        result_types=result_types,
        location=location,
        attributes={},
    )

  def _infer_result_type(
      self,
      y: SSAValue | core.MlirOpBase,
      x: SSAValue | core.MlirOpBase,
  ):
    y_type = _utils.get_tensor_type(y)
    x_type = _utils.get_tensor_type(x)

    # Trait SameOperandsAndResultType implies element types must match.
    if y_type.element_type != x_type.element_type:
      raise ValueError(
          "Element types of y and x do not match:"
          f" {y_type.element_type} != {x_type.element_type}"
      )

    # Element-wise ops typically support broadcasting.
    try:
      output_shape = np.broadcast_shapes(y_type.shape, x_type.shape)
    except ValueError as e:
      raise ValueError(
          f"Shapes are not broadcastable: {y_type.shape} vs {x_type.shape}"
      ) from e

    return mlir.RankedTensorType(output_shape, y_type.element_type)

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(Atan2Op)
def atan2(*args, **kwargs):
  return Atan2Op(*args, **kwargs).output
