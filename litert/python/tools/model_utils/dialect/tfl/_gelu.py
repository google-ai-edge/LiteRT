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
"""tfl.gelu operation definition."""

import numpy as np  # For np.generic type hint
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.gelu")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class GeluOp(core.MlirOpBase):
  """GELU activation function.

  Computes GELU activation function element-wise.
  """

  name = "tfl.gelu"

  input = irdl.operand_def()
  output = irdl.result_def()

  approximate = irdl.opt_attr_def(mlir.BoolAttr)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      *,
      approximate: bool | np.generic | mlir.BoolAttr = False,
      location=None,
  ):
    input_val = SSAValue.get(input)

    # Normalize attribute
    approximate_attr = mlir.BoolAttr(approximate)

    result_types = [result_type or self._infer_result_type(input_val)]

    super().__init__(
        operands=[input_val],
        result_types=result_types,
        attributes={
            "approximate": approximate_attr,
        },
        location=location,
    )

  def _infer_result_type(
      self,
      input_val: SSAValue | core.MlirOpBase,
  ):
    # The 'SameOperandsAndResultShape' trait implies the result type has the
    # same shape and element type as the input.
    input_type = _utils.get_tensor_type(input_val)
    return input_type

  @property
  def approximate_val(self) -> bool:
    """Returns the boolean value of the approximate attribute."""
    # approximate is optional, defaults to False if not present
    if self.approximate is None:
      return False
    return _utils.to_bool(self.approximate)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        approximate=mlir.BoolAttr.op_attribute_accessor("approximate"),
    )


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(GeluOp)
def gelu(*args, **kwargs):
  # approximate defaults to False in __init__ if not provided
  return GeluOp(*args, **kwargs).output
