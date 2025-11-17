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
"""tfl.no_value definition."""

from litert.python.mlir import ir
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

# pylint: disable=redefined-builtin

SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.no_value")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class NoValueOp(core.MlirOpBase):
  """Operator that returns a no-value."""

  name = "tfl.no_value"

  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      *,
      location=None,
  ):
    super().__init__(
        operands=[],
        result_types=[mlir.MlirType(ir.Type.parse("none"))],
        location=location,
        attributes={
            "value": mlir.MlirAttribute(ir.Attribute.parse("unit")),
        },
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(NoValueOp)
def no_value(*args, **kwargs):
  return NoValueOp(*args, **kwargs).output
