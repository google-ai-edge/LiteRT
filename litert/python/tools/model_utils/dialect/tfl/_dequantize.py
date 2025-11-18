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
"""tfl.dequantize operation definition."""

from xdsl import irdl

from litert.python.tools.model_utils import core

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.dequantize")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class DequantizeOp(core.MlirOpBase):
  """Dequantize operator.

  Converts quantized array of integers to floating-points according to the
  quantization parameters.
  """

  name = "tfl.dequantize"

  input = irdl.operand_def()
  output = irdl.result_def()

  # No attributes defined in the spec.

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase = None,
      *,
      location=None,
  ):
    input_val = SSAValue.get(input)
    result_types = [result_type]
    super().__init__(
        operands=[input_val],
        result_types=result_types,
        location=location,
        attributes={},
    )

  @classmethod
  def overload_cls_attrs(cls):
    # No attributes to overload.
    return {}


@_utils.op_builder_wraps(DequantizeOp)
def dequantize(*args, **kwargs):
  return DequantizeOp(*args, **kwargs).output
