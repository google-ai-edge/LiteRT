# Copyright 2026 Google LLC.
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
"""tfl.quantize operation definition."""

from litert.python.mlir import ir
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("tfl.quantize")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class QuantizeOp(core.MlirOpBase):
  """Quantize operator.

  Quantizes float or integer tensors according to the quantization parameters.
  """

  name = "tfl.quantize"

  input = irdl.operand_def()
  output = irdl.result_def()

  qtype = irdl.opt_attr_def(mlir.MlirAttribute)

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase,
      *,
      location=None,
  ):
    input_val = SSAValue.get(input)
    attributes = {}
    if result_type is not None:
      mlir_type = result_type.to_mlir()
      attributes["qtype"] = mlir.MlirAttribute(ir.TypeAttr.get(mlir_type))
    super().__init__(
        operands=[input_val],
        result_types=[result_type],
        location=location,
        attributes=attributes,
    )

  @classmethod
  def overload_cls_attrs(cls):
    return {}


@_utils.op_builder_wraps(QuantizeOp)
def quantize(*args, **kwargs):
  return QuantizeOp(*args, **kwargs).output
