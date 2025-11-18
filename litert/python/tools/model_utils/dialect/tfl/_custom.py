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
"""tfl.custom operation definition."""

from typing import Sequence

from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _utils
from . import const_bytes_attr

ConstBytesAttr = const_bytes_attr.ConstBytesAttr
SSAValue = irdl.SSAValue


@core.register_mlir_transform("tfl.custom")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CustomOp(core.MlirOpBase):
  """Custom op.

  A generic op for any TFLite custom operation.

  input: A list of inputs in the original op.
  custom_code: A string used to identify which exactly this op is.
  custom_option: a holder to save the op attributes in bytes fashion.
  output: A list of outputs in the original op.
  """

  name = "tfl.custom"

  input = irdl.var_operand_def()
  output = irdl.var_result_def()

  custom_code = irdl.attr_def(mlir.StringAttr)
  custom_option = irdl.attr_def(ConstBytesAttr)  # Use ConstBytesAttr here

  def __init__(
      self,
      inputs: Sequence[SSAValue | core.MlirOpBase],
      result_types: Sequence[core.MlirTypeBase],
      *,
      custom_code: str | mlir.StringAttr,
      custom_option: bytes | ConstBytesAttr,  # Accept bytes or the attr type
      location=None,
  ):
    input_values = [SSAValue.get(i) for i in inputs]

    if not result_types:
      raise ValueError(
          "Result types must be explicitly specified for tfl.custom"
      )

    for rt in result_types:
      if not isinstance(rt, core.MlirTypeBase):
        raise TypeError(
            f"All result_types must be MlirTypeBase, got {type(rt)}"
        )

    custom_code_attr = mlir.StringAttr(_utils.to_str(custom_code))
    custom_option_attr = ConstBytesAttr(custom_option)

    super().__init__(
        operands=[input_values],
        result_types=[list(result_types)],
        attributes={
            "custom_code": custom_code_attr,
            "custom_option": custom_option_attr,
        },
        location=location,
    )

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        custom_code=mlir.StringAttr.op_attribute_accessor("custom_code"),
    )


# pylint:disable=missing-function-docstring
@_utils.op_builder_wraps(CustomOp)
def custom(
    inputs: SSAValue | core.MlirOpBase,
    result_types: Sequence[core.MlirTypeBase],
    custom_code: str | mlir.StringAttr,
    custom_option: bytes | ConstBytesAttr,
    **kwargs,
):
  if not isinstance(result_types, Sequence) or not all(
      isinstance(rt, core.MlirTypeBase) for rt in result_types
  ):
    raise TypeError(
        "'result_types' must be a sequence of MlirTypeBase instances."
    )
  if not isinstance(custom_option, (bytes, ConstBytesAttr)):
    raise TypeError("'custom_option' must be of type bytes or ConstBytesAttr.")

  op = CustomOp(
      list(inputs),
      list(result_types),
      custom_code=custom_code,
      custom_option=custom_option,
      **kwargs,
  )
  return op.output
