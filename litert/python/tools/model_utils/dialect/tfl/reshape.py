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
from litert.python.mlir import ir
import numpy as np
from xdsl.ir.core import *
from xdsl.irdl import *

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

from . import _const

ConstantOp = _const.ConstantOp


@core.register_mlir_transform("tfl.reshape")
@irdl_op_definition
class ReshapeOp(core.MlirOpBase):
  name = "tfl.reshape"

  input = operand_def()
  _shape = operand_def()
  output = result_def()

  def __init__(
      self,
      input: SSAValue | core.MlirOpBase,
      shape: SSAValue | core.MlirOpBase,
      result_type: core.MlirTypeBase | None = None,
      location: ir.Location | None = None,
  ):
    input = SSAValue.get(input)
    shape = SSAValue.get(shape)
    if result_type is None:
      if (
          not isinstance(shape.owner, core.MlirOpBase)
          or not isinstance(input.type, mlir.RankedTensorType)
          or shape.owner.name != ConstantOp.name
      ):
        raise ValueError(
            "result_type must be specified when shape is not from a const op"
        )
      result_type = mlir.RankedTensorType(
          shape.owner.numpy().astype(np.int32),
          input.type.element_type,
      )

    super().__init__(
        operands=[input, shape],
        result_types=[result_type],
        location=location,
    )

  @property
  def shape(self):
    owner = self._shape.owner
    if not isinstance(owner, Operation) or owner.name != ConstantOp.name:
      return self._shape
    return owner.numpy().tolist()


def reshape(input, shape, *args, **kwargs):
  if not isinstance(shape, (SSAValue, core.MlirOpBase)):
    shape = ConstantOp(shape)
  return ReshapeOp(input, shape, *args, **kwargs).output
