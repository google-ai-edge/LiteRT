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
"""MLIR arith dialect definitions."""

from typing import Any

from litert.python.mlir import ir
import numpy as np
from xdsl import irdl

from litert.python.mlir._mlir_libs import converter_api_ext
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir

irdl_op_definition = irdl.irdl_op_definition
attr_def = irdl.attr_def
result_def = irdl.result_def


@core.register_mlir_transform("arith.constant")
@irdl_op_definition
class ConstantOp(core.MlirOpBase):
  """MLIR arith.constant op."""

  name = "arith.constant"

  value = attr_def(mlir.MlirAttribute | mlir.DenseElementsAttr)
  output = result_def()

  def __init__(
      self,
      value: (
          mlir.MlirAttribute
          | mlir.DenseElementsAttr
          | np.ndarray
          | list[Any]
          | tuple[Any, ...]
      ),
      location: ir.Location | None = None,
  ):
    if not isinstance(value, (mlir.MlirAttribute, mlir.DenseElementsAttr)):
      value = mlir.DenseElementsAttr(value)

    super().__init__(
        result_types=[mlir.RankedTensorType.from_mlir(value.data.type)],
        attributes={"value": value},
        location=location,
    )

  def numpy(self):
    if isinstance(self.value, mlir.DenseElementsAttr):
      return self.value.numpy()
    return converter_api_ext.dense_resource_elements_attr_to_numpy(
        self.value.to_mlir()
    )


def constant(*args, **kwargs):
  return ConstantOp(*args, **kwargs).output


def const(*args, **kwargs):
  return ConstantOp(*args, **kwargs).output
