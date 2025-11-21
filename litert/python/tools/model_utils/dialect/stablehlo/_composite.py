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
"""stablehlo.composite operation definition."""

import copy
from typing import Any

from litert.python.mlir import ir
from xdsl import irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


# pylint: disable=redefined-builtin
@core.register_mlir_transform("stablehlo.composite")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class CompositeOp(core.MlirOpBase):
  """stablehlo.composite operator."""

  name = "stablehlo.composite"

  inputs = irdl.var_operand_def()
  outputs = irdl.var_result_def()

  composite_attributes = irdl.attr_def(mlir.DictAttr)
  decomposition = irdl.opt_attr_def(mlir.MlirAttribute)

  @property
  def composite_name(self) -> str:
    return self.attributes["name"].data

  @composite_name.setter
  def composite_name(self, name: str | mlir.StringAttr):
    self.attributes["name"] = mlir.StringAttr(name)

  @property
  def decomposition_func(self) -> func.FuncOp:
    module = self
    while not isinstance(module, mlir.ModuleOp):
      if module is None:
        raise ValueError("Failed to find container module for CompositeOp.")
      module = module.parent

    decomp = str(self.decomposition.data).strip("@#")
    for fn in module.ops:
      if isinstance(fn, func.FuncOp) and fn.sym_name == decomp:
        return fn
    raise ValueError(f"Failed to find decomposition function {decomp}.")

  @decomposition_func.setter
  def decomposition_func(
      self,
      decomposition: func.FuncOp | str | mlir.StringAttr,
  ):
    if isinstance(decomposition, mlir.StringAttr):
      decomp = decomposition.data
    elif isinstance(decomposition, func.FuncOp):
      decomp = decomposition.name
    else:
      decomp = str(decomposition)

    decomp = decomp.strip("@#")
    self.decomposition = mlir.MlirAttribute(ir.Attribute.parse("@" + decomp))


# pylint: disable=missing-function-docstring
def composite(
    *inputs: SSAValue | core.MlirOpBase,
    name: str | mlir.StringAttr,
    decomposition: func.FuncOp,
    composite_attributes: dict[str, Any] | None = None,
):
  operands = inputs
  result_types = [x.type for x in decomposition.return_op.operands]

  composite_attributes = copy.copy(composite_attributes) or {}
  for k, v in composite_attributes.items():
    if isinstance(v, str):
      composite_attributes[k] = mlir.StringAttr(v)
    elif isinstance(v, bool):
      composite_attributes[k] = mlir.BoolAttr(v)
    elif isinstance(v, int):
      composite_attributes[k] = mlir.IntAttr(v)
    elif isinstance(v, float):
      composite_attributes[k] = mlir.FloatAttr(v)
    elif isinstance(v, core.MlirAttributeBase):
      composite_attributes[k] = v
    else:
      raise ValueError(f"Unsupported composite_attribute: {k}: {v}")

  op = CompositeOp.build(
      operands=operands,
      result_types=result_types,
      attributes={
          "composite_attributes": mlir.DictAttr(composite_attributes),
          "decomposition": mlir.MlirAttribute(
              ir.Attribute.parse("@" + decomposition.sym_name)
          ),
      },
  )
  op.attributes["name"] = mlir.StringAttr(name)

  results = op.results
  if len(results) == 1:
    return results[0]
  return results
