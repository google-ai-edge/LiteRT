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
"""MLIR func dialect definitions."""

from litert.python.mlir import ir
from xdsl import irdl
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir


SSAValue = irdl.SSAValue


@core.register_mlir_transform("func.return")
@irdl.irdl_op_definition
class ReturnOp(core.MlirOpBase):
  """MLIR func.return op."""

  name = "func.return"

  arguments = irdl.var_operand_def()

  def __init__(self, *return_vals: SSAValue | irdl.Operation):
    return_vals = core.utils.tree_flatten(return_vals)
    super().__init__(operands=[return_vals])


@core.register_mlir_transform("func.func")
@core.overload_cls_attrs
@irdl.irdl_op_definition
class FuncOp(core.MlirOpBase):
  """MLIR func.func op."""

  name = "func.func"

  body = irdl.region_def("single_block")
  sym_name = irdl.attr_def(mlir.StringAttr)
  sym_visibility = irdl.opt_attr_def(mlir.StringAttr)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        sym_name=mlir.StringAttr.op_attribute_accessor("sym_name"),
        sym_visibility=mlir.StringAttr.op_attribute_accessor("sym_visibility"),
    )

  @property
  def ops(self) -> list[core.MlirOpBase]:
    return list(self.body.ops)

  @property
  def return_op(self) -> ReturnOp | None:
    """Returns the first func.return op in the body, or None if there is none."""
    return_op = [op for op in self.body.ops if op.name == "func.return"]
    if not return_op:
      return None
    return return_op[0]

  def update_function_type(self):
    """Updates the function_type attribute of this op.

    Based on the this op's body block arguments and first func.return op in
    body.
    """

    def _get_ir_type(value: SSAValue):
      ty = value.type
      if not hasattr(ty, "to_mlir"):
        raise ValueError(f"Type {ty} does not have a to_mlir attribute.")
      return ty.to_mlir()

    input_types = [_get_ir_type(x) for x in self.body.block.args]

    return_op = self.return_op
    if return_op:
      result_types = [_get_ir_type(x) for x in return_op.operands]
    else:
      result_types = []

    self.attributes["function_type"] = mlir.MlirAttribute(
        ir.TypeAttr.get(ir.FunctionType.get(input_types, result_types))
    )
