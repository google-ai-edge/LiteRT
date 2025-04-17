from jax import tree_util
from xdsl.ir.core import *
from xdsl.irdl import *
from litert.python.tools.model_utils import core
from . import mlir


@core.register_mlir_transform("func.func")
@core.overload_cls_attrs
@irdl_op_definition
class FuncOp(core.MlirOpBase):
  name = "func.func"

  body = region_def("single_block")
  sym_name = attr_def(mlir.StringAttr)
  sym_visibility = opt_attr_def(mlir.StringAttr)

  @classmethod
  def overload_cls_attrs(cls):
    return dict(
        sym_name=mlir.StringAttr.op_attribute_accessor("sym_name"),
        sym_visibility=mlir.StringAttr.op_attribute_accessor("sym_visibility"),
    )

  @property
  def ops(self) -> list[core.MlirOpBase]:
    return list(self.body.ops)


@core.register_mlir_transform("func.return")
@irdl_op_definition
class ReturnOp(core.MlirOpBase):
  name = "func.return"

  arguments = var_operand_def()

  def __init__(self, *return_vals: SSAValue | Operation):
    return_vals, _ = tree_util.tree_flatten(return_vals)
    super().__init__(operands=[return_vals])
