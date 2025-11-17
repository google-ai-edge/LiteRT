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
"""Utilities for building TFLite models in Python."""

from typing import cast

from litert.python.mlir import ir
import xdsl.irdl

from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir

__all__ = [
    "build_block_from_py_func",
    "build_func_op_from_py_func",
    "build_module_from_py_func",
]

Block = xdsl.irdl.Block
Region = xdsl.irdl.Region
SSAValue = xdsl.irdl.SSAValue


def build_block_from_py_func(
    *arg_types: mlir.RankedTensorType | core.MlirTypeBase,
):
  """Builds a xDSL block from a Python function."""

  def build(fn):
    block = Block(arg_types=arg_types)

    with core.OpBuildingContext(ir.Location.unknown(), no_insert=True) as opctx:
      outputs = fn(*block.args)

      if outputs is not None:
        if isinstance(outputs, SSAValue):
          outputs = [outputs]

        if not isinstance(outputs, (list, tuple)):
          raise ValueError(
              "The builder function'sreturn must be a list of SSAValues."
          )
        func.ReturnOp(*outputs)

    block.add_ops(opctx.new_ops)
    return block

  return build


def build_func_op_from_py_func(
    *arg_types: mlir.RankedTensorType | core.MlirTypeBase,
    sym_name: str = "main",
    sym_visibility: str = "public",
):
  """Builds a func.FuncOp from a Python function."""
  build_block = build_block_from_py_func(*arg_types)

  def build(fn):
    block = build_block(fn)

    def get_ir_type(value: SSAValue):
      return cast(core.MlirTypeBase, value.type).to_mlir()

    return_op = [op for op in block.ops if op.name == func.ReturnOp.name]
    if not return_op:
      return_types = []
    else:
      return_types = map(get_ir_type, return_op[0].operands)

    op = func.FuncOp.build(
        attributes={
            "sym_name": mlir.StringAttr(sym_name),
            "sym_visibility": mlir.StringAttr(sym_visibility),
            "function_type": mlir.MlirAttribute(
                ir.TypeAttr.get(
                    ir.FunctionType.get(
                        list(map(get_ir_type, block.args)),
                        list(return_types),
                    )
                )
            ),
        },
        regions=[Region([block])],
    )
    op.location = None
    return op

  return build


def build_module_from_py_func(
    *arg_types: mlir.RankedTensorType | core.MlirTypeBase,
    sym_name: str = "main",
    sym_visibility: str = "public",
):
  """Builds a mlir.ModuleOp from a Python function."""
  build_func = build_func_op_from_py_func(
      *arg_types,
      sym_name=sym_name,
      sym_visibility=sym_visibility,
  )

  def build(fn):
    func_op = build_func(fn)
    return mlir.ModuleOp([func_op])

  return build
