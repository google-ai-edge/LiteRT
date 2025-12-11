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
"""Transforms between ModelUtils objects, MLIR objects, and flatbuffers."""

import contextlib
import os
import re

from litert.python.mlir import ir
from litert.python.mlir import passmanager
from xdsl import irdl

import os # import gfile
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir


_Path = str | os.PathLike[str]


def get_ir_context(ctx: ir.Context | None = None, allow_new=True):
  if ctx is not None:
    return ctx
  if ir.Context.current is not None:
    return ir.Context.current
  if allow_new:
    ctx = ir.Context()
    ctx.enable_multithreading = False
    ctx.allow_unregistered_dialects = True
    core.pybind.register_dialects(ctx)
    return ctx
  return contextlib.nullcontext()


def _run_ir_pass_pipeline(pipeline: str, ir_module: ir.Module):
  pipeline = re.sub(r"\s", "", pipeline)
  pm = passmanager.PassManager.parse(pipeline)
  pm.run(ir_module)


def read_flatbuffer(
    file: _Path | None = None,
    content: bytes | None = None,
    ir_context: ir.Context | None = None,
) -> tuple[mlir.ModuleOp, ir.Context]:
  """Reads a flatbuffer file and returns a tuple of the ModelUtils mlir.ModuleOp and the MLIR context.

  Args:
    file: The path to the flatbuffer file.
    content: The content of the flatbuffer file.
    ir_context: The MLIR context to use. If None, the current context is used.

  Returns:
    A tuple of the ModelUtils mlir.ModuleOp and the MLIR context.
  """
  if content is None:
    if file is None:
      raise ValueError("Flatbuffer file or content must be specified")

    with open(file, "rb") as f:
      content = f.read()

  ir_context = get_ir_context(ir_context)
  ir_module = core.pybind.flatbuffer_to_mlir(content, ir_context)
  with ir_context:
    module = _mlir_to_python(ir_module)
  return module, ir_context


def read_mlir(
    file: _Path | None = None,
    content: str | None = None,
    operation: ir.Module | ir.Operation | None = None,
    ir_context: ir.Context | None = None,
) -> tuple[mlir.ModuleOp, ir.Context]:
  """Reads an MLIR file and returns a tuple of the ModelUtils mlir.ModuleOp and the MLIR context.

  Args:
    file: The path to the MLIR file.
    content: The content of the MLIR file.
    operation: The loaded ir.Module operation.
    ir_context: The MLIR context to use. If None, the current context is used.

  Returns:
    A tuple of the ModelUtils mlir.ModuleOp and the MLIR context.
  """
  ir_context = get_ir_context(ir_context)

  with ir_context:
    if operation is not None:
      ir_module = operation
    else:
      if content is None:
        if file is None:
          raise ValueError("Flatbuffer file or content must be specified")

        with open(file, "r") as f:
          content = f.read()
      ir_module = ir.Module.parse(content)

    module = _mlir_to_python(ir_module)
  return module, ir_context


def convert_to_mlir(
    module: mlir.ModuleOp,
    ir_context: ir.Context | None = None,
) -> ir.Module:
  """Converts a ModelUtils mlir.ModuleOp to an MLIR ir.Module operation.

  Args:
    module: The ModelUtils mlir.ModuleOp.
    ir_context: The MLIR context to use. If None, the current context is used.

  Returns:
    The MLIR module operation in ir.Operation.
  """
  with get_ir_context(ir_context, allow_new=False):
    ir_module = _python_to_mlir(module)
  return ir_module


def write_flatbuffer(
    module_op: mlir.ModuleOp | ir.Module | ir.Operation,
    file: _Path | None = None,
    ir_context: ir.Context | None = None,
) -> bytes:
  """Writes the ModelUtils ModuleOp or MLIR ir.Operation to a TFLite flatbuffer file.

  Args:
    module_op: The module op representing the TFLite model.
    file: The path to the output flatbuffer file.
    ir_context: The MLIR context to use. If None, the current context is used.

  Returns:
    The flatbuffer content in bytes.
  """
  if isinstance(module_op, ir.Module):
    ir_module = module_op.operation
  elif isinstance(module_op, ir.Operation):
    ir_module = module_op
  else:
    ir_module = convert_to_mlir(module_op, ir_context)

  with get_ir_context(ir_context, allow_new=False):
    _run_ir_pass_pipeline(
        """builtin.module(
            cse,
            stablehlo-legalize-vhlo,
            reconcile-unrealized-casts
        )""",
        ir_module,
    )
    fbs = core.pybind.mlir_to_flatbuffer(ir_module)

  if file is not None:
    with open(file, "wb") as f:
      f.write(fbs)
  return fbs


def _mlir_to_python(ir_op: ir.Operation | ir.Module):
  """Converts an MLIR operation to a ModelUtils operation."""

  if isinstance(ir_op, ir.Module):
    ir_op = ir_op.operation

  def build_op(ir_op: ir.Operation, mapping):
    """Builds a ModelUtils operation from an MLIR operation."""
    ir_op = ir.OpView(ir_op)
    operands = [mapping[ir_value] for ir_value in ir_op.operands]
    result_types = [
        mlir.type_from_mlir(result.type) for result in ir_op.results
    ]
    regions = [build_region(ir_region) for ir_region in ir_op.regions]

    attributes = {
        name: mlir.attribute_from_mlir(ir_op.attributes[name])
        for name in core.pybind.get_operation_attribute_names(ir_op)
    }

    op_cls = core.mlir_transforms.get(ir_op.name)
    with core.OpBuildingContext(ir_op.location):
      if op_cls is not None:
        # MlirOpBase.build automatically unflattens variadic operands and
        # results.
        op = op_cls.build(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
        )
      else:
        op = mlir.MlirOp(
            name=ir_op.name,
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
        )

    for ir_value, value in zip(ir_op.results, op.results):
      mapping[ir_value] = value
    return op, mapping

  def build_block(ir_block: ir.Block):
    ir_args = ir_block.arguments

    arg_types = [mlir.type_from_mlir(arg.type) for arg in ir_args]
    block = irdl.Block(arg_types=arg_types)
    mapping = {ir_value: value for ir_value, value in zip(ir_args, block.args)}
    ops = []
    for ir_op in ir_block.operations:
      op, mapping = build_op(ir_op, mapping)
      ops.append(op)

    block.add_ops(ops)
    return block

  def build_region(ir_region: ir.Region):
    blocks = [build_block(ir_block) for ir_block in ir_region.blocks]
    return irdl.Region(blocks)

  op = build_op(ir_op, {})[0]
  return op


def _python_to_mlir(op: core.MlirOpBase):
  """Converts a ModelUtils operation to an MLIR operation."""

  def build_region(region: irdl.Region, ir_region: ir.Region):
    prev_ir_block = None
    for block in reversed(region.blocks):
      ir_arg_types = []
      for arg in block.args:
        if not hasattr(arg.type, "to_mlir"):
          raise ValueError(f"Type {arg.type} does not have a to_mlir method.")
        ir_arg_types.append(arg.type.to_mlir())

      if prev_ir_block is None:
        ir_block = ir.Block.create_at_start(ir_region, ir_arg_types)
      else:
        ir_block = prev_ir_block.create_after(*ir_arg_types)

      mapping = {
          value: ir_value
          for value, ir_value in zip(block.args, ir_block.arguments)
      }
      for op in block.ops:
        ir_op, mapping = build_op(op, mapping)
        ir_block.append(ir_op)
      prev_ir_block = ir_block

  def build_op(op: core.MlirOpBase, mapping):
    """Builds an MLIR operation from a ModelUtils operation."""
    # TODO(cnchan): handle successors
    ir_result_types = []
    for ty in op.result_types:
      if not hasattr(ty, "to_mlir"):
        raise ValueError(f"Type {ty} does not have a to_mlir method.")
      ir_result_types.append(ty.to_mlir())

    operands = [mapping[value] for value in op.operands]
    attributes = {}
    for name, attr in op.attributes.items():
      if not hasattr(attr, "to_mlir"):
        raise ValueError(f"Attribute {attr} does not have a to_mlir method.")
      attributes[name] = attr.to_mlir()

    with op.location or ir.Location.unknown():
      ir_op = ir.Operation.create(
          op.name,
          results=ir_result_types,
          operands=operands,
          attributes=attributes,
          regions=len(op.regions),
      )
    for ir_region, region in zip(ir_op.regions, op.regions):
      build_region(region, ir_region)
    for result, ir_result in zip(op.results, ir_op.results):
      mapping[result] = ir_result
    return ir_op, mapping

  with ir.Location.unknown():
    return build_op(op, {})[0]
