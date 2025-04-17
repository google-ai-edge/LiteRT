import contextlib
import re

from mlir import ir
from mlir import passmanager
from xdsl.ir.core import *
from xdsl.irdl import *

from google3.pyglib import gfile
from litert.python.tools.model_utils import core
from litert.python.tools.model_utils.dialect import mlir


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
    file: str | None = None,
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

    with gfile.Open(file, "rb") as f:
      content = f.read()

  ir_context = get_ir_context(ir_context)
  ir_module = core.pybind.flatbuffer_to_mlir(content, ir_context)
  with ir_context:
    module = _mlir_to_python(ir_module)
  return module, ir_context


def read_mlir(
    file: str | None = None,
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

        with gfile.Open(file, "r") as f:
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
    file: str | None = None,
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
    with gfile.Open(file, "wb") as f:
      f.write(fbs)
  return fbs


def _mlir_to_python(ir_op: ir.Operation | ir.Module):
  if isinstance(ir_op, ir.Module):
    ir_op = ir_op.operation

  def build_attribute(ir_attr: ir.Attribute):
    attr_cls = core.mlir_transforms.get(type(ir_attr))
    if attr_cls is not None:
      return attr_cls.from_mlir(ir_attr)

    return mlir.MlirAttribute(ir_attr)

  def build_type(ir_type: ir.Type):
    type_cls = core.mlir_transforms.get(ir_type) or core.mlir_transforms.get(
        type(ir_type)
    )
    if type_cls is not None:
      return type_cls.from_mlir(ir_type)

    return mlir.MlirType(ir_type)

  def build_op(ir_op: ir.Operation, mapping: dict):
    ir_op = ir.OpView(ir_op)
    operands = [mapping[ir_value] for ir_value in ir_op.operands]
    result_types = [build_type(result.type) for result in ir_op.results]
    regions = [build_region(ir_region) for ir_region in ir_op.regions]

    attributes = {
        name: build_attribute(ir_op.attributes[name])
        for name in core.pybind.get_operation_attribute_names(ir_op)
    }

    op_cls = core.mlir_transforms.get(ir_op.name)
    with core.OpBuildingContext(ir_op.location):
      if op_cls is not None:
        op_def = op_cls.get_irdl_definition()

        # Greedily mapped var operands and results to the first var operand or
        # result definition.
        xdsl_operands = []
        for i, operand in enumerate(operands):
          _, operand_def = op_def.operands[i]
          if isinstance(operand_def, VarOperandDef):
            xdsl_operands.append(operands[i:])
            break
          else:
            xdsl_operands.append(operand)

        xdsl_result_types = []
        for i, result_type in enumerate(result_types):
          _, result_def = op_def.results[i]
          if isinstance(result_def, VarResultDef):
            xdsl_result_types.append(result_types[i:])
            break
          else:
            xdsl_result_types.append(result_type)

        op = op_cls.build(
            operands=xdsl_operands,
            result_types=xdsl_result_types,
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

    arg_types = [build_type(arg.type) for arg in ir_args]
    block = Block(arg_types=arg_types)
    mapping = {ir_value: value for ir_value, value in zip(ir_args, block.args)}
    ops = []
    for ir_op in ir_block.operations:
      op, mapping = build_op(ir_op, mapping)
      ops.append(op)

    block.add_ops(ops)
    return block

  def build_region(ir_region: ir.Region):
    blocks = [build_block(ir_block) for ir_block in ir_region.blocks]
    return Region(blocks)

  op = build_op(ir_op, {})[0]
  return op


def _python_to_mlir(op: core.MlirOpBase):

  def build_region(region: Region, ir_region: ir.Region):
    prev_ir_block = None
    for block in reversed(region.blocks):
      ir_arg_types = [arg.type.to_mlir() for arg in block.args]
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

  def build_op(op: core.MlirOpBase, mapping: dict):
    # TODO: handle successors
    ir_result_types = [ty.to_mlir() for ty in op.result_types]
    operands = [mapping[value] for value in op.operands]
    attributes = {name: attr.to_mlir() for name, attr in op.attributes.items()}

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
