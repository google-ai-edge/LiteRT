"""Testing utilities for ModelUtils and MLIR."""

import functools

from mlir import ir

from litert.python.google.tools.model_utils import transform
from litert.python.google.tools.model_utils.dialect import mlir


def run_in_ir_context(fn):

  @functools.wraps(fn)
  def new_fn(*args, **kwargs):
    with transform.get_ir_context():
      return fn(*args, **kwargs)

  return new_fn


def print_ir(name: str, module: mlir.ModuleOp | ir.Module | ir.Operation):
  """Prints the MLIR text of the given module."""
  print("\nTEST:" + name)

  if isinstance(module, mlir.ModuleOp):
    module = transform.convert_to_mlir(module)

  if isinstance(module, ir.Module):
    module = module.operation

  if not isinstance(module, ir.Operation):
    raise ValueError("Module must be an ir.Operation")

  module.verify()
  print(module.get_asm(enable_debug_info=False, large_elements_limit=1000))
