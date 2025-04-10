import logging
from typing import Sequence
from mlir import ir
from litert.python.google.tools.model_utils import core
from litert.python.google.tools.model_utils import transform


_mlir_opt_module_passes = []


def register_module_pass(*args, **kwargs):
  def reg(cls):
    py_pass = cls(*args, **kwargs)
    assert py_pass.name is not None
    _mlir_opt_module_passes.append(py_pass)
    return cls

  return reg


def _build_py_pass_fn(py_pass: core.ModulePassBase):
  """Builds a Python callable from a ModelUtils Pass for mlir-opt."""

  def py_pass_fn(ir_module: ir.Module):
    logging.warning("[MlirOpt-PyPassFn] Running %s", py_pass.name)
    ir_context = ir_module.context
    with ir_context, ir.Location.unknown():

      module = transform.mlir_to_python(ir_module.operation)
      py_pass(module)

      new_ir_module = transform.python_to_mlir(module)

    return new_ir_module

  return py_pass_fn


def main(argv: Sequence[str]):
  argv = list(argv)
  argv += [
      "--mlir-disable-threading",
      "--mlir-elide-elementsattrs-if-larger=16",
  ]

  pass_names = []
  pass_descriptions = []
  pass_fns = []

  for py_pass in _mlir_opt_module_passes:
    pass_names.append(py_pass.name)
    pass_descriptions.append(py_pass.description)
    pass_fns.append(_build_py_pass_fn(py_pass))

  core.pybind.mlir_opt_main(argv, pass_names, pass_descriptions, pass_fns)
