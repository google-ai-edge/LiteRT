from mlir import passmanager
from litert.python.google.tools import model_utils as mu
from litert.python.google.tools.model_utils.dialect import mlir


class MlirPass(mu.core.ModulePassBase):
  """Run registered MLIR pass and pass pipeline via passmanager."""

  def __init__(self, pipeline: str):
    if not pipeline.startswith("builtin.module("):
      pipeline = "builtin.module(" + pipeline + ")"
    self.pipeline = pipeline

  def call(self, module: mlir.ModuleOp):
    pm = passmanager.PassManager.parse(self.pipeline)
    ir_module = mu.transform._python_to_mlir(module)

    pm.run(ir_module)

    new_module = mu.transform._mlir_to_python(ir_module)

    module.replace_by(new_module)
    return module


class CsePass(MlirPass):
  """Eliminate common sub-expressions."""

  def __init__(self):
    super().__init__("builtin.module(cse)")


class CanonicalizePass(MlirPass):
  """Canonicalize operations."""

  def __init__(self):
    super().__init__("builtin.module(canonicalize)")
