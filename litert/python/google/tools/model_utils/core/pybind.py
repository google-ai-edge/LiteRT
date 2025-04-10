from mlir import ir
import google3.third_party.tensorflow.compiler.mlir.lite.integrations.model_utils_core_pybind as _pybind
from google3.third_party.tensorflow.compiler.mlir.lite.integrations.model_utils_core_pybind import *


def flatbuffer_to_mlir(buffer: bytes, ir_context: ir.Context | None = None):
  if ir_context is None:
    ir_context = ir.Context.current
  return _pybind.flatbuffer_to_mlir(buffer, ir_context)
