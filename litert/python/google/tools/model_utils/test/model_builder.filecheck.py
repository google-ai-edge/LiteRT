# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from absl import app
import numpy as np

from litert.python.google.tools.model_utils import model_builder
from litert.python.google.tools.model_utils import testing
from litert.python.google.tools.model_utils.dialect import mlir
from litert.python.google.tools.model_utils.dialect import tfl


@testing.run_in_ir_context
def main(_) -> None:

  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"),
      mlir.RankedTensorType([2, 2], "f32"),
  )
  def _add(x, y):
    return tfl.add(x, y, fused_activation_function="NONE")

  testing.print_ir("add", _add)
  # CEHECK-LABEL: add
  # CHECK: %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %0 : tensor<2x2xf32>

  @model_builder.build_module_from_py_func(mlir.RankedTensorType([2, 2], "f32"))
  def _add_mul(x):
    cst = tfl.constant(np.array([1.0, 2.0], dtype=np.float32))
    x = tfl.add(x, cst)
    x = tfl.mul(x, x)
    return x

  testing.print_ir("add_mul", _add_mul)
  # CHECK-LABEL: add_mul
  # CHECK: %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  # CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  # CHECK: %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %1 : tensor<2x2xf32>


if __name__ == "__main__":
  app.run(main)
