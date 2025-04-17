# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from absl import app
import numpy as np

from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl
from litert.python.tools.model_utils.litert import passes


@testing.run_in_ir_context
def _example_tests():
  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"), sym_name="mul"
  )
  def _mul_func(x):
    x = tfl.mul(x, x)
    return x

  passes.ExampleRewrites()(_mul_func)

  testing.print_ir("single_mul", _mul_func)
  # CHECK-LABEL: single_mul
  # CHECK: %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %0 : tensor<2x2xf32>

  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"),
      sym_name="double_mul",
  )
  def _double_mul_func(x):
    cst = tfl.constant(np.array([1.0, 2.0], dtype=np.float32))
    x = tfl.mul(x, cst)
    x = tfl.mul(x, x)
    return x

  passes.ExampleRewrites()(_double_mul_func)

  testing.print_ir("double_mul", _double_mul_func)
  # CHECK-LABEL: double_mul
  # CHECK: %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  # CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  # CHECK: %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %1 : tensor<2x2xf32>


@testing.run_in_ir_context
def _qualcomm_tests():
  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"),
      mlir.RankedTensorType([2, 2], "f32"),
      mlir.RankedTensorType([], "f32"),
      sym_name="fc",
  )
  def _fc_func(input, filter, bias):
    r = tfl.fully_connected(
        input, filter, bias, result_type=mlir.RankedTensorType([2, 2], "f32")
    )
    return r

  passes.QualcommRewrites()(_fc_func)

  # TODO(lukeboyer): Add a check for the rewrite.
  testing.print_ir("fc", _fc_func)
  # CHECK-LABEL: fc
  # CHECK: %0 = "tfl.fully_connected"


def main(_) -> None:
  _example_tests()
  _qualcomm_tests()


if __name__ == "__main__":
  app.run(main)
