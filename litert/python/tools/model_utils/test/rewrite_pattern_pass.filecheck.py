"""FileCheck test for rewrite pattern pass."""

# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from absl import app
import numpy as np

from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl


class PassWithPattern(mu.RewritePatternPassBase):
  """A simple pass that rewrites tfl.mul to tfl.add based on a pattern."""

  name = "pass-with-pattern"


# Simple pattern that replaces tfl.mul with tfl.add.
@PassWithPattern.register_rewrite_pattern(tfl.MulOp)
def mul_to_add(op: tfl.MulOp, rewriter) -> None:

  with mu.MatchingContext():
    mu.match.pred(op.name == "tfl.mul")
    mu.match.pred(op.fused_activation_function == "NONE")

    lhs, rhs = op.operands
    out = op.results[0]

    with mu.OpBuildingContext(anchor=rewriter):
      new_out = tfl.add(lhs, rhs)
      out.replace_by(new_out)

      # Old ops must be erased to avoid re-scheduling.
      rewriter.erase_op(op)


# FileCheck tooling can be applied against pass results on python-defined
# modules.
# pylint: disable=line-too-long
@testing.run_in_ir_context
def main(_) -> None:
  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"), sym_name="mul"
  )
  def _mul_func(x):
    x = tfl.mul(x, x)
    return x

  PassWithPattern()(_mul_func)

  testing.print_ir("single_mul", _mul_func)
  # CHECK-LABEL: single_mul
  # CHECK: %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %0 : tensor<2x2xf32>

  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"),
      mlir.RankedTensorType([2, 2], "f32"),
      sym_name="double_mul",
  )
  def _double_mul_func(x, _):
    cst = tfl.constant(np.array([1.0, 2.0], dtype=np.float32))
    x = tfl.mul(x, cst)
    x = tfl.mul(x, x)
    return x

  PassWithPattern()(_double_mul_func)

  testing.print_ir("double_mul", _double_mul_func)
  # CHECK-LABEL: double_mul
  # CHECK: %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  # CHECK: %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  # CHECK: %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  # CHECK: return %1 : tensor<2x2xf32>


if __name__ == "__main__":
  app.run(main)
