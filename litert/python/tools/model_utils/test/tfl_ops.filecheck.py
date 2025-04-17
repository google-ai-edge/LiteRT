"""FileCheck test for TFL op builders and shape inference."""

# RUN: %PYTHON %s | FileCheck %s --dump-input=always

from collections.abc import Callable

from absl import app
from xdsl import irdl

import litert.python.tools.model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl

SSAValue = irdl.SSAValue


def TY(shape, elty):  # pylint: disable=invalid-name
  """Short-cut for creating a RankedTensorType."""
  return mlir.RankedTensorType(shape, elty)


def print_module_and_print(*input_types):
  def inner(builder: Callable[..., SSAValue | tuple[SSAValue, ...]]):
    module = model_builder.build_module_from_py_func(*input_types)(builder)
    testing.print_ir(getattr(builder, "__name__"), module)

    # Check no errors during round-trip conversion
    ir_module = mu.transform.convert_to_mlir(module)
    mu.transform.read_mlir(operation=ir_module)
    print("Round-trip transformation succeeded")

  return inner


# pylint: disable=line-too-long
# pylint: disable=unused-variable
@testing.run_in_ir_context
def main(_) -> None:

  @print_module_and_print(TY([2, 2], "f32"))
  def abs_1(x):
    # CHECK-LABEL: abs_1
    return tfl.abs(x)
    # CHECK: tfl.abs
    # CHECK-SAME: (tensor<2x2xf32>) -> tensor<2x2xf32>

  @print_module_and_print(TY([2, 2], "f32"), TY([2, 2], "f32"))
  def add_1(x, y):
    # CHECK-LABEL: add_1
    return tfl.add(x, y, fused_activation_function="NONE")
    # CHECK: %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>

  @print_module_and_print(TY([2, 2], "f32"), TY([2], "f32"))
  def add_broadcast_1(x, y):
    # CHECK-LABEL: add_broadcast_1
    return tfl.add(x, y, fused_activation_function="NONE")
    # CHECK: %0 = tfl.add
    # CHECK-SAME: fused_activation_function = "NONE"
    # CHECK-SAME: (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>

  @print_module_and_print(TY([2, 2], "f32"))
  def arg_max_1(x):
    # CHECK-LABEL: arg_max_1
    return tfl.arg_max(x, dim=1)
    # CHECK: arith.constant dense<1> : tensor<i32>
    # CHECK: tfl.arg_max
    # CHECK-SAME: (tensor<2x2xf32>, tensor<i32>) -> tensor<2xi32>

  @print_module_and_print(TY([2, 2], "f32"))
  def arg_min_1(x):
    # CHECK-LABEL: arg_min_1
    return tfl.arg_min(x, dim=1)
    # CHECK: arith.constant dense<1> : tensor<i32>
    # CHECK: tfl.arg_min
    # CHECK-SAME: (tensor<2x2xf32>, tensor<i32>) -> tensor<2xi32>

  @print_module_and_print(TY([2, 2], "f32"), TY([2, 2], "f32"))
  def atan2_1(y, x):
    # CHECK-LABEL: atan2_1
    return tfl.atan2(y, x)
    # CHECK: tfl.atan2
    # CHECK-SAME: (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

  @print_module_and_print(TY([1, 10, 10, 3], "f32"))
  def average_pool_2d_1(x):
    # CHECK-LABEL: average_pool_2d_1
    return tfl.average_pool_2d(
        x,
        filter_height=2,
        filter_width=2,
        stride_h=1,
        stride_w=1,
        padding="VALID",
    )
    # CHECK: tfl.average_pool_2d
    # CHECK-SAME: filter_height = 2 : i32
    # CHECK-SAME: filter_width = 2 : i32
    # CHECK-SAME: fused_activation_function = "NONE"
    # CHECK-SAME: padding = "VALID"
    # CHECK-SAME: stride_h = 1 : i32
    # CHECK-SAME: stride_w = 1 : i32
    # CHECK-SAME: (tensor<1x10x10x3xf32>) -> tensor<1x9x9x3xf32>

  @print_module_and_print(TY([1, 2, 3, 4], "f32"), TY([4, 10], "f32"))
  def batch_matmul_1(x, y):
    # CHECK-LABEL: batch_matmul_1
    return tfl.batch_matmul(x, y)
    # CHECK: tfl.batch_matmul
    # CHECK-SAME: adj_x = false
    # CHECK-SAME: adj_y = false
    # CHECK-SAME: (tensor<1x2x3x4xf32>, tensor<4x10xf32>) -> tensor<1x2x3x10xf32>

  @print_module_and_print(TY([1, 2, 4, 3], "f32"), TY([4, 10], "f32"))
  def batch_matmul_adj_x(x, y):
    # CHECK-LABEL: batch_matmul_adj_x
    return tfl.batch_matmul(x, y, adj_x=True)
    # CHECK: tfl.batch_matmul
    # CHECK-SAME: adj_x = true
    # CHECK-SAME: adj_y = false
    # CHECK-SAME: (tensor<1x2x4x3xf32>, tensor<4x10xf32>) -> tensor<1x2x3x10xf32>

  @print_module_and_print(TY([1, 2, 3, 4], "f32"), TY([10, 4], "f32"))
  def batch_matmul_adj_y(x, y):
    # CHECK-LABEL: batch_matmul_adj_y
    return tfl.batch_matmul(x, y, adj_y=True)
    # CHECK: tfl.batch_matmul
    # CHECK-SAME: adj_x = false
    # CHECK-SAME: adj_y = true
    # CHECK-SAME: (tensor<1x2x3x4xf32>, tensor<10x4xf32>) -> tensor<1x2x3x10xf32>

  @print_module_and_print(TY([1, 2, 3, 4], "i32"))
  def bitcast_1(x):
    # CHECK-LABEL: bitcast_1
    return tfl.bitcast(
        x, result_type=mlir.RankedTensorType([1, 2, 3, 4], "ui32")
    )
    # CHECK: tfl.bitcast
    # CHECK-SAME: (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xui32>

  @print_module_and_print(TY([2, 2], "i32"), TY([2, 2], "i32"))
  def bitwise_xor_1(x, y):
    # CHECK-LABEL: bitwise_xor_1
    return tfl.bitwise_xor(x, y)
    # CHECK: tfl.bitwise_xor
    # CHECK-SAME: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

  @print_module_and_print(TY([4], "i32"), TY([2], "i32"))
  def broadcast_args_1(x, y):
    # CHECK-LABEL: broadcast_args_1
    return tfl.broadcast_args(x, y)
    # CHECK: tfl.broadcast_args
    # CHECK-SAME: (tensor<4xi32>, tensor<2xi32>) -> tensor<4xi32>

  @print_module_and_print(TY([2, 2], "f32"))
  def broadcast_to_1(x):
    # CHECK-LABEL: broadcast_to_1
    return tfl.broadcast_to(x, [1, 1, 2, 2])
    # CHECK: arith.constant dense<[1, 1, 2, 2]> : tensor<4xi32>
    # CHECK: tfl.broadcast_to
    # CHECK-SAME: (tensor<2x2xf32>, tensor<4xi32>) -> tensor<1x1x2x2xf32>


if __name__ == "__main__":
  app.run(main)
