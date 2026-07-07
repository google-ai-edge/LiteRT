# Copyright 2026 Google LLC.
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
"""Tests for mixed_precision."""

from litert.python.mlir import ir
from xdsl import irdl
from absl.testing import absltest as googletest
from litert.python.tools import model_utils as mu
from litert.python.tools.mixed_precision import mixed_precision
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import func
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl


@irdl.irdl_op_definition
class DummyRegionOp(mu.core.MlirOpBase):
  name = "test.dummy_region"
  operands = irdl.var_operand_def()
  body = irdl.region_def()


class MixedPrecisionTest(testing.ModelUtilsTestCase):

  def setUp(self):
    super().setUp()
    self.ir_context = mu.get_ir_context()
    self.ir_context.__enter__()

  def tearDown(self):
    self.ir_context.__exit__(None, None, None)
    super().tearDown()

  def test_convert_to_fp16_normal(self):
    tensor_type = mlir.RankedTensorType([2], "f32")

    # Build a simple module:
    # func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    #   %0 = tfl.add %arg0, %arg0
    #   return %0
    # }
    func_block = irdl.Block(arg_types=[tensor_type])
    arg0 = func_block.args[0]
    add_op = tfl.AddOp(arg0, arg0)
    func_block.add_op(add_op)
    func_block.add_op(func.ReturnOp(add_op.output))

    func_op = func.FuncOp.build(
        attributes={
            "sym_name": mlir.StringAttr("main"),
            "sym_visibility": mlir.StringAttr("public"),
            "function_type": mlir.MlirAttribute(
                ir.TypeAttr.get(
                    ir.FunctionType.get(
                        [tensor_type.to_mlir()],
                        [tensor_type.to_mlir()],
                    )
                )
            ),
        },
        regions=[irdl.Region([func_block])],
    )
    func_op.location = None
    module = mlir.ModuleOp([func_op])

    # Convert to FP16
    with self.ir_context:
      mixed_precision.convert_to_fp16(module, self.ir_context)

    # Verify that the AddOp result type is now f16
    self.assertEqual(add_op.output.type.elty, "f16")

  def test_convert_to_fp16_nested_region_skipped(self):
    tensor_type = mlir.RankedTensorType([2], "f32")

    func_block = irdl.Block(arg_types=[tensor_type])
    arg0 = func_block.args[0]

    # Build nested region with block arguments to avoid referencing 
    # outer SSA values directly
    nested_block = irdl.Block(arg_types=[tensor_type])
    nested_arg = nested_block.args[0]
    add_op = tfl.AddOp(nested_arg, nested_arg)
    nested_block.add_op(add_op)

    region = irdl.Region([nested_block])
    dummy_op = DummyRegionOp.build(operands=[arg0], regions=[region])
    dummy_op.location = None

    func_block.add_op(dummy_op)
    func_block.add_op(func.ReturnOp(arg0))

    func_op = func.FuncOp.build(
        attributes={
            "sym_name": mlir.StringAttr("fp32_func"),
            "sym_visibility": mlir.StringAttr("public"),
            "function_type": mlir.MlirAttribute(
                ir.TypeAttr.get(
                    ir.FunctionType.get(
                        [tensor_type.to_mlir()],
                        [tensor_type.to_mlir()],
                    )
                )
            ),
        },
        regions=[irdl.Region([func_block])],
    )
    func_op.location = None
    module = mlir.ModuleOp([func_op])

    # Predicate to mark @fp32_func as FP32
    def fp32_op_predicate(op):
      return isinstance(op, func.FuncOp) and op.sym_name == "fp32_func"

    # Convert to FP16 with predicate
    with self.ir_context:
      mixed_precision.convert_to_fp16(
          module, self.ir_context, fp32_op_predicate
      )

    # Verify that the AddOp result type is STILL f32 (skipped)
    self.assertEqual(add_op.output.type.elty, "f32")

  def test_convert_to_fp16_with_custom_op_predicate(self):
    @model_builder.build_module_from_py_func(
        mlir.RankedTensorType([2, 2], "f32"),
        mlir.RankedTensorType([2, 2], "f32"),
    )
    def module(x, y):
      x = tfl.add(x, y)
      x = tfl.mul(x, y)
      return x

    # Keep AddOp in FP32. MulOp and other parts of the graph should be
    # converted to FP16.
    mixed_precision.convert_to_fp16(
        module, self.ir_context, lambda op: isinstance(op, tfl.AddOp)
    )

    # Let's inspect the ops in the module
    add_ops = [op for op in module.walk() if isinstance(op, tfl.AddOp)]
    mul_ops = [op for op in module.walk() if isinstance(op, tfl.MulOp)]

    self.assertLen(add_ops, 1)
    self.assertLen(mul_ops, 1)

    # AddOp results should be f32
    self.assertEqual(add_ops[0].results[0].type.elty, "f32")
    # MulOp results should be f16
    self.assertEqual(mul_ops[0].results[0].type.elty, "f16")

    # Verify that the function inputs were converted to f16
    main_func = [
        op
        for op in module.walk()
        if isinstance(op, func.FuncOp) and op.sym_name == "main"
    ][0]
    self.assertEqual(main_func.body.block.args[0].type.elty, "f16")
    self.assertEqual(main_func.body.block.args[1].type.elty, "f16")

    # Verify that CastOps are inserted to convert f16 inputs to f32 for AddOp
    cast_ops = [op for op in module.walk() if isinstance(op, tfl.CastOp)]
    self.assertLen(
        cast_ops, 3
    )  # Two casts for AddOp operands (x, y), one for MulOp operand

    # AddOp operands should be the results of CastOps (f16 -> f32)
    self.assertEqual(add_ops[0].operands[0].type.elty, "f32")
    self.assertEqual(add_ops[0].operands[1].type.elty, "f32")
    self.assertIsInstance(add_ops[0].operands[0].owner, tfl.CastOp)
    self.assertIsInstance(add_ops[0].operands[1].owner, tfl.CastOp)

    # MulOp operand should be the result of a CastOp (f32 -> f16)
    # on the AddOp's output
    self.assertEqual(mul_ops[0].operands[0].type.elty, "f16")
    self.assertIsInstance(mul_ops[0].operands[0].owner, tfl.CastOp)
    self.assertEqual(
        mul_ops[0].operands[0].owner.operands[0], add_ops[0].results[0]
    )

  def test_convert_to_fp16_with_conv2d_op(self):
    @model_builder.build_module_from_py_func(
        mlir.RankedTensorType([1, 8, 8, 3], "f32"),
        mlir.RankedTensorType([16, 3, 3, 3], "f32"),
        mlir.RankedTensorType([16], "f32"),
    )
    def module(x, f, b):
      return tfl.conv_2d(x, f, b)

    # Keep Conv2DOp in FP32.
    mixed_precision.convert_to_fp16(
        module, self.ir_context, lambda op: isinstance(op, tfl.Conv2DOp)
    )

    conv_ops = [op for op in module.walk() if isinstance(op, tfl.Conv2DOp)]
    self.assertLen(conv_ops, 1)

    # Conv2DOp outputs/operands should be FP32 (kept)
    self.assertEqual(conv_ops[0].results[0].type.elty, "f32")
    self.assertEqual(conv_ops[0].operands[0].type.elty, "f32")  # input
    self.assertEqual(conv_ops[0].operands[1].type.elty, "f32")  # filter
    self.assertEqual(conv_ops[0].operands[2].type.elty, "f32")  # bias

    # Verify function inputs were converted to f16
    main_func = [
        op
        for op in module.walk()
        if isinstance(op, func.FuncOp) and op.sym_name == "main"
    ][0]
    self.assertEqual(main_func.body.block.args[0].type.elty, "f16")
    self.assertEqual(main_func.body.block.args[1].type.elty, "f16")
    self.assertEqual(main_func.body.block.args[2].type.elty, "f16")

    # CastOps should be inserted to cast inputs from f16 to f32 for Conv2DOp,
    # and to cast output from f32 to f16 for the function return.
    cast_ops = [op for op in module.walk() if isinstance(op, tfl.CastOp)]
    self.assertLen(cast_ops, 4)

  def test_convert_to_fp16_with_custom_name_predicate(self):
    @model_builder.build_module_from_py_func(
        mlir.RankedTensorType([2, 2], "f32"),
        mlir.RankedTensorType([2, 2], "f32"),
    )
    def module(x, y):
      x = tfl.add(x, y)
      x = tfl.mul(x, y)
      return x

    # Locate the AddOp and set its location to "my_custom_layer"
    add_ops = [op for op in module.walk() if isinstance(op, tfl.AddOp)]
    self.assertLen(add_ops, 1)
    with self.ir_context:
      add_ops[0].location = ir.Location.name("my_custom_layer")

    # Keep ops matching "my_custom_layer" pattern in FP32
    mixed_precision.convert_to_fp16(
        module,
        self.ir_context,
        lambda op: mixed_precision.match_op_by_name(op, ["my_custom_layer"]),
    )

    # Let's inspect the ops in the module
    add_ops = [op for op in module.walk() if isinstance(op, tfl.AddOp)]
    mul_ops = [op for op in module.walk() if isinstance(op, tfl.MulOp)]

    self.assertLen(add_ops, 1)
    self.assertLen(mul_ops, 1)

    # AddOp results should be f32 (kept in FP32 since name matched)
    self.assertEqual(add_ops[0].results[0].type.elty, "f32")
    # MulOp results should be f16 (converted)
    self.assertEqual(mul_ops[0].results[0].type.elty, "f16")

    # Verify function inputs were converted to f16
    main_func = [
        op
        for op in module.walk()
        if isinstance(op, func.FuncOp) and op.sym_name == "main"
    ][0]
    self.assertEqual(main_func.body.block.args[0].type.elty, "f16")
    self.assertEqual(main_func.body.block.args[1].type.elty, "f16")

    # Verify CastOps were correctly inserted:
    # 2 casts for AddOp inputs (f16 -> f32)
    # 1 cast for AddOp output to MulOp (f32 -> f16)
    cast_ops = [op for op in module.walk() if isinstance(op, tfl.CastOp)]
    self.assertLen(cast_ops, 3)

  def test_parse_fp32_ops(self):
    # Test valid ops
    classes = mixed_precision.parse_fp32_ops(["tfl.AddOp", "tfl.CumsumOp"])
    self.assertEqual(classes, [tfl.AddOp, tfl.CumsumOp])

    # Test invalid format
    with self.assertRaises(ValueError):
      mixed_precision.parse_fp32_ops(["tflAddOp"])

    # Test unknown dialect
    with self.assertRaises(ValueError):
      mixed_precision.parse_fp32_ops(["unknown.AddOp"])

    # Test unknown op name
    with self.assertRaises(ValueError):
      mixed_precision.parse_fp32_ops(["tfl.UnknownOp"])


if __name__ == "__main__":
  googletest.main()
