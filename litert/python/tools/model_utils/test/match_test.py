# Copyright 2025 Google LLC.
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
"""Tests for model_utils.match module."""

import numpy as np
from xdsl import irdl
from absl.testing import absltest as googletest
from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir
from litert.python.tools.model_utils.dialect import tfl

SSAValue = irdl.SSAValue


def build_sample_model() -> mlir.ModuleOp:
  # pylint: disable=line-too-long
  # module {
  #   func.func public @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  #     %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  #     %1 = tfl.mul %0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  #     %cst = arith.constant dense<1.000000e+00> : tensor<1xf32>
  #     %2 = tfl.sub(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<1xf32>) -> tensor<2x2xf32>
  #     return %2 : tensor<2x2xf32>
  #   }
  # }
  # pylint: enable=line-too-long

  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"), mlir.RankedTensorType([2, 2], "f32")
  )
  def module(x, y):
    x = tfl.add(x, y)
    x = tfl.mul(x, y)
    x = tfl.sub(x, tfl.const(np.array([1.0], dtype=np.float32)))
    return x

  return module


def find_op_by_name(tc: googletest.TestCase, module: mlir.ModuleOp, name: str):
  """Finds the first op in the module with the given op name."""
  for op in module.walk():
    if op.name == name:
      return op
  tc.fail("Op not found: %s" % name)


class MatchTest(testing.ModelUtilsTestCase):

  def setUp(self):
    super().setUp()
    self.ir_context = mu.get_ir_context()
    self.ir_context.__enter__()

  def tearDown(self):
    self.ir_context.__exit__(None, None, None)
    super().tearDown()

  def test_matching_context_early_exit(self):
    with mu.MatchingContext():
      mu.match.fail()
      self.fail("NoMatchError should be caught by MatchingContext")

  def test_match_op_by_operand_name(self):
    module = build_sample_model()
    add_op = find_op_by_name(self, module, "tfl.add")

    # Match the MulOp where the first operand is add_op.
    mul_op = mu.match.op("tfl.mul", operands=[add_op, mu.match.ANY])
    self.assertEqual(mul_op.name, "tfl.mul")

  def test_match_op_by_operand_class(self):
    module = build_sample_model()
    add_op = find_op_by_name(self, module, "tfl.add")

    # Match the MulOp where the first operand is add_op.
    mul_op = mu.match.op(tfl.MulOp, operands=[add_op, mu.match.ANY])
    self.assertEqual(mul_op.name, "tfl.mul")

  def test_match_op_by_result_class(self):
    module = build_sample_model()
    sub_op = find_op_by_name(self, module, "tfl.sub")

    # Match the MulOp where the first result is sub_op's first operand.
    mul_op = mu.match.op(tfl.MulOp, results=[sub_op.operands[0]])
    self.assertEqual(mul_op.name, "tfl.mul")

  def test_match_op_by_operand_and_result(self):
    module = build_sample_model()
    add_op = find_op_by_name(self, module, "tfl.add")
    sub_op = find_op_by_name(self, module, "tfl.sub")

    # Match the MulOp where the first operand is add_op and the first result
    # is sub_op's first operand.
    mul_op = mu.match.op(
        "tfl.mul",
        operands=[add_op, mu.match.ANY],
        results=[sub_op.operands[0]],
    )
    self.assertEqual(mul_op.name, "tfl.mul")

  def test_failed_match_op(self):
    module = build_sample_model()
    sub_op = find_op_by_name(self, module, "tfl.sub")
    self.assertRaises(
        mu.match.NoMatchError,
        lambda: mu.match.op("UNKNOWN", results=[sub_op.operands[0]]),
    )

  def test_match_pred_lambda(self):
    module = build_sample_model()
    add_op = find_op_by_name(self, module, "tfl.add")
    mu.match.pred(lambda: add_op.name == "tfl.add")

  def test_match_pred_value_lambda(self):
    module = build_sample_model()
    add_op = find_op_by_name(self, module, "tfl.add")
    mu.match.pred(add_op, lambda x: x.name == "tfl.add")

  def test_failed_match_pred_lambda(self):
    self.assertRaises(
        mu.match.NoMatchError,
        lambda: mu.match.pred(lambda: False),
    )

  def test_match_dag_results(self):
    module = build_sample_model()
    sub_op = find_op_by_name(self, module, "tfl.sub")
    _ = mu.match.dag(
        """
          (TFL_SubOp 
            (TFL_MulOp 
              (TFL_AddOp $x, $y):$add_res, 
              $y2
            ))""",
        sub_op,
    )

    self.assertIsInstance(_.x, SSAValue)
    self.assertIsInstance(_.x.owner, irdl.Block)

    self.assertIsInstance(_.y, SSAValue)
    self.assertIsInstance(_.y.owner, irdl.Block)

    self.assertIsInstance(_.add_res, SSAValue)
    self.assertIsInstance(_.add_res.owner, tfl.AddOp)

    self.assertIs(_.y, _.y2)

  def test_failed_match_dag(self):
    module = build_sample_model()
    sub_op = find_op_by_name(self, module, "tfl.sub")
    self.assertRaises(
        mu.match.NoMatchError,
        lambda: mu.match.dag(
            """(TFL_SubOp TFL_SubOp)""",
            sub_op,
        ),
    )


if __name__ == "__main__":
  googletest.main()
