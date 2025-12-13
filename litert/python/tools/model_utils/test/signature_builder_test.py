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
"""Tests for flatbuffer signature builder."""
import os

from xdsl import irdl

import os # import gfile
from absl.testing import absltest as googletest
from litert.python.tools import model_utils as mu
from litert.python.tools.model_utils import model_builder
from litert.python.tools.model_utils import testing
from litert.python.tools.model_utils.dialect import mlir

SSAValue = irdl.SSAValue


def build_sample_model() -> mlir.ModuleOp:
  @model_builder.build_module_from_py_func(
      mlir.RankedTensorType([2, 2], "f32"),
      mlir.RankedTensorType([2, 2], "f32"),
  )
  def module(_, y):
    return y, y

  signature_builder = mu.SignatureBuilder(module.ops[0])
  signature_builder.name = "serving_default"
  signature_builder.input_names = ["args_0", "args_1"]
  signature_builder.output_names = ["output_0", "output_1"]
  return module


def get_ir_text(module: mlir.ModuleOp) -> str:
  return mu.transform.convert_to_mlir(module).operation.get_asm()


class SignatureBuilderTest(testing.ModelUtilsTestCase):

  def assert_write_flatbuffer_ok(self, module: mlir.ModuleOp):
    """Asserts that the module can be exported to a flatbuffer without error."""
    out_dir = self.create_tempdir()
    flatbuffer_path = os.path.join(out_dir, "model.tflite")
    mu.transform.write_flatbuffer(module, flatbuffer_path)
    self.assertTrue(os.path.exists(flatbuffer_path))

  def test_get_signature_name(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    self.assertEqual(signature_builder.name, "serving_default")

  def test_get_input_map(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    self.assertEqual(
        signature_builder.get_inputs_map(),
        {
            "args_0": module.ops[0].body.block.args[0],
            "args_1": module.ops[0].body.block.args[1],
        },
    )
    self.assert_write_flatbuffer_ok(module)

  def test_get_output_map(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    self.assertEqual(
        signature_builder.get_outputs_map(),
        {
            "output_0": module.ops[0].return_op.operands[0],
            "output_1": module.ops[0].return_op.operands[1],
        },
    )
    self.assert_write_flatbuffer_ok(module)

  def test_set_input_names(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.input_names = ["x", "y"]

    self.assert_filecheck(
        module,
        r"""
        CHECK: %arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["x"]}
        CHECK-SAME: %arg1: tensor<2x2xf32> {tf_saved_model.index_path = ["y"]}
        CHECK-SAME: tf.entry_function =
        CHECK-SAME: inputs = "serving_default_x,serving_default_y"
        """,
    )
    self.assert_write_flatbuffer_ok(module)

  def test_set_output_names(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.output_names = ["o1", "o2"]

    self.assert_filecheck(
        module,
        r"""
        CHECK: tensor<2x2xf32> {tf_saved_model.index_path = ["o1"]}
        CHECK-SAME: tensor<2x2xf32> {tf_saved_model.index_path = ["o2"]}
        CHECK-SAME: tf.entry_function =
        CHECK-SAME: outputs = "serving_default_o1__output,serving_default_o2__output"
        """,
    )
    self.assert_write_flatbuffer_ok(module)

  def test_erase_input_by_name(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.erase_input("args_0")

    self.assertLen(module.ops[0].body.block.args, 1)
    self.assertLen(signature_builder.inputs, 1)
    self.assertEqual(
        signature_builder.get_inputs_map(),
        {"args_1": module.ops[0].body.block.args[0]},
    )
    self.assert_write_flatbuffer_ok(module)

  def test_erase_output_by_name(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.erase_output("output_0")

    self.assertLen(module.ops[0].return_op.operands, 1)
    self.assertLen(signature_builder.outputs, 1)
    self.assertEqual(
        signature_builder.get_outputs_map(),
        {"output_1": module.ops[0].return_op.operands[0]},
    )
    self.assert_write_flatbuffer_ok(module)

  def test_insert_input(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.insert_input(
        mlir.RankedTensorType([10, 10], "f32"), 1, "x"
    )

    self.assertLen(module.ops[0].body.block.args, 3)
    self.assertLen(signature_builder.inputs, 3)
    self.assertEqual(
        signature_builder.get_inputs_map(),
        {
            "args_0": module.ops[0].body.block.args[0],
            "x": module.ops[0].body.block.args[1],
            "args_1": module.ops[0].body.block.args[2],
        },
    )
    self.assertEqual(
        signature_builder.get_inputs_map()["x"].type.shape,
        [10, 10],
    )
    self.assertEqual(
        signature_builder.get_inputs_map()["x"].type.elty,
        "f32",
    )
    self.assert_write_flatbuffer_ok(module)

  def test_insert_output(self):
    module = build_sample_model()
    signature_builder = mu.SignatureBuilder(module.ops[0])
    signature_builder.insert_output(module.ops[0].body.block.args[0], 1, "y")

    self.assertLen(module.ops[0].return_op.operands, 3)
    self.assertLen(signature_builder.outputs, 3)
    self.assertEqual(
        signature_builder.get_outputs_map(),
        {
            "output_0": module.ops[0].return_op.operands[0],
            "y": module.ops[0].body.block.args[0],
            "output_1": module.ops[0].return_op.operands[2],
        },
    )
    self.assert_write_flatbuffer_ok(module)

  def test_erase_all_inputs_outputs(self):
    module = build_sample_model()

    signature_builder = mu.SignatureBuilder(module.ops[0])

    while signature_builder.output_names:
      signature_builder.erase_output(0)

    for op in reversed(module.ops[0].ops):
      if op.name == "func.return":
        continue
      op.detach()
      op.erase()

    while signature_builder.input_names:
      signature_builder.erase_input(0)

    self.assertEmpty(module.ops[0].body.block.args)
    self.assertEmpty(signature_builder.inputs)
    self.assertEmpty(signature_builder.input_names)
    self.assertEmpty(module.ops[0].return_op.operands)
    self.assertEmpty(signature_builder.outputs)
    self.assertEmpty(signature_builder.output_names)
    self.assert_write_flatbuffer_ok(module)


if __name__ == "__main__":
  googletest.main()
