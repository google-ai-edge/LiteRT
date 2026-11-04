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

"""Runs models built and edited with litert_model on CompiledModel."""

import numpy as np

from absl.testing import absltest as googletest
from litert.python.litert_model import litert_model as ir
from litert.python.litert_wrapper.compiled_model_wrapper import compiled_model
from litert.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import

F32 = ir.ElementType.FLOAT32


def _run(
    data: bytes, signature: str, **inputs: np.ndarray
) -> dict[str, np.ndarray]:
  cm = compiled_model.CompiledModel.from_buffer(data)
  input_buffers = {}
  for name, arr in inputs.items():
    buf = cm.create_input_buffer_by_name(signature, name)
    buf.write(np.ascontiguousarray(arr))
    input_buffers[name] = buf
  output_details = cm.get_output_tensor_details(signature)
  output_buffers = {
      name: cm.create_output_buffer_by_name(signature, name)
      for name in output_details
  }
  cm.run_by_name(signature, input_buffers, output_buffers)
  outputs = {}
  for name, details in output_details.items():
    shape = details["shape"]
    flat = output_buffers[name].read(int(np.prod(shape)), details["dtype"])
    outputs[name] = flat.reshape(shape)
  return outputs


def _build() -> tuple[ir.Model, ir.Subgraph]:
  """main(x, y) = (x + y) * c, c = [[0, 1, 2], [3, 4, 5]]."""
  model = ir.Model.create()
  subgraph = model.add_subgraph("main")
  x = subgraph.add_tensor(shape=(2, 3), element_type=F32, name="x")
  y = subgraph.add_tensor(shape=(2, 3), element_type=F32, name="y")
  total = subgraph.add_tensor(shape=(2, 3), element_type=F32, name="sum")
  c = subgraph.add_tensor(
      shape=(2, 3),
      element_type=F32,
      name="c",
      weights=np.arange(6, dtype=np.float32).reshape(2, 3),
  )
  out = subgraph.add_tensor(shape=(2, 3), element_type=F32, name="out")
  subgraph.add_op(
      ir.OpCode.ADD, [x, y], [total], builtin_options=schema_fb.AddOptionsT()
  )
  subgraph.add_op(
      ir.OpCode.MUL, [total, c], [out], builtin_options=schema_fb.MulOptionsT()
  )
  model.add_signature("serving", subgraph, [("x", x), ("y", y)], [("out", out)])
  return model, subgraph


class NumericsTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(seed=0)
    self.x = rng.standard_normal((2, 3)).astype(np.float32)
    self.y = rng.standard_normal((2, 3)).astype(np.float32)
    self.c = np.arange(6, dtype=np.float32).reshape(2, 3)

  def test_model_built_from_scratch(self):
    model, _ = _build()
    result = _run(model.serialize(), "serving", x=self.x, y=self.y)
    np.testing.assert_allclose(result["out"], (self.x + self.y) * self.c)

  def test_reloaded_model_round_trip(self):
    model, _ = _build()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    result = _run(reloaded.serialize(), "serving", x=self.x, y=self.y)
    np.testing.assert_allclose(result["out"], (self.x + self.y) * self.c)

  def test_fused_activation_via_builtin_options(self):
    model, subgraph = _build()
    mul = subgraph.ops[1]
    options = mul.builtin_options
    options.fusedActivationFunction = schema_fb.ActivationFunctionType.RELU
    mul.set_builtin_options(options)
    result = _run(model.serialize(), "serving", x=self.x, y=self.y)
    np.testing.assert_allclose(
        result["out"], np.maximum((self.x + self.y) * self.c, 0.0)
    )

  def test_inserted_transpose_and_replaced_uses(self):
    model, subgraph = _build()
    _, y = subgraph.inputs
    add, mul = subgraph.ops
    # y is fed transposed: y' has shape (3, 2) and TRANSPOSE(y') feeds ADD.
    y.set_shape([3, 2])
    perm = subgraph.add_tensor(
        shape=(2,),
        element_type=ir.ElementType.INT32,
        weights=np.array([1, 0], np.int32),
    )
    y_t = subgraph.add_tensor(shape=(2, 3), element_type=F32, name="y_t")
    subgraph.add_op(ir.OpCode.TRANSPOSE, [y, perm], [y_t], index=add.index)
    add.set_input(1, y_t)
    # Negate the result: out' = NEG(out), replacing the subgraph output.
    (out,) = subgraph.outputs
    neg_out = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    neg = subgraph.add_op(ir.OpCode.NEG, [out], [neg_out], index=mul.index + 1)
    subgraph.replace_all_uses(out, neg_out, exclude=[neg])
    model.infer_shapes(validation_only=True)
    model.validate()
    result = _run(model.serialize(), "serving", x=self.x, y=self.y.T.copy())
    np.testing.assert_allclose(result["out"], -(self.x + self.y) * self.c)

  def test_dce_after_bypassing_op(self):
    model, subgraph = _build()
    add, mul = subgraph.ops
    # Bypass ADD: MUL now reads x directly; ADD becomes dead.
    mul.set_input(0, subgraph.inputs[0])
    self.assertTrue(subgraph.dce())
    self.assertEqual([op.name for op in subgraph.ops], ["MUL"])
    with self.assertRaises(ir.LiteRtError):
      _ = add.name
    model.validate()
    result = _run(model.serialize(), "serving", x=self.x, y=self.y)
    np.testing.assert_allclose(result["out"], self.x * self.c)


if __name__ == "__main__":
  googletest.main()
