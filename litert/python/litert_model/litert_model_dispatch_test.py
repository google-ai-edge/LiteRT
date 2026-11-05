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

"""Tests for litert_model: shape inference, dispatch ops and assets."""

import gc
import pathlib
import struct

import numpy as np

from absl.testing import absltest as googletest
from litert.python.aot.core import common
from litert.python.litert_model import litert_model as ir

# pylint: disable=g-import-not-at-top
try:
  from litert.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import
except ImportError:
  from tflite.python import schema_py_generated as schema_fb  # pyrefly: ignore[missing-import]
# pylint: enable=g-import-not-at-top

_TESTDATA = pathlib.Path("test/testdata")
F32 = ir.ElementType.FLOAT32
# Layout of the build stamp metadata (litert/core/build_stamp.cc).
_STAMP_FIELD_SIZE = 125


def _path(name: str) -> str:
  return str(common.get_resource(_TESTDATA / name))


def _schema_model(data: bytes) -> schema_fb.ModelT:
  return schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAs(data, 0))


def _add_function(
    model: ir.Model, name: str, constant: float, size: int
) -> ir.Subgraph:
  """Adds subgraph `name`(x, y) = (x + y) * constant with a signature."""
  subgraph = model.add_subgraph(name)
  x = subgraph.add_tensor(shape=(size,), element_type=F32, name=f"{name}_x")
  y = subgraph.add_tensor(shape=(size,), element_type=F32, name=f"{name}_y")
  total = subgraph.add_tensor(shape=(size,), element_type=F32)
  c = subgraph.add_tensor(
      shape=(size,),
      element_type=F32,
      weights=np.full((size,), constant, np.float32),
  )
  out = subgraph.add_tensor(shape=(size,), element_type=F32, name=f"{name}_o")
  subgraph.add_op(
      ir.OpCode.ADD, [x, y], [total], builtin_options=schema_fb.AddOptionsT()
  )
  subgraph.add_op(
      ir.OpCode.MUL, [total, c], [out], builtin_options=schema_fb.MulOptionsT()
  )
  model.add_signature(name, subgraph, [("x", x), ("y", y)], [("out", out)])
  return subgraph


class ShapeInferenceTest(googletest.TestCase):

  def test_infer_shapes_restores_cleared_shapes(self):
    model = ir.Model.load_from_file(_path("island_partial.tflite"))
    subgraph = model.subgraph(0)
    intermediates = [t for t in subgraph.tensors if t.defining_op is not None]
    self.assertNotEmpty(intermediates)
    expected = {t: t.shape for t in intermediates}
    for tensor in intermediates:
      tensor.set_shape([-1] * len(tensor.shape))
    model.infer_shapes()
    self.assertEqual({t: t.shape for t in intermediates}, expected)
    for tensor in intermediates:
      tensor.set_shape([])
    subgraph.infer_shapes()
    self.assertEqual({t: t.shape for t in intermediates}, expected)
    model.infer_shapes(validation_only=True)
    model.validate()

  def test_validation_only_reports_failing_op(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    subgraph = model.subgraph(0)
    (add,) = subgraph.ops
    (out,) = add.outputs
    out.set_shape([3, 3])
    for infer in (model.infer_shapes, subgraph.infer_shapes):
      with self.assertRaises(ir.ShapeInferenceError) as error:
        infer(validation_only=True)
      self.assertIsInstance(error.exception, ir.LiteRtError)
      self.assertEqual(error.exception.op, add)
      self.assertIsInstance(error.exception.op, ir.Op)
      self.assertEqual(
          error.exception.status, "kLiteRtStatusErrorShapeInferenceFailed"
      )
      self.assertIn("ADD", str(error.exception))
      self.assertEqual(out.shape, (3, 3))  # Not modified.
    model.infer_shapes()
    self.assertEqual(out.shape, (2, 2))

  def test_unsupported_op(self):
    model = ir.Model.load_from_file(_path("dynamic_string_custom_ops.tflite"))
    with self.assertRaises(ir.ShapeInferenceError) as error:
      model.infer_shapes()
    self.assertEqual(
        error.exception.status, "kLiteRtStatusErrorUnsupportedOpShapeInferer"
    )
    self.assertEqual(error.exception.op.code, ir.OpCode.CUSTOM)

  def test_specialize_subgraph(self):
    model = ir.Model.load_from_file(_path("dynamic_add_model.tflite"))
    dynamic = model.subgraph(0)
    specialized = model.specialize_subgraph(dynamic, [[4, 2, 3], [4, 2, 3]])
    self.assertEqual(specialized.index, 1)
    self.assertLen(model.subgraphs, 2)
    self.assertEqual(specialized.outputs[0].shape, (4, 2, 3))
    self.assertEqual([op.name for op in specialized.ops], ["ADD"])
    self.assertEqual(dynamic.outputs[0].shape, (-1, 2, 3))  # Unchanged.
    self.assertLen(model.signatures, 1)
    model.validate()
    with self.assertRaisesRegex(ValueError, "expected 2 input shapes"):
      model.specialize_subgraph(dynamic, [[4, 2, 3]])
    with self.assertRaises(ir.ShapeInferenceError):
      model.specialize_subgraph(dynamic, [[4, 2, 3], [5, 7, 3]])
    self.assertLen(model.subgraphs, 2)  # The failed copy was dropped.
    model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.subgraph(1).outputs[0].shape, (4, 2, 3))


class DispatchTest(googletest.TestCase):

  def _build_compiled_model(
      self, shared_asset: bool
  ) -> tuple[bytes, dict[str, bytes]]:
    model = ir.Model.create()
    subgraphs = [
        _add_function(model, "sig_a", 2.0, 3),
        _add_function(model, "sig_b", 3.0, 5),
    ]
    model.validate()
    # Odd sizes so that the 8-byte alignment of appended assets matters.
    if shared_asset:
      assets = {"sig_a": b"SHARED-BYTECODE", "sig_b": b"SHARED-BYTECODE"}
      buffer_id = model.register_buffer(assets["sig_a"])
      buffer_ids = {"sig_a": buffer_id, "sig_b": buffer_id}
    else:
      assets = {"sig_a": b"bytecode-A", "sig_b": b"bytecode-for-B!!!"}
      buffer_ids = {
          key: model.register_buffer(asset) for key, asset in assets.items()
      }
    for subgraph in subgraphs:
      tensors_before = subgraph.tensors
      op = subgraph.outline_to_dispatch_op()
      self.assertEqual(subgraph.ops, [op])
      self.assertTrue(op.is_dispatch_op)
      self.assertEqual(op.code, ir.OpCode.CUSTOM)
      self.assertEqual(op.custom_code, ir.DISPATCH_OP_CUSTOM_CODE)
      self.assertEqual(op.inputs, subgraph.inputs)
      self.assertEqual(op.outputs, subgraph.outputs)
      self.assertIsNone(op.dispatch_options())  # Not serialized yet.
      self.assertCountEqual(
          subgraph.tensors, subgraph.inputs + subgraph.outputs
      )
      for removed in set(tensors_before) - set(subgraph.tensors):
        with self.assertRaises(ir.LiteRtError):
          _ = removed.shape
      model.attach_asset(op, buffer_ids[subgraph.name], subgraph.name)
    model.set_build_stamp("TestVendor", "TestSocModel")
    self.assertEqual(model.build_stamp(), ("TestVendor", "TestSocModel"))
    model.validate()
    return model.serialize(bytecode_alignment=8), assets

  def _check_compiled_model(
      self, data: bytes, assets: dict[str, bytes], model: ir.Model
  ):
    model.validate()
    self.assertEqual(model.build_stamp(), ("TestVendor", "TestSocModel"))
    self.assertEqual([s.key for s in model.signatures], ["sig_a", "sig_b"])
    offsets = set()
    for signature, size in zip(model.signatures, (3, 5)):
      self.assertEqual(signature.input_names, ["x", "y"])
      self.assertEqual(signature.output_names, ["out"])
      subgraph = signature.subgraph
      (op,) = subgraph.ops
      self.assertEqual(op.custom_code, ir.DISPATCH_OP_CUSTOM_CODE)
      self.assertTrue(op.is_dispatch_op)
      self.assertEqual(op.inputs, signature.inputs)
      self.assertEqual(op.outputs, signature.outputs)
      self.assertEqual([t.shape for t in op.inputs], [(size,), (size,)])
      options = op.dispatch_options()
      self.assertIsInstance(options, ir.DispatchOpOptions)
      self.assertEqual(options.name, signature.key)
      asset = assets[signature.key]
      self.assertLen(asset, options.bytecode_size)
      self.assertEqual(options.bytecode_offset % 8, 0)
      self.assertEqual(
          data[options.bytecode_offset : options.bytecode_offset + len(asset)],
          asset,
      )
      offsets.add(options.bytecode_offset)
    return offsets

  def test_compile_flow_stress_with_live_views(self):
    # Mimics a compiler plugin: build/edit, keep weight views alive, outline,
    # register + attach assets, stamp, serialize and drop the model.
    views = []
    for iteration in range(40):
      model = ir.Model.create()
      subgraphs = [
          _add_function(model, f"sig_{k}", float(iteration + k), 3 + k)
          for k in range(3)
      ]
      for subgraph in subgraphs:
        for op in subgraph.ops:
          for tensor in op.inputs:
            if tensor.is_constant:
              tensor.set_weights(np.asarray(tensor.weights) + 1.0)
              views.append((tensor.weights, float(iteration) + 1.0))
      for k, subgraph in enumerate(subgraphs):
        op = subgraph.outline_to_dispatch_op()
        buffer_id = model.register_buffer(bytes([k]) * (17 + iteration))
        model.attach_asset(op, buffer_id, subgraph.name)
      model.set_build_stamp("TestVendor", "TestSocModel")
      data = model.serialize(bytecode_alignment=8)
      self.assertTrue(model.is_serialized)
      del model, subgraphs
      reloaded = ir.Model.load_from_bytes(data)
      self.assertLen(reloaded.signatures, 3)
      del reloaded
    gc.collect()
    for view, first in views:
      self.assertGreaterEqual(view[0], first)
      self.assertFalse(view.flags.writeable)

  def test_outline_attach_serialize_reload(self):
    data, assets = self._build_compiled_model(shared_asset=False)
    offsets = self._check_compiled_model(
        data, assets, ir.Model.load_from_bytes(data)
    )
    self.assertLen(offsets, 2)
    # Cross-check with the raw flatbuffer.
    schema_model = _schema_model(data)
    custom_codes = [
        c.customCode
        for c in schema_model.operatorCodes
        if c.builtinCode == schema_fb.BuiltinOperator.CUSTOM
    ]
    self.assertEqual(custom_codes, [b"DISPATCH_OP"])
    for subgraph in schema_model.subgraphs:
      self.assertLen(subgraph.operators, 1)
      self.assertLen(subgraph.tensors, 3)  # No intermediates or constants.
    stamp = next(
        schema_model.buffers[m.buffer].data
        for m in schema_model.metadata
        if m.name == ir.BUILD_STAMP_METADATA_KEY.encode()
    )
    manufacturer, soc_model = struct.unpack(
        f"{_STAMP_FIELD_SIZE}s{_STAMP_FIELD_SIZE}s", bytes(stamp)
    )
    self.assertEqual(manufacturer.rstrip(b"\x00"), b"TestVendor")
    self.assertEqual(soc_model.rstrip(b"\x00"), b"TestSocModel")

  def test_shared_asset_is_appended_once(self):
    data, assets = self._build_compiled_model(shared_asset=True)
    offsets = self._check_compiled_model(
        data, assets, ir.Model.load_from_bytes(data)
    )
    self.assertLen(offsets, 1)

  def test_reload_from_file_and_reserialize(self):
    data, assets = self._build_compiled_model(shared_asset=False)
    path = self.create_tempfile("compiled.tflite", content=data, mode="wb")
    model = ir.Model.load_from_file(path.full_path)
    self._check_compiled_model(data, assets, model)
    # Assets of dispatch ops are attached at load time, so a re-serialized
    # model carries them again.
    data2 = model.serialize(bytecode_alignment=16)
    reloaded = ir.Model.load_from_bytes(data2)
    for signature in reloaded.signatures:
      options = signature.subgraph.ops[0].dispatch_options()
      self.assertEqual(options.bytecode_offset % 16, 0)
      self.assertEqual(
          data2[
              options.bytecode_offset : options.bytecode_offset
              + options.bytecode_size
          ],
          assets[signature.key],
      )

  def test_outline_real_model(self):
    model = ir.Model.load_from_file(_path("simple_conv_2d_op.tflite"))
    subgraph = model.subgraph(0)
    op = subgraph.outline_to_dispatch_op()
    self.assertEqual(op.inputs, subgraph.inputs)
    model.attach_asset(op, model.register_buffer(b"\x01" * 33), "main")
    model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize(bytecode_alignment=8))
    (signature,) = reloaded.signatures
    (dispatch,) = signature.subgraph.ops
    self.assertEqual(dispatch.dispatch_options().bytecode_size, 33)
    self.assertEqual(
        [t.shape for t in dispatch.inputs],
        [(1, 216, 288, 24), (24, 3, 3, 24), (24,)],
    )

  def test_outline_forwarded_constant_and_repeated_outputs(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph("main")
    x = subgraph.add_tensor(shape=(2,), element_type=F32, name="x")
    y = subgraph.add_tensor(shape=(2,), element_type=F32, name="y")
    cst = subgraph.add_tensor(
        shape=(3,),
        element_type=ir.ElementType.INT8,
        name="cst",
        weights=np.array([1, 2, 3], np.int8),
        quantization=ir.PerChannelQuantization(
            scales=[0.1, 0.2, 0.3], zero_points=[0, 0, 0], quantized_dimension=0
        ),
    )
    total = subgraph.add_tensor(shape=(2,), element_type=F32, name="sum")
    subgraph.add_op(ir.OpCode.ADD, [x, y], [total])
    # x is forwarded and also used by ADD; sum is returned twice.
    model.add_signature(
        "serving",
        subgraph,
        [("x", x), ("y", y)],
        [("a", total), ("b", x), ("c", cst), ("d", total)],
    )
    model.validate()
    op = subgraph.outline_to_dispatch_op()
    outputs = subgraph.outputs
    self.assertEqual(op.outputs, outputs)
    self.assertEqual(op.inputs, [x, y])
    self.assertEqual(subgraph.inputs, [x, y])
    self.assertEqual(outputs[0], total)
    self.assertLen(set(outputs), 4)
    self.assertNotIn(x, outputs)
    self.assertEqual(
        [(t.name, t.shape, t.element_type) for t in outputs],
        [
            ("sum", (2,), F32),
            ("x", (2,), F32),
            ("cst", (3,), ir.ElementType.INT8),
            ("sum", (2,), F32),
        ],
    )
    self.assertFalse(outputs[2].is_constant)
    self.assertEqual(
        outputs[2].quantization,
        ir.PerChannelQuantization([0.1, 0.2, 0.3], [0, 0, 0], 0),
    )
    self.assertTrue(all(t.defining_op == op for t in outputs))
    with self.assertRaises(ir.LiteRtError):
      _ = cst.name  # The constant became unused and was removed.
    self.assertEqual(x.users, [(op, 0)])
    (signature,) = model.signatures
    self.assertEqual(signature.outputs, outputs)
    self.assertEqual(signature.output_names, ["a", "b", "c", "d"])
    self.assertEqual(signature.get_output("b"), outputs[1])
    model.attach_asset(op, model.register_buffer(b"code"), "serving")
    model.validate()

    data = model.serialize(bytecode_alignment=8)
    reloaded = ir.Model.load_from_bytes(data)
    reloaded.validate()
    (signature,) = reloaded.signatures
    self.assertEqual(signature.output_names, ["a", "b", "c", "d"])
    (dispatch,) = signature.subgraph.ops
    self.assertEqual(dispatch.outputs, signature.outputs)
    self.assertEqual(dispatch.inputs, signature.inputs)
    self.assertEqual(
        [t.shape for t in signature.outputs], [(2,), (2,), (3,), (2,)]
    )
    self.assertEqual(
        signature.get_output("c").quantization,
        ir.PerChannelQuantization([0.1, 0.2, 0.3], [0, 0, 0], 0),
    )
    self.assertEqual(dispatch.dispatch_options().bytecode_size, 4)
    self.assertLen(signature.subgraph.tensors, 6)

  def test_outline_real_model_with_constant_output(self):
    model = ir.Model.load_from_file(_path("constant_output_tensor.tflite"))
    (signature,) = model.signatures
    names = signature.output_names
    subgraph = signature.subgraph
    op = subgraph.outline_to_dispatch_op()
    self.assertEqual(op.outputs, subgraph.outputs)
    self.assertEqual(signature.output_names, names)
    self.assertTrue(all(not t.is_constant for t in subgraph.outputs))
    model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.signatures[0].output_names, names)
    self.assertLen(reloaded.subgraph(0).ops[0].outputs, len(names))

  def test_outline_placeholder_signature_repeated_output(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph("main")
    x = subgraph.add_tensor(shape=(2,), element_type=F32, name="x")
    y = subgraph.add_tensor(shape=(2,), element_type=F32, name="y")
    subgraph.add_op(ir.OpCode.RELU, [x], [y])
    subgraph.set_inputs([x])
    subgraph.set_outputs([y, y, x])
    model = ir.Model.load_from_bytes(model.serialize())
    (signature,) = model.signatures
    self.assertTrue(signature.is_placeholder)
    self.assertEqual(signature.output_names, ["y", "y", "x"])
    op = model.subgraph(0).outline_to_dispatch_op()
    self.assertLen(set(op.outputs), 3)
    self.assertEqual(signature.output_names, ["y", "y_1", "x"])
    self.assertEqual(signature.outputs, op.outputs)
    model.attach_asset(op, model.register_buffer(b"code"), "main")
    reloaded = ir.Model.load_from_bytes(model.serialize(bytecode_alignment=8))
    reloaded.validate()
    (signature,) = reloaded.signatures
    self.assertEqual(signature.key, ir.DEFAULT_SIGNATURE_KEY)
    self.assertEqual(signature.output_names, ["y", "y_1", "x"])
    (dispatch,) = reloaded.subgraph(0).ops
    self.assertEqual(dispatch.outputs, signature.outputs)
    self.assertLen(set(dispatch.outputs), 3)

  def test_default_signature_key(self):
    self.assertEqual(ir.DEFAULT_SIGNATURE_KEY, "<placeholder signature>")
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    (signature,) = model.signatures
    self.assertEqual(signature.key, ir.DEFAULT_SIGNATURE_KEY)
    self.assertTrue(signature.is_placeholder)
    self.assertEqual(model.find_signature(ir.DEFAULT_SIGNATURE_KEY), signature)
    other = ir.Model.create()
    _add_function(other, "serving", 1.0, 2)
    self.assertFalse(other.signatures[0].is_placeholder)

  def test_attach_asset_errors(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    (add,) = model.subgraph(0).ops
    buffer_id = model.register_buffer(b"code")
    with self.assertRaisesRegex(ValueError, "dispatch"):
      model.attach_asset(add, buffer_id, "main")
    op = model.subgraph(0).outline_to_dispatch_op()
    with self.assertRaisesRegex(ValueError, "buffer id"):
      model.attach_asset(op, 12345, "main")
    with self.assertRaisesRegex(ValueError, "buffer id"):
      model.attach_asset(op, 0, "main")
    with self.assertRaisesRegex(ValueError, "name"):
      model.attach_asset(op, buffer_id, "")
    other = ir.Model.load_from_file(_path("add_simple.tflite"))
    with self.assertRaisesRegex(ValueError, "different model"):
      other.attach_asset(op, buffer_id, "main")
    with self.assertRaises(ir.LiteRtError):
      _ = add.code  # Removed by outlining.

  def test_build_stamp_errors(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    self.assertIsNone(model.build_stamp())
    with self.assertRaises(ir.LiteRtError) as error:
      model.set_build_stamp("x" * 200, "model")
    self.assertEqual(
        error.exception.status, "kLiteRtStatusErrorInvalidArgument"
    )
    model.set_build_stamp("TestVendor", "TestSocModel")
    with self.assertRaises(ir.LiteRtError) as error:
      model.set_build_stamp("TestVendor", "TestSocModel")
    self.assertEqual(error.exception.status, "kLiteRtStatusErrorAlreadyExists")
    self.assertIn(ir.BUILD_STAMP_METADATA_KEY, model.metadata_keys())


if __name__ == "__main__":
  googletest.main()
