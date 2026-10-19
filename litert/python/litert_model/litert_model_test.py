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

"""Tests for litert_model: enums, loading/inspection and graph mutation."""

import gc
import pathlib
import re
from typing import Any, cast

import numpy as np

from absl.testing import absltest as googletest
from litert.python.aot.core import common
from litert.python.litert_model import _pywrap_litert_model
from litert.python.litert_model import litert_model as ir
from litert.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import

_TESTDATA = pathlib.Path("test/testdata")
_C_HEADERS = pathlib.Path("c")
F32 = ir.ElementType.FLOAT32


def _path(name: str) -> str:
  return str(common.get_resource(_TESTDATA / name))


def _read(name: str) -> bytes:
  with open(_path(name), "rb") as f:
    return f.read()


def _c_enumerators(header: str, enum_name: str) -> list[str]:
  """Returns the enumerator names of `typedef enum {...} enum_name;`."""
  path = common.get_resource(_C_HEADERS / header)
  with open(path) as f:
    source = f.read()
  match = re.search(
      r"typedef enum \{(.*?)\}\s*" + enum_name + ";", source, re.DOTALL
  )
  assert match is not None, f"{enum_name} not found in {header}"
  body = re.sub(r"//[^\n]*", "", match.group(1))
  return re.findall(r"\b(kLiteRt\w+)\s*=", body)


def _schema_model(data: bytes) -> schema_fb.ModelT:
  return schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAs(data, 0))


def _build_add_mul_model() -> tuple[ir.Model, dict[str, object]]:
  """Builds main(x, y) = (x + y) * c with signature "serving"."""
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
  add = subgraph.add_op(
      ir.OpCode.ADD, [x, y], [total], builtin_options=schema_fb.AddOptionsT()
  )
  mul = subgraph.add_op(
      ir.OpCode.MUL, [total, c], [out], builtin_options=schema_fb.MulOptionsT()
  )
  model.add_signature("serving", subgraph, [("x", x), ("y", y)], [("out", out)])
  return model, dict(
      subgraph=subgraph, x=x, y=y, total=total, c=c, out=out, add=add, mul=mul
  )


def _repeated_output_model_bytes() -> bytes:
  """A model without signature whose subgraph returns `y` twice."""
  model = ir.Model.create()
  subgraph = model.add_subgraph("main")
  x = subgraph.add_tensor(shape=(2,), element_type=F32, name="x")
  y = subgraph.add_tensor(shape=(2,), element_type=F32, name="y")
  subgraph.add_op(ir.OpCode.RELU, [x], [y])
  subgraph.set_inputs([x])
  subgraph.set_outputs([y, y])
  return model.serialize()


class EnumTest(googletest.TestCase):

  def test_cpp_lists_match_c_headers(self):
    for header, enum_name, cpp in (
        (
            "litert_model_types.h",
            "LiteRtElementType",
            _pywrap_litert_model.element_type_values(),
        ),
        (
            "litert_op_code.h",
            "LiteRtOpCode",
            _pywrap_litert_model.op_code_values(),
        ),
    ):
      with self.subTest(enum_name):
        self.assertCountEqual(_c_enumerators(header, enum_name), cpp.keys())

  def test_element_type_matches_cpp(self):
    cpp = _pywrap_litert_model.element_type_values()
    self.assertCountEqual(cpp.values(), [int(t) for t in ir.ElementType])
    for cpp_name, value in cpp.items():
      python_name = ir.ElementType(value).name
      self.assertEqual(
          cpp_name.removeprefix("kLiteRtElementType").upper(),
          python_name.replace("_", ""),
      )

  def test_op_code_matches_cpp_and_tflite_schema(self):
    cpp = _pywrap_litert_model.op_code_values()
    self.assertLen(cpp, len(ir.OpCode))
    self.assertCountEqual(cpp.values(), [int(c) for c in ir.OpCode])
    for code in ir.OpCode:
      self.assertEqual(getattr(schema_fb.BuiltinOperator, code.name), code)
    self.assertEqual(cpp["kLiteRtOpCodeTflAdd"], ir.OpCode.ADD)
    self.assertEqual(cpp["kLiteRtOpCodeTflCustom"], ir.OpCode.CUSTOM)
    self.assertEqual(
        cpp["kLiteRtOpCodeShloComposite"], ir.OpCode.STABLEHLO_COMPOSITE
    )

  def test_unknown_op_code(self):
    code = ir.OpCode(1000)
    self.assertEqual(code, 1000)
    self.assertEqual(code.name, "UNKNOWN_1000")
    with self.assertRaises(ValueError):
      ir.OpCode(-1)

  def test_element_type_numpy_helpers(self):
    self.assertEqual(ir.ElementType.from_numpy(np.float32), F32)
    self.assertEqual(ir.ElementType.from_numpy("int8"), ir.ElementType.INT8)
    self.assertEqual(ir.ElementType.from_numpy(np.bool_), ir.ElementType.BOOL)
    self.assertEqual(F32.numpy_dtype, np.float32)
    self.assertIsNone(ir.ElementType.INT4.numpy_dtype)
    self.assertIsNone(ir.ElementType.BFLOAT16.numpy_dtype)
    self.assertEqual(ir.ElementType.INT4.bit_width, 4)
    self.assertEqual(ir.ElementType.INT2.bit_width, 2)
    self.assertEqual(ir.ElementType.FLOAT16.bit_width, 16)
    self.assertEqual(ir.ElementType.COMPLEX128.bit_width, 128)
    self.assertEqual(ir.ElementType.TF_STRING.bit_width, 0)
    with self.assertRaises(TypeError):
      ir.ElementType.from_numpy(np.str_)


class LoadInspectTest(googletest.TestCase):

  def test_load_add_simple(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    self.assertLen(model.subgraphs, 1)
    subgraph = model.subgraph(0)
    self.assertEqual(subgraph.index, 0)
    (op,) = subgraph.ops
    self.assertEqual(op.code, ir.OpCode.ADD)
    self.assertEqual(op.name, "ADD")
    self.assertEqual(op.index, 0)
    self.assertIsNone(op.custom_code)
    (arg0,) = subgraph.inputs
    self.assertEqual(op.inputs, [arg0, arg0])
    self.assertEqual(arg0.shape, (2, 2))
    self.assertEqual(arg0.element_type, F32)
    self.assertTrue(arg0.is_ranked)
    self.assertTrue(arg0.is_subgraph_input)
    self.assertFalse(arg0.is_constant)
    self.assertIsNone(arg0.weights)
    self.assertIsNone(arg0.quantization)
    self.assertEqual(op.outputs, subgraph.outputs)
    options = op.builtin_options
    self.assertIsInstance(options, schema_fb.AddOptionsT)
    self.assertEqual(
        options.fusedActivationFunction, schema_fb.ActivationFunctionType.NONE
    )
    (signature,) = model.signatures
    self.assertEqual(signature.subgraph, subgraph)
    self.assertEqual(signature.input_names, ["arg0"])
    self.assertEqual(signature.output_names, ["tfl.add"])
    self.assertEqual(signature.inputs, [arg0])
    model.validate()

  def test_load_from_bytes_matches_file(self):
    data = _read("simple_conv_2d_op.tflite")
    for source in (data, bytearray(data), memoryview(data)):
      model = ir.Model.load_from_bytes(source)
      (op,) = model.subgraph(0).ops
      self.assertEqual(op.code, ir.OpCode.CONV_2D)
      self.assertEqual(
          [t.shape for t in op.inputs],
          [(1, 216, 288, 24), (24, 3, 3, 24), (24,)],
      )
    with self.assertRaises(ir.LiteRtError):
      ir.Model.load_from_bytes(b"not a model")

  def test_load_from_missing_file(self):
    with self.assertRaises(ir.LiteRtError):
      ir.Model.load_from_file("/nonexistent/model.tflite")

  def test_def_use(self):
    model = ir.Model.load_from_file(_path("simple_model.tflite"))
    subgraph = model.subgraph(0)
    (add,) = subgraph.ops
    arg0, arg1 = subgraph.inputs
    (out,) = subgraph.outputs
    self.assertEqual(arg0.users, [(add, 0)])
    self.assertEqual(arg1.users, [(add, 1)])
    self.assertIsNone(arg0.defining_op)
    self.assertEqual(arg0.defining_op_output_index, -1)
    self.assertEqual(out.defining_op, add)
    self.assertEqual(out.defining_op_output_index, 0)
    self.assertEqual(out.users, [])
    self.assertTrue(out.is_subgraph_output)
    self.assertCountEqual(subgraph.tensors, [arg0, arg1, out])
    self.assertEqual([t.index for t in subgraph.tensors], [0, 1, 2])

  def test_weights_zero_copy_view(self):
    data = _read("multi_subgraph.tflite")
    schema_model = _schema_model(data)
    model = ir.Model.load_from_file(_path("multi_subgraph.tflite"))
    expected = {0: -1.0, 1: 1.0, 2: 2.0}
    for subgraph in model.subgraphs:
      (op,) = subgraph.ops
      cst = op.inputs[1]
      self.assertTrue(cst.is_constant)
      weights = cst.weights
      self.assertEqual(weights.dtype, np.float32)
      self.assertEqual(weights.shape, (4,))
      np.testing.assert_array_equal(
          weights, np.full((4,), expected[subgraph.index], np.float32)
      )
      self.assertFalse(weights.flags.writeable)
      with self.assertRaises(ValueError):
        weights[0] = 3.0
      # Same memory on every access (no copy), equal to the flatbuffer data.
      self.assertEqual(
          weights.__array_interface__["data"][0],
          cst.weights.__array_interface__["data"][0],
      )
      raw = cst.weights_bytes
      self.assertIsInstance(raw, memoryview)
      self.assertTrue(raw.readonly)
      schema_tensor = schema_model.subgraphs[subgraph.index].tensors[cst.index]
      self.assertEqual(
          bytes(raw), bytes(schema_model.buffers[schema_tensor.buffer].data)
      )

  def test_handle_identity(self):
    model = ir.Model.load_from_file(_path("simple_model.tflite"))
    subgraph = model.subgraph(0)
    op_a = subgraph.ops[0]
    op_b = subgraph.ops[0]
    self.assertIsNot(op_a, op_b)
    self.assertEqual(op_a, op_b)
    self.assertEqual(hash(op_a), hash(op_b))
    self.assertEqual({op_a: 1}[op_b], 1)
    self.assertEqual(subgraph.inputs[0], op_a.inputs[0])
    self.assertNotEqual(subgraph.inputs[0], subgraph.inputs[1])
    self.assertNotEqual(op_a, subgraph.inputs[0])
    self.assertEqual(model.subgraph(0), op_a.subgraph)
    self.assertEqual(model.signatures[0], model.signatures[0])
    other = ir.Model.load_from_file(_path("simple_model.tflite"))
    self.assertNotEqual(other.subgraph(0).ops[0], op_a)

  def test_handles_keep_model_alive(self):
    model = ir.Model.load_from_file(_path("multi_subgraph.tflite"))
    cst = model.subgraph(0).ops[0].inputs[1]
    weights = cst.weights
    signature = model.signatures[0]
    del model
    gc.collect()
    self.assertEqual(cst.shape, (4,))
    np.testing.assert_array_equal(cst.weights, weights)
    np.testing.assert_array_equal(weights, np.full((4,), -1.0, np.float32))
    self.assertEqual(signature.output_names, ["tfl.add"])

  def test_per_tensor_quantization(self):
    data = _read("simple_quantized_ops.tflite")
    schema_model = _schema_model(data)
    model = ir.Model.load_from_bytes(data)
    subgraph = model.subgraph(0)
    found = 0
    for tensor in subgraph.tensors:
      schema_q = schema_model.subgraphs[0].tensors[tensor.index].quantization
      quantization = tensor.quantization
      if quantization is None:
        continue
      found += 1
      self.assertIsInstance(quantization, ir.PerTensorQuantization)
      self.assertAlmostEqual(quantization.scale, schema_q.scale[0], places=6)
      self.assertEqual(quantization.zero_point, schema_q.zeroPoint[0])
    self.assertGreater(found, 0)

  def test_per_channel_quantization(self):
    data = _read("static_w8_a16_quantized_k_einsum.tflite")
    schema_model = _schema_model(data)
    model = ir.Model.load_from_bytes(data)
    found = 0
    for tensor in model.subgraph(0).tensors:
      quantization = tensor.quantization
      if not isinstance(quantization, ir.PerChannelQuantization):
        continue
      found += 1
      schema_q = schema_model.subgraphs[0].tensors[tensor.index].quantization
      np.testing.assert_allclose(quantization.scales, schema_q.scale)
      np.testing.assert_array_equal(
          quantization.zero_points, schema_q.zeroPoint
      )
      self.assertEqual(
          quantization.quantized_dimension, schema_q.quantizedDimension
      )
      self.assertLen(
          quantization.scales,
          tensor.shape[quantization.quantized_dimension],
      )
      self.assertEqual(tensor.weights.shape, tensor.shape)
    self.assertGreater(found, 0)

  def test_block_wise_quantization_load(self):
    data = _read("blockwise_quantized.tflite")
    schema_model = _schema_model(data)
    model = ir.Model.load_from_bytes(data)
    found = 0
    for tensor in model.subgraph(0).tensors:
      quantization = tensor.quantization
      if not isinstance(quantization, ir.BlockWiseQuantization):
        continue
      found += 1
      details = (
          schema_model.subgraphs[0].tensors[tensor.index].quantization.details
      )
      self.assertEqual(quantization.block_size, details.blockSize)
      self.assertEqual(quantization.scales.index, details.scales)
      if details.zeroPoints == -1:
        self.assertIsNone(quantization.zero_points)
      else:
        self.assertEqual(quantization.zero_points.index, details.zeroPoints)
    self.assertGreater(found, 0)

  def test_signatures(self):
    model = ir.Model.load_from_file(_path("reverse_signature_model.tflite"))
    (signature,) = model.signatures
    self.assertEqual(signature.key, "serving_default")
    self.assertEqual(model.find_signature("serving_default"), signature)
    with self.assertRaises(KeyError):
      model.find_signature("missing")
    subgraph = signature.subgraph
    # Names follow the subgraph I/O order.
    self.assertEqual(signature.inputs, subgraph.inputs)
    self.assertEqual(signature.outputs, subgraph.outputs)
    self.assertCountEqual(signature.input_names, ["x", "y"])
    x = signature.get_input("x")
    self.assertIn(x, subgraph.inputs)
    self.assertEqual(signature.input_names[subgraph.inputs.index(x)], "x")
    self.assertIn(signature.get_output("sum"), subgraph.outputs)
    with self.assertRaises(KeyError):
      signature.get_input("z")

  def test_multi_subgraph_composite(self):
    model = ir.Model.load_from_file(_path("simple_shlo_composite.tflite"))
    self.assertLen(model.subgraphs, 2)
    (composite,) = model.subgraph(0).ops
    self.assertEqual(composite.code, ir.OpCode.STABLEHLO_COMPOSITE)
    options = composite.builtin_options
    self.assertIsInstance(options, schema_fb.StableHLOCompositeOptionsT)
    self.assertEqual(options.name, b"stablehlo.add_n")
    self.assertEqual(options.decompositionSubgraphIndex, 1)
    self.assertEqual(options.version, 3)
    model.validate()

  def test_dynamic_shapes(self):
    model = ir.Model.load_from_file(_path("dynamic_add_model.tflite"))
    for tensor in model.subgraph(0).inputs:
      self.assertEqual(tensor.shape, (-1, 2, 3))
    reloaded = ir.Model.load_from_bytes(model.serialize())
    for tensor in reloaded.subgraph(0).tensors:
      self.assertEqual(tensor.shape, (-1, 2, 3))

  def test_metadata(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    self.assertIn("min_runtime_version", model.metadata_keys())
    self.assertIsInstance(model.metadata("min_runtime_version"), bytes)
    self.assertIsNone(model.metadata("missing"))
    model.push_metadata("custom", b"\x00\x01payload")
    self.assertEqual(model.metadata("custom"), b"\x00\x01payload")
    self.assertEqual(model.metadata_keys(), sorted(model.metadata_keys()))
    with self.assertRaises(ir.LiteRtError) as error:
      model.push_metadata("custom", b"other")
    self.assertEqual(error.exception.status, "kLiteRtStatusErrorAlreadyExists")
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.metadata("custom"), b"\x00\x01payload")

  def test_repr(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    self.assertIn("subgraphs=1", repr(model))
    self.assertIn("ADD", repr(model.subgraph(0).ops[0]))
    self.assertIn("FLOAT32", repr(model.subgraph(0).inputs[0]))
    self.assertIn("arg0", repr(model.signatures[0]))


class BuiltinOptionsTest(googletest.TestCase):

  def test_options_created_with_op_round_trip(self):
    # Options set at op creation can be read back repeatedly.
    model = ir.Model.create()
    subgraph = model.add_subgraph("main")
    a, b, y = [
        subgraph.add_tensor(shape=(2,), element_type=F32) for _ in range(3)
    ]
    op = subgraph.add_op(
        ir.OpCode.ADD, [a, b], [y], builtin_options=schema_fb.AddOptionsT()
    )
    for _ in range(3):
      self.assertIsInstance(op.builtin_options, schema_fb.AddOptionsT)

  def test_conv2d_options(self):
    model = ir.Model.load_from_file(_path("simple_conv_2d_op.tflite"))
    (conv,) = model.subgraph(0).ops
    options = conv.builtin_options
    self.assertIsInstance(options, schema_fb.Conv2DOptionsT)
    self.assertEqual(options.padding, schema_fb.Padding.SAME)
    self.assertEqual((options.strideH, options.strideW), (1, 1))
    options.strideW = 2
    options.fusedActivationFunction = schema_fb.ActivationFunctionType.RELU
    self.assertEqual(conv.builtin_options.strideW, 1)  # A copy was returned.
    conv.set_builtin_options(options)
    self.assertEqual(conv.builtin_options.strideW, 2)
    reloaded_data = model.serialize()
    schema_op = _schema_model(reloaded_data).subgraphs[0].operators[0]
    self.assertEqual(schema_op.builtinOptions.strideW, 2)
    self.assertEqual(
        schema_op.builtinOptions.fusedActivationFunction,
        schema_fb.ActivationFunctionType.RELU,
    )
    reloaded = ir.Model.load_from_bytes(reloaded_data)
    self.assertEqual(reloaded.subgraph(0).ops[0].builtin_options.strideW, 2)

  def test_concatenation_options(self):
    model = ir.Model.load_from_file(_path("simple_concatenation_op.tflite"))
    (concat,) = model.subgraph(0).ops
    options = concat.builtin_options
    self.assertIsInstance(options, schema_fb.ConcatenationOptionsT)
    self.assertEqual(options.axis, 2)
    options.axis = -2
    concat.set_builtin_options(options)
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.subgraph(0).ops[0].builtin_options.axis, -2)

  def test_stablehlo_composite_options(self):
    model = ir.Model.load_from_file(_path("simple_shlo_composite.tflite"))
    (composite,) = model.subgraph(0).ops
    options = composite.builtin_options
    options.version = 7
    options.name = "stablehlo.renamed"
    composite.set_builtin_options(options)
    reloaded = ir.Model.load_from_bytes(model.serialize())
    options = reloaded.subgraph(0).ops[0].builtin_options
    self.assertIsInstance(options, schema_fb.StableHLOCompositeOptionsT)
    self.assertEqual(options.version, 7)
    self.assertEqual(options.name, b"stablehlo.renamed")
    self.assertEqual(options.decompositionSubgraphIndex, 1)

  def test_clear_options(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    (add,) = model.subgraph(0).ops
    add.set_builtin_options(None)
    self.assertIsNone(add.builtin_options)
    add.set_builtin_options(schema_fb.AddOptionsT())
    self.assertIsInstance(add.builtin_options, schema_fb.AddOptionsT)

  def test_wrong_options_type(self):
    model = ir.Model.load_from_file(_path("add_simple.tflite"))
    (add,) = model.subgraph(0).ops
    with self.assertRaisesRegex(TypeError, "AddOptions"):
      add.set_builtin_options(schema_fb.Conv2DOptionsT())
    with self.assertRaisesRegex(TypeError, "not a builtin options type"):
      add.set_builtin_options(schema_fb.TensorT())
    with self.assertRaisesRegex(TypeError, "not a builtin options type"):
      add.set_builtin_options(object())
    # The op keeps its options.
    self.assertIsInstance(add.builtin_options, schema_fb.AddOptionsT)
    subgraph = model.subgraph(0)
    t = subgraph.add_tensor(shape=(2, 2), element_type=F32)
    with self.assertRaisesRegex(TypeError, "Conv2DOptions"):
      subgraph.add_op(
          ir.OpCode.CONV_2D,
          [subgraph.inputs[0]],
          [t],
          builtin_options=schema_fb.AddOptionsT(),
      )
    with self.assertRaisesRegex(TypeError, "CUSTOM"):
      subgraph.add_op(
          ir.OpCode.CUSTOM,
          [subgraph.inputs[0]],
          [t],
          custom_code="my_op",
          builtin_options=schema_fb.AddOptionsT(),
      )
    # Nothing was added by the failed calls.
    self.assertLen(subgraph.ops, 1)

  def test_options_without_op_named_type(self):
    # MAXIMUM uses MaximumMinimumOptions (not named after the op).
    model = ir.Model.create()
    subgraph = model.add_subgraph()
    a, b, c = [subgraph.add_tensor(shape=(1,), element_type=F32) for _ in "abc"]
    op = subgraph.add_op(
        ir.OpCode.MAXIMUM,
        [a, b],
        [c],
        builtin_options=schema_fb.MaximumMinimumOptionsT(),
    )
    self.assertIsInstance(op.builtin_options, schema_fb.MaximumMinimumOptionsT)


class MutationTest(googletest.TestCase):

  def test_build_from_scratch_and_reload(self):
    model, parts = _build_add_mul_model()
    model.validate()
    subgraph = parts["subgraph"]
    self.assertEqual(subgraph.inputs, [parts["x"], parts["y"]])
    self.assertEqual(subgraph.outputs, [parts["out"]])
    self.assertEqual([op.name for op in subgraph.ops], ["ADD", "MUL"])
    self.assertEqual(parts["total"].users, [(parts["mul"], 0)])
    data = model.serialize()
    self.assertTrue(model.is_serialized)

    reloaded = ir.Model.load_from_bytes(data)
    reloaded.validate()
    (signature,) = reloaded.signatures
    self.assertEqual(signature.key, "serving")
    self.assertEqual(signature.input_names, ["x", "y"])
    self.assertEqual(signature.output_names, ["out"])
    sub = reloaded.subgraph(0)
    self.assertEqual(sub.name, "main")
    add, mul = sub.ops
    self.assertEqual((add.code, mul.code), (ir.OpCode.ADD, ir.OpCode.MUL))
    self.assertIsInstance(add.builtin_options, schema_fb.AddOptionsT)
    self.assertIsInstance(mul.builtin_options, schema_fb.MulOptionsT)
    np.testing.assert_array_equal(
        mul.inputs[1].weights, np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    self.assertEqual(mul.inputs[0], add.outputs[0])
    schema_model = _schema_model(data)
    self.assertCountEqual(
        [c.builtinCode for c in schema_model.operatorCodes],
        [
            schema_fb.BuiltinOperator.ADD,
            schema_fb.BuiltinOperator.MUL,
            schema_fb.BuiltinOperator.CUSTOM,  # Reserved for dispatch ops.
        ],
    )

  def test_broadcast_to_uses_opcode_version_2(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph("main")
    x = subgraph.add_tensor(shape=(1, 3), element_type=ir.ElementType.INT32)
    shape = subgraph.add_tensor(
        shape=(2,),
        element_type=ir.ElementType.INT32,
        weights=np.array([2, 3], dtype=np.int32),
    )
    out = subgraph.add_tensor(shape=(2, 3), element_type=ir.ElementType.INT32)
    subgraph.add_op(ir.OpCode.BROADCAST_TO, [x, shape], [out])
    subgraph.set_inputs([x])
    subgraph.set_outputs([out])
    schema_model = _schema_model(model.serialize())
    versions = {c.builtinCode: c.version for c in schema_model.operatorCodes}
    self.assertEqual(versions[schema_fb.BuiltinOperator.BROADCAST_TO], 2)

  def test_add_signature_validation(self):
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    with self.assertRaisesRegex(ValueError, "already exists"):
      model.add_signature("serving", subgraph, [], [])
    with self.assertRaisesRegex(ValueError, "1 inputs but the subgraph has 2"):
      model.add_signature(
          "other", subgraph, [("x", parts["x"])], [("out", parts["out"])]
      )
    with self.assertRaisesRegex(ValueError, "missing from the signature"):
      model.add_signature(
          "other",
          subgraph,
          [("x", parts["x"]), ("c", parts["c"])],
          [("out", parts["out"])],
      )
    with self.assertRaisesRegex(ValueError, "duplicate"):
      model.add_signature(
          "other",
          subgraph,
          [("a", parts["x"]), ("a", parts["y"])],
          [("out", parts["out"])],
      )
    # Any order is accepted; names are stored in subgraph order.
    signature = model.add_signature(
        "other",
        subgraph,
        [("second", parts["y"]), ("first", parts["x"])],
        [("result", parts["out"])],
    )
    self.assertEqual(signature.input_names, ["first", "second"])
    self.assertEqual([s.key for s in model.signatures], ["serving", "other"])
    model.remove_signature("other")
    with self.assertRaises(ir.LiteRtError):
      _ = signature.key  # Stale handle.
    with self.assertRaises(KeyError):
      model.remove_signature("other")
    model.validate()

  def test_insert_transpose_before_op(self):
    model = ir.Model.load_from_file(_path("simple_model.tflite"))
    subgraph = model.subgraph(0)
    (add,) = subgraph.ops
    arg0, arg1 = subgraph.inputs
    perm = subgraph.add_tensor(
        shape=(1,),
        element_type=ir.ElementType.INT32,
        name="perm",
        weights=np.array([0], np.int32),
    )
    transposed = subgraph.add_tensor(
        shape=(2,), element_type=F32, name="transposed"
    )
    transpose = subgraph.add_op(
        ir.OpCode.TRANSPOSE, [arg1, perm], [transposed], index=add.index
    )
    add.set_input(1, transposed)
    model.validate()
    self.assertEqual([op.name for op in subgraph.ops], ["TRANSPOSE", "ADD"])
    self.assertEqual((transpose.index, add.index), (0, 1))
    self.assertEqual(arg1.users, [(transpose, 0)])
    self.assertEqual(transposed.users, [(add, 1)])
    self.assertEqual(transposed.defining_op, transpose)
    self.assertEqual(add.inputs, [arg0, transposed])
    reloaded = ir.Model.load_from_bytes(model.serialize())
    reloaded.validate()
    self.assertEqual(
        [op.name for op in reloaded.subgraph(0).ops], ["TRANSPOSE", "ADD"]
    )

  def test_insert_op_after_and_replace_all_uses(self):
    model = ir.Model.load_from_file(_path("simple_model.tflite"))
    subgraph = model.subgraph(0)
    (add,) = subgraph.ops
    (old_out,) = subgraph.outputs
    relu_out = subgraph.add_tensor(shape=(2,), element_type=F32, name="relu")
    relu = subgraph.add_op(
        ir.OpCode.RELU, [old_out], [relu_out], index=add.index + 1
    )
    subgraph.replace_all_uses(old_out, relu_out, exclude=[relu])
    model.validate()
    self.assertEqual(subgraph.outputs, [relu_out])
    self.assertEqual(old_out.users, [(relu, 0)])
    (signature,) = model.signatures
    self.assertEqual(signature.outputs, [relu_out])
    self.assertEqual(signature.output_names, ["tfl.add"])  # Name is kept.
    reloaded = ir.Model.load_from_bytes(model.serialize())
    out = reloaded.signatures[0].get_output("tfl.add")
    self.assertEqual(out.defining_op.code, ir.OpCode.RELU)

  def test_replace_all_uses_rejects_topological_violation(self):
    _, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    # `out` is produced after the ADD op that uses `x`.
    with self.assertRaisesRegex(ValueError, "not before"):
      subgraph.replace_all_uses(parts["x"], parts["out"])
    with self.assertRaisesRegex(ValueError, "itself"):
      subgraph.replace_all_uses(parts["x"], parts["x"])

  def test_add_op_errors(self):
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    fresh = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    with self.assertRaisesRegex(ValueError, "topological"):
      subgraph.add_op(ir.OpCode.RELU, [parts["out"]], [fresh], index=0)
    with self.assertRaisesRegex(ValueError, "already produced"):
      subgraph.add_op(ir.OpCode.RELU, [parts["x"]], [parts["total"]])
    with self.assertRaisesRegex(ValueError, "constant"):
      subgraph.add_op(ir.OpCode.RELU, [parts["x"]], [parts["c"]])
    with self.assertRaisesRegex(ValueError, "subgraph input"):
      subgraph.add_op(ir.OpCode.RELU, [parts["out"]], [parts["x"]])
    with self.assertRaises(IndexError):
      subgraph.add_op(ir.OpCode.RELU, [parts["x"]], [fresh], index=99)
    with self.assertRaises(IndexError):
      subgraph.add_op(ir.OpCode.RELU, [parts["x"]], [fresh], index=-1)
    with self.assertRaisesRegex(ValueError, "custom code"):
      subgraph.add_op(ir.OpCode.CUSTOM, [parts["x"]], [fresh])
    with self.assertRaisesRegex(ValueError, "custom code"):
      subgraph.add_op(ir.OpCode.RELU, [parts["x"]], [fresh], custom_code="nope")
    with self.assertRaisesRegex(ValueError, "trailing"):
      subgraph.add_op(ir.OpCode.ADD, [None, parts["x"]], [fresh])
    with self.assertRaises(ValueError):
      subgraph.add_op(9999, [parts["x"]], [fresh])
    self.assertLen(subgraph.ops, 2)
    model.validate()

  def test_add_op_trailing_optional_inputs_and_custom_op(self):
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    fresh = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    custom = subgraph.add_op(
        ir.OpCode.CUSTOM,
        [parts["out"], None, None],
        [fresh],
        custom_code="MyCustomOp",
        custom_options=b"\x01\x02",
    )
    self.assertEqual(custom.inputs, [parts["out"]])
    self.assertEqual(custom.custom_code, "MyCustomOp")
    self.assertEqual(custom.custom_options, b"\x01\x02")
    self.assertFalse(custom.is_dispatch_op)
    self.assertIsNone(custom.builtin_options)
    custom.set_custom_options(b"\x03")
    self.assertEqual(custom.custom_options, b"\x03")
    subgraph.set_outputs([fresh])
    model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    op = reloaded.subgraph(0).ops[-1]
    self.assertEqual(op.custom_code, "MyCustomOp")
    self.assertEqual(op.custom_options, b"\x03")
    self.assertEqual(reloaded.signatures[0].output_names, ["out"])

  def test_set_empty_custom_options_clears_them(self):
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    fresh = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    custom = subgraph.add_op(
        ir.OpCode.CUSTOM,
        [parts["out"]],
        [fresh],
        custom_code="MyCustomOp",
        custom_options=b"\x01\x02",
    )
    custom.set_custom_options(b"")
    self.assertEqual(custom.custom_options, b"")
    subgraph.set_outputs([fresh])
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.subgraph(0).ops[-1].custom_options, b"")

  def test_add_op_with_newest_op_codes(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph("main")
    x = subgraph.add_tensor(shape=(2,), element_type=F32, name="x")
    y = subgraph.add_tensor(shape=(2,), element_type=F32, name="y")
    op = subgraph.add_op(ir.OpCode.STABLEHLO_CBRT, [x], [y])
    self.assertEqual(op.name, "STABLEHLO_CBRT")
    subgraph.set_inputs([x])
    subgraph.set_outputs([y])
    model.validate()
    # The C++ loader does not accept op codes above STABLEHLO_COMPOSITE yet, so
    # check the flatbuffer directly.
    schema_model = _schema_model(model.serialize())
    self.assertIn(
        schema_fb.BuiltinOperator.STABLEHLO_CBRT,
        [c.builtinCode for c in schema_model.operatorCodes],
    )

  def test_stale_handles_do_not_alias_new_objects(self):
    # New objects may reuse the addresses of removed ones; handles to the
    # removed objects must still raise and not compare equal to the new ones.
    _, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    spare = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    relu_out = subgraph.add_tensor(shape=(2, 3), element_type=F32)
    relu = subgraph.add_op(ir.OpCode.RELU, [parts["out"]], [relu_out])
    subgraph.remove_op(relu)
    subgraph.remove_tensor(relu_out)
    subgraph.remove_tensor(spare)
    new_tensors = [
        subgraph.add_tensor(shape=(2, 3), element_type=F32) for _ in range(64)
    ]
    new_ops = [
        subgraph.add_op(ir.OpCode.RELU, [parts["out"]], [t])
        for t in new_tensors
    ]
    for stale, new_objects in (
        (relu, new_ops),
        (relu_out, new_tensors),
        (spare, new_tensors),
    ):
      self.assertNotIn(stale, new_objects)
      with self.assertRaisesRegex(ir.LiteRtError, "removed"):
        _ = stale.index

  def test_set_input_output_errors(self):
    model, parts = _build_add_mul_model()
    add, mul = parts["add"], parts["mul"]
    with self.assertRaises(IndexError):
      add.set_input(5, parts["x"])
    with self.assertRaises(IndexError):
      add.set_output(1, parts["x"])
    with self.assertRaisesRegex(IndexError, "negative"):
      add.set_input(-1, parts["x"])
    with self.assertRaisesRegex(IndexError, "negative"):
      add.set_output(-1, parts["x"])
    with self.assertRaisesRegex(ValueError, "not before"):
      add.set_input(0, parts["out"])
    with self.assertRaisesRegex(ValueError, "already produced"):
      add.set_output(0, parts["out"])
    other = ir.Model.load_from_file(_path("simple_model.tflite"))
    with self.assertRaisesRegex(ValueError, "different model"):
      add.set_input(0, other.subgraph(0).inputs[0])
    second = model.add_subgraph("second")
    stray = second.add_tensor(shape=(1,), element_type=F32)
    with self.assertRaisesRegex(ValueError, "different subgraph"):
      mul.set_input(0, stray)
    # set_output: the previous output loses its producer.
    new_total = parts["subgraph"].add_tensor(shape=(2, 3), element_type=F32)
    add.set_output(0, new_total)
    self.assertIsNone(parts["total"].defining_op)
    with self.assertRaisesRegex(ir.ValidationError, "neither constant"):
      model.validate()
    mul.set_input(0, new_total)
    model.validate()

  def test_remove_op_and_dce(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph()
    x = subgraph.add_tensor(shape=(4,), element_type=F32, name="x")
    r1 = subgraph.add_tensor(shape=(4,), element_type=F32, name="r1")
    r2 = subgraph.add_tensor(shape=(4,), element_type=F32, name="r2")
    r3 = subgraph.add_tensor(shape=(4,), element_type=F32, name="r3")
    unused_cst = subgraph.add_tensor(
        shape=(4,), element_type=F32, weights=np.ones(4, np.float32)
    )
    relu1 = subgraph.add_op(ir.OpCode.RELU, [x], [r1])
    relu2 = subgraph.add_op(ir.OpCode.RELU, [r1], [r2])
    relu3 = subgraph.add_op(ir.OpCode.RELU, [x], [r3])
    model.add_signature("main", subgraph, [("x", x)], [("y", r2)])
    model.validate()

    subgraph.remove_op(relu3)
    with self.assertRaisesRegex(ir.LiteRtError, "removed"):
      _ = relu3.code
    self.assertEqual(x.users, [(relu1, 0)])
    self.assertIsNone(r3.defining_op)  # The tensor is kept.
    model.validate()

    relu2.set_input(0, x)
    self.assertTrue(subgraph.dce())
    self.assertEqual(subgraph.ops, [relu2])
    self.assertCountEqual(subgraph.tensors, [x, r2])
    for stale in (relu1, r1, r3, unused_cst):
      with self.assertRaises(ir.LiteRtError):
        _ = stale.name
    self.assertFalse(subgraph.dce())
    model.validate()
    with self.assertRaisesRegex(ValueError, "different subgraph|removed"):
      model.add_subgraph().remove_op(relu2)

  def test_remove_tensor(self):
    _, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    spare = subgraph.add_tensor(shape=(1,), element_type=F32)
    subgraph.remove_tensor(spare)
    with self.assertRaises(ir.LiteRtError):
      _ = spare.shape
    with self.assertRaisesRegex(ValueError, "connected"):
      subgraph.remove_tensor(parts["total"])
    with self.assertRaisesRegex(ValueError, "connected|input"):
      subgraph.remove_tensor(parts["x"])

  def test_set_shape_and_element_type(self):
    model, parts = _build_add_mul_model()
    x = parts["x"]
    x.set_shape([-1, 3])
    self.assertEqual(x.shape, (-1, 3))
    x.set_shape([])
    self.assertEqual(x.shape, ())
    x.set_shape((2, 3))
    with self.assertRaisesRegex(ValueError, "rank"):
      x.set_shape([1] * 9)
    with self.assertRaisesRegex(ValueError, "dimension"):
      x.set_shape([-2])
    with self.assertRaisesRegex(ValueError, "constant data"):
      parts["c"].set_shape([3, 3])
    parts["c"].set_shape([3, 2])  # Same byte size.
    parts["c"].set_shape([2, 3])
    x.set_element_type(ir.ElementType.FLOAT16)
    self.assertEqual(x.element_type, ir.ElementType.FLOAT16)
    self.assertEqual(x.shape, (2, 3))
    with self.assertRaisesRegex(ValueError, "constant data"):
      parts["c"].set_element_type(ir.ElementType.FLOAT16)
    with self.assertRaises(ValueError):
      x.set_element_type(1234)
    x.set_element_type(F32)
    x.set_name("renamed")
    self.assertEqual(x.name, "renamed")
    model.validate()

  def test_set_weights(self):
    model, parts = _build_add_mul_model()
    c = parts["c"]
    old_view = c.weights
    c.set_weights(np.full((2, 3), 7.0, np.float32))
    np.testing.assert_array_equal(c.weights, np.full((2, 3), 7.0))
    # Views of the previous data stay valid (buffers are owned by the model).
    np.testing.assert_array_equal(
        old_view, np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    c.set_weights(np.full((2, 3), 1.0, np.float32).tobytes())
    np.testing.assert_array_equal(c.weights, np.ones((2, 3)))
    with self.assertRaisesRegex(TypeError, "dtype"):
      c.set_weights(np.zeros((2, 3), np.int32))
    with self.assertRaisesRegex(ValueError, "bytes"):
      c.set_weights(np.zeros((3,), np.float32))
    with self.assertRaisesRegex(ValueError, "produced by an op"):
      parts["total"].set_weights(np.zeros((2, 3), np.float32))
    with self.assertRaisesRegex(ValueError, "subgraph input"):
      parts["x"].set_weights(np.zeros((2, 3), np.float32))
    # Non-contiguous arrays are made contiguous.
    c.set_weights(np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2])
    np.testing.assert_array_equal(
        c.weights, np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
    )
    # Sub-byte types take packed raw bytes.
    int4 = parts["subgraph"].add_tensor(
        shape=(3,),
        element_type=ir.ElementType.INT4,
        weights=np.array([0x21, 0x03], np.uint8),
    )
    np.testing.assert_array_equal(int4.weights, [0x21, 0x03])
    with self.assertRaisesRegex(ValueError, "bytes"):
      int4.set_weights(b"\x00")
    reloaded = ir.Model.load_from_bytes(model.serialize())
    np.testing.assert_array_equal(
        reloaded.subgraph(0).ops[1].inputs[1].weights,
        np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2],
    )

  def test_weights_of_overflowing_shape(self):
    # (2^29 - 2^15 + 1) * (2^29 + 2^15 + 1) = 2^58 + 1, so a float32 tensor of
    # this shape needs 2^64 + 64 bits: 8 bytes if the size wraps around.
    huge = (2, 2**29 - 2**15 + 1, 2**29 + 2**15 + 1)
    _, parts = _build_add_mul_model()
    tensor = parts["subgraph"].add_tensor(shape=huge, element_type=F32)
    with self.assertRaisesRegex(ValueError, "too large"):
      tensor.set_weights(np.zeros((2,), np.float32))
    with self.assertRaisesRegex(ValueError, "too large"):
      parts["c"].set_shape((3,) + huge)  # 24 bytes if the size wraps around.

  def test_weight_views_survive_buffer_registrations(self):
    # Weight views stay valid while `BufferManager` grows (it copies and frees
    # the buffers it owns when it does).
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    constants = [
        subgraph.add_tensor(
            shape=(64,),
            element_type=F32,
            weights=np.full((64,), float(i), np.float32),
        )
        for i in range(8)
    ]
    views = [t.weights for t in constants]
    addresses = [v.__array_interface__["data"][0] for v in views]
    for i in range(300):
      model.register_buffer(bytes([i % 256]) * 33)
      model.push_metadata(f"key_{i}", b"metadata")
    parts["c"].set_weights(np.ones((2, 3), np.float32))
    for i, (tensor, view) in enumerate(zip(constants, views)):
      np.testing.assert_array_equal(view, np.full((64,), float(i)))
      self.assertEqual(
          tensor.weights.__array_interface__["data"][0], addresses[i]
      )
    data = model.serialize()
    del model, parts, constants
    gc.collect()
    for i, view in enumerate(views):
      np.testing.assert_array_equal(view, np.full((64,), float(i)))
    reloaded = ir.Model.load_from_bytes(data)
    self.assertLen(reloaded.metadata_keys(), 300)

  def test_set_quantization(self):
    model = ir.Model.create()
    subgraph = model.add_subgraph()
    x = subgraph.add_tensor(
        shape=(2, 4),
        element_type=ir.ElementType.INT8,
        name="x",
        quantization=ir.PerTensorQuantization(scale=0.5, zero_point=-3),
    )
    weights = subgraph.add_tensor(
        shape=(3, 4),
        element_type=ir.ElementType.INT8,
        name="w",
        weights=np.arange(12, dtype=np.int8).reshape(3, 4),
        quantization=ir.PerChannelQuantization(
            scales=[0.1, 0.2, 0.3], zero_points=[0, 0, 0], quantized_dimension=0
        ),
    )
    block_scales = subgraph.add_tensor(
        shape=(3, 2),
        element_type=F32,
        name="block_scales",
        weights=np.ones((3, 2), np.float32),
    )
    block_weights = subgraph.add_tensor(
        shape=(3, 4),
        element_type=ir.ElementType.INT8,
        name="block_weights",
        weights=np.zeros((3, 4), np.int8),
    )
    block_weights.set_quantization(
        ir.BlockWiseQuantization(
            scales=block_scales, zero_points=None, block_size=2
        )
    )
    out = subgraph.add_tensor(shape=(2, 3), element_type=ir.ElementType.INT8)
    out2 = subgraph.add_tensor(shape=(2, 3), element_type=ir.ElementType.INT8)
    subgraph.add_op(ir.OpCode.FULLY_CONNECTED, [x, weights], [out])
    subgraph.add_op(ir.OpCode.FULLY_CONNECTED, [x, block_weights], [out2])
    model.add_signature("main", subgraph, [("x", x)], [("y", out), ("z", out2)])
    model.validate()

    self.assertEqual(x.quantization, ir.PerTensorQuantization(0.5, -3))
    per_channel = weights.quantization
    self.assertEqual(
        per_channel,
        ir.PerChannelQuantization([0.1, 0.2, 0.3], [0, 0, 0], 0),
    )
    self.assertEqual(hash(per_channel), hash(weights.quantization))
    self.assertFalse(per_channel.scales.flags.writeable)
    block = block_weights.quantization
    self.assertEqual(block.scales, block_scales)
    self.assertIsNone(block.zero_points)
    self.assertEqual(block.block_size, 2)
    # Block-wise scales are kept by DCE while referenced.
    self.assertFalse(subgraph.dce())

    with self.assertRaisesRegex(ValueError, "channels"):
      weights.set_quantization(ir.PerChannelQuantization([1.0, 2.0], [0, 0], 0))
    with self.assertRaisesRegex(ValueError, "out of range"):
      weights.set_quantization(ir.PerChannelQuantization([1.0], [0], 5))
    with self.assertRaisesRegex(ValueError, "as many"):
      weights.set_quantization(ir.PerChannelQuantization([1.0], [0, 1], 0))
    with self.assertRaisesRegex(ValueError, "positive"):
      block_weights.set_quantization(
          ir.BlockWiseQuantization(block_scales, None, 0)
      )
    with self.assertRaisesRegex(ValueError, "its own"):
      block_weights.set_quantization(
          ir.BlockWiseQuantization(block_weights, None, 2)
      )
    with self.assertRaisesRegex(ValueError, "its own"):
      block_weights.set_quantization(
          ir.BlockWiseQuantization(block_scales, block_weights, 2)
      )
    with self.assertRaisesRegex(ValueError, "non-negative"):
      weights.set_quantization(ir.PerChannelQuantization([1.0], [0], -1))
    with self.assertRaisesRegex(ValueError, "block-wise"):
      subgraph.remove_tensor(block_scales)
    with self.assertRaises(TypeError):
      x.set_quantization(cast(Any, "int8"))
    # Failed edits leave the quantization unchanged.
    self.assertEqual(block_weights.quantization.scales, block_scales)
    self.assertEqual(weights.quantization, per_channel)

    reloaded = ir.Model.load_from_bytes(model.serialize())
    sub = reloaded.subgraph(0)
    by_name = {t.name: t for t in sub.tensors}
    self.assertEqual(
        by_name["x"].quantization, ir.PerTensorQuantization(0.5, -3)
    )
    self.assertEqual(
        by_name["w"].quantization,
        ir.PerChannelQuantization([0.1, 0.2, 0.3], [0, 0, 0], 0),
    )
    reloaded_block = by_name["block_weights"].quantization
    self.assertIsInstance(reloaded_block, ir.BlockWiseQuantization)
    self.assertEqual(reloaded_block.scales, by_name["block_scales"])
    self.assertEqual(reloaded_block.block_size, 2)

    x2 = sub.inputs[0]
    x2.set_quantization(None)
    self.assertIsNone(x2.quantization)

  def test_subgraph_io_keeps_signatures_in_sync(self):
    model, parts = _build_add_mul_model()
    subgraph = parts["subgraph"]
    subgraph.set_inputs([parts["y"], parts["x"]])
    (signature,) = model.signatures
    self.assertEqual(signature.input_names, ["y", "x"])
    self.assertEqual(signature.inputs, [parts["y"], parts["x"]])
    extra = subgraph.add_tensor(shape=(1,), element_type=F32, name="")
    subgraph.set_outputs([parts["out"], parts["total"]])
    self.assertEqual(signature.output_names, ["out", "sum"])
    with self.assertRaisesRegex(ValueError, "produced by an op"):
      subgraph.set_inputs([parts["total"]])
    with self.assertRaisesRegex(ValueError, "more than once"):
      subgraph.set_inputs([parts["x"], parts["x"]])
    with self.assertRaisesRegex(ValueError, "constant"):
      subgraph.set_inputs([parts["c"]])
    subgraph.set_inputs([parts["y"], parts["x"], extra])
    self.assertEqual(signature.input_names, ["y", "x", "input_2"])
    model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual(reloaded.signatures[0].input_names, ["y", "x", "input_2"])

  def test_placeholder_signature_with_repeated_output_round_trips(self):
    model = ir.Model.load_from_bytes(_repeated_output_model_bytes())
    (signature,) = model.signatures
    self.assertTrue(signature.is_placeholder)
    self.assertEqual(signature.output_names, ["y", "y"])
    model.validate()  # Repeated signature names are accepted.
    reloaded = ir.Model.load_from_bytes(model.serialize())
    reloaded.validate()
    (signature,) = reloaded.signatures
    self.assertEqual(signature.output_names, ["y", "y"])
    outputs = reloaded.subgraph(0).outputs
    self.assertLen(outputs, 2)
    self.assertEqual(outputs[0], outputs[1])

  def test_serialize_accepts_what_the_loader_accepts(self):
    # This model uses a tensor that is neither constant, an input nor produced
    # by an op: `validate` reports it, but it round trips like in C++.
    model = ir.Model.load_from_file(_path("blockwise_quantized.tflite"))
    with self.assertRaisesRegex(ir.ValidationError, "neither constant"):
      model.validate()
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual([op.name for op in reloaded.subgraph(0).ops], ["ADD"])
    self.assertTrue(
        any(
            isinstance(t.quantization, ir.BlockWiseQuantization)
            for t in reloaded.subgraph(0).tensors
        )
    )

  def test_validate_rejects_undefined_tensor_serialize_accepts(self):
    model, parts = _build_add_mul_model()
    parts["subgraph"].remove_op(parts["add"])
    with self.assertRaisesRegex(ir.ValidationError, "neither constant") as e:
      model.validate()
    self.assertEqual(
        e.exception.status, "kLiteRtStatusErrorInvalidGraphInvariant"
    )
    # Serialization only needs structural consistency (like C++), so this
    # still serializes.
    reloaded = ir.Model.load_from_bytes(model.serialize())
    self.assertEqual([op.name for op in reloaded.subgraph(0).ops], ["MUL"])

  def test_use_after_serialize(self):
    model, parts = _build_add_mul_model()
    weights = parts["c"].weights
    data = model.serialize()
    self.assertNotEmpty(data)
    for access in (
        lambda: model.subgraphs,
        model.serialize,
        lambda: parts["x"].shape,
        lambda: parts["add"].inputs,
        lambda: parts["subgraph"].ops,
        lambda: model.signatures,
    ):
      with self.assertRaisesRegex(ir.LiteRtError, "model has been serialized"):
        access()
    self.assertIn("serialized", repr(model))
    # Previously obtained weight views stay readable.
    np.testing.assert_array_equal(
        weights, np.arange(6, dtype=np.float32).reshape(2, 3)
    )

  def test_bad_indices_and_arguments(self):
    model, parts = _build_add_mul_model()
    with self.assertRaises(IndexError):
      model.subgraph(5)
    with self.assertRaises(IndexError):
      model.subgraph(-1)
    with self.assertRaises(ValueError):
      parts["subgraph"].add_tensor(shape=(2,), element_type=999)
    with self.assertRaisesRegex(ValueError, "power of two"):
      model.serialize(bytecode_alignment=3)
    with self.assertRaisesRegex(ValueError, "power of two"):
      model.serialize(bytecode_alignment=0)
    with self.assertRaises(TypeError):
      ir.Model(cast(Any, "not a handle"))
    self.assertFalse(model.is_serialized)


def _add_relu_subgraph(
    model: ir.Model, name: str, *, with_signature: bool = False
) -> ir.Subgraph:
  subgraph = model.add_subgraph(name)
  x = subgraph.add_tensor(shape=(2,), element_type=F32, name=f"{name}_x")
  y = subgraph.add_tensor(shape=(2,), element_type=F32, name=f"{name}_y")
  subgraph.add_op(ir.OpCode.RELU, [x], [y])
  subgraph.set_inputs([x])
  subgraph.set_outputs([y])
  if with_signature:
    model.add_signature(name, subgraph, [("x", x)], [("y", y)])
  return subgraph


def _add_composite_subgraph(
    model: ir.Model,
    name: str,
    decomp_index: int,
    *,
    with_signature: bool = True,
) -> tuple[ir.Subgraph, ir.Op]:
  subgraph = model.add_subgraph(name)
  x = subgraph.add_tensor(shape=(2,), element_type=F32, name=f"{name}_x")
  y = subgraph.add_tensor(shape=(2,), element_type=F32, name=f"{name}_y")
  options = schema_fb.StableHLOCompositeOptionsT()
  options.name = "odml.rms_norm"
  options.decompositionSubgraphIndex = decomp_index
  options.version = 1
  op = subgraph.add_op(
      ir.OpCode.STABLEHLO_COMPOSITE, [x], [y], builtin_options=options
  )
  subgraph.set_inputs([x])
  subgraph.set_outputs([y])
  if with_signature:
    model.add_signature(name, subgraph, [("x", x)], [("y", y)])
  return subgraph, op


class PruneUnreachableSubgraphsTest(googletest.TestCase):

  def test_prunes_unreferenced_decomposition_and_remaps_index(self):
    model = ir.Model.create()
    _, composite = _add_composite_subgraph(model, "main", decomp_index=2)
    unused_decomp = _add_relu_subgraph(model, "unused_decomp")
    kept_decomp = _add_relu_subgraph(model, "kept_decomp")
    unused_op = unused_decomp.ops[0]
    unused_tensor = unused_decomp.inputs[0]
    model.validate()

    self.assertEqual(model.prune_unreachable_subgraphs(), 1)
    self.assertLen(model.subgraphs, 2)
    self.assertEqual([s.name for s in model.subgraphs], ["main", "kept_decomp"])
    self.assertEqual(kept_decomp.index, 1)
    self.assertEqual(composite.builtin_options.decompositionSubgraphIndex, 1)
    for stale in (unused_decomp, unused_op, unused_tensor):
      with self.assertRaisesRegex(ir.LiteRtError, "removed"):
        _ = stale.index
    self.assertEqual(model.prune_unreachable_subgraphs(), 0)
    model.validate()

    reloaded = ir.Model.load_from_bytes(model.serialize())
    reloaded.validate()
    self.assertLen(reloaded.subgraphs, 2)
    self.assertEqual(
        reloaded.subgraph(0).ops[0].builtin_options.decompositionSubgraphIndex,
        1,
    )

  def test_outline_to_dispatch_op_then_prune_removes_orphaned_decomposition(
      self,
  ):
    model = ir.Model.load_from_file(_path("simple_shlo_composite.tflite"))
    self.assertLen(model.subgraphs, 2)
    decomp = model.subgraph(1)
    dispatch = model.subgraph(0).outline_to_dispatch_op()
    self.assertEqual(model.prune_unreachable_subgraphs(), 1)
    self.assertLen(model.subgraphs, 1)
    self.assertEqual(model.subgraph(0).ops, [dispatch])
    with self.assertRaisesRegex(ir.LiteRtError, "removed"):
      _ = decomp.name
    model.validate()

  def test_two_signatures_partial_outline_keeps_and_remaps_remaining(self):
    model = ir.Model.create()
    sig_a, _ = _add_composite_subgraph(model, "sig_a", decomp_index=2)
    _, comp_b = _add_composite_subgraph(model, "sig_b", decomp_index=3)
    decomp_a = _add_relu_subgraph(model, "decomp_a")
    decomp_b = _add_relu_subgraph(model, "decomp_b")
    model.validate()

    sig_a.outline_to_dispatch_op()
    self.assertEqual(model.prune_unreachable_subgraphs(), 1)
    self.assertLen(model.subgraphs, 3)
    self.assertEqual(
        [s.name for s in model.subgraphs], ["sig_a", "sig_b", "decomp_b"]
    )
    self.assertEqual(decomp_b.index, 2)
    self.assertEqual(comp_b.builtin_options.decompositionSubgraphIndex, 2)
    with self.assertRaisesRegex(ir.LiteRtError, "removed"):
      _ = decomp_a.name
    model.validate()

  def test_control_flow_ops_tolerated_without_mutation(self):
    for code, options_cls in (
        (ir.OpCode.WHILE, schema_fb.WhileOptionsT),
        (ir.OpCode.IF, schema_fb.IfOptionsT),
    ):
      with self.subTest(code.name):
        model = ir.Model.create()
        main = model.add_subgraph("main")
        x = main.add_tensor(shape=(2,), element_type=F32, name="x")
        y = main.add_tensor(shape=(2,), element_type=F32, name="y")
        main.add_op(code, [x], [y], builtin_options=options_cls())
        model.add_signature("serving", main, [("x", x)], [("y", y)])
        _add_relu_subgraph(model, "branch_1")
        _add_relu_subgraph(model, "branch_2")
        _add_relu_subgraph(model, "unused")
        model.validate()

        self.assertEqual(model.prune_unreachable_subgraphs(), 0)
        self.assertLen(model.subgraphs, 4)
        self.assertEqual([s.key for s in model.signatures], ["serving"])
        model.validate()

  def test_signatureless_model_keeps_main_and_reachable_decompositions(self):
    self.assertEqual(ir.Model.create().prune_unreachable_subgraphs(), 0)

    model = ir.Model.create()
    _, composite = _add_composite_subgraph(
        model, "main", decomp_index=2, with_signature=False
    )
    unreachable = _add_relu_subgraph(model, "unreachable")
    kept_decomp = _add_relu_subgraph(model, "kept_decomp")
    self.assertEmpty(model.signatures)
    model.validate()

    self.assertEqual(model.prune_unreachable_subgraphs(), 1)
    self.assertEmpty(model.signatures)
    self.assertEqual([s.name for s in model.subgraphs], ["main", "kept_decomp"])
    self.assertEqual(kept_decomp.index, 1)
    self.assertEqual(composite.builtin_options.decompositionSubgraphIndex, 1)
    with self.assertRaisesRegex(ir.LiteRtError, "removed"):
      _ = unreachable.name
    model.validate()


if __name__ == "__main__":
  googletest.main()
