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

import pathlib

import flatbuffers

from absl.testing import absltest as googletest
from litert.tools import externalize_tflite_flatbuffer
from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import


class ExternalizeTfliteFlatbufferTest(googletest.TestCase):

  def test_preserves_existing_weight_blob(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model = root / "input.tflite"
    existing_weights = root / "existing_weights"
    existing_weights.write_bytes(b"existing external weights")

    model = schema.ModelT()
    model.version = 3
    model.operatorCodes = []
    model.subgraphs = []
    model.buffers = []
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    input_model.write_bytes(bytes(builder.Output()))

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=256,
        existing_weights=existing_weights,
    )

    self.assertTrue((output_dir / "model.tflite").is_file())
    self.assertEqual(
        (output_dir / "tflite_weights").read_bytes(),
        existing_weights.read_bytes(),
    )

  def test_externalizes_appended_tflite_buffers_without_losing_bias(self):
    root = pathlib.Path(self.create_tempdir().full_path)
    input_model = root / "input.tflite"
    weight_data = bytes(range(64))
    bias_data = b"bias data bytes!"

    def make_tensor(shape, buffer_index):
      tensor = schema.TensorT()
      tensor.shape = shape
      tensor.type = externalize_tflite_flatbuffer._enum_value(
          schema.TensorType, "FLOAT32"
      )
      tensor.buffer = buffer_index
      return tensor

    model = schema.ModelT()
    model.version = 3
    operator_code = schema.OperatorCodeT()
    operator_code.builtinCode = externalize_tflite_flatbuffer._enum_value(
        schema.BuiltinOperator, "FULLY_CONNECTED"
    )
    operator_code.deprecatedBuiltinCode = operator_code.builtinCode
    model.operatorCodes = [operator_code]

    operator = schema.OperatorT()
    operator.opcodeIndex = 0
    operator.inputs = [0, 1, 2]
    operator.outputs = [3]
    subgraph = schema.SubGraphT()
    subgraph.tensors = [
        make_tensor([1, 4], 0),
        make_tensor([4, 4], 1),
        make_tensor([4], 2),
        make_tensor([1, 4], 0),
    ]
    subgraph.inputs = [0]
    subgraph.outputs = [3]
    subgraph.operators = [operator]
    model.subgraphs = [subgraph]
    model.buffers = [schema.BufferT(), schema.BufferT(), schema.BufferT()]
    model.buffers[1].offset = 1
    model.buffers[1].size = len(weight_data)
    model.buffers[2].offset = 1
    model.buffers[2].size = len(bias_data)

    def pack_model():
      builder = flatbuffers.Builder(1024)
      builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
      return bytes(builder.Output())

    placeholder = pack_model()
    weight_offset = (len(placeholder) + 15) // 16 * 16
    bias_offset = (weight_offset + len(weight_data) + 15) // 16 * 16
    model.buffers[1].offset = weight_offset
    model.buffers[2].offset = bias_offset
    packed_model = pack_model()
    self.assertEqual(len(packed_model), len(placeholder))
    input_bytes = bytearray(packed_model)
    input_bytes.extend(b"\0" * (weight_offset - len(input_bytes)))
    input_bytes.extend(weight_data)
    input_bytes.extend(b"\0" * (bias_offset - len(input_bytes)))
    input_bytes.extend(bias_data)
    input_model.write_bytes(input_bytes)

    output_dir = root / "output"
    externalize_tflite_flatbuffer.externalize(
        input_model=input_model,
        output_dir=output_dir,
        group_name="tflite_weights",
        num_elements_threshold=4,
    )

    self.assertEqual((output_dir / "tflite_weights").read_bytes(), weight_data)
    processed_model = schema.ModelT.InitFromPackedBuf(
        (output_dir / "model.tflite").read_bytes(), 0
    )
    weight_tensor = processed_model.subgraphs[0].tensors[1]
    self.assertNotEqual(weight_tensor.externalBuffer, 0)
    self.assertEqual(weight_tensor.buffer, 0)
    processed_bias = processed_model.buffers[2]
    self.assertEqual(bytes(processed_bias.data), bias_data)
    self.assertEqual(processed_bias.offset, 0)
    self.assertEqual(processed_bias.size, 0)


if __name__ == "__main__":
  googletest.main()
