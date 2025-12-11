# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for LiteRT flatbuffer_utils."""
import copy
import os
import subprocess
import sys
import flatbuffers
from absl.testing import absltest as googletest
from litert.python.tools import flatbuffer_utils
from litert.python import schema_py_generated as schema_fb  # pylint:disable=g-direct-tensorflow-import

_SKIPPED_BUFFER_INDEX = 1

_TFLITE_SCHEMA_VERSION = 3


def build_mock_flatbuffer_model():
  """Creates a flatbuffer containing an example model."""
  builder = flatbuffers.Builder(1024)

  schema_fb.BufferStart(builder)
  buffer0_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStartDataVector(builder, 12)
  builder.PrependUint8(11)
  builder.PrependUint8(10)
  builder.PrependUint8(9)
  builder.PrependUint8(8)
  builder.PrependUint8(7)
  builder.PrependUint8(6)
  builder.PrependUint8(5)
  builder.PrependUint8(4)
  builder.PrependUint8(3)
  builder.PrependUint8(2)
  builder.PrependUint8(1)
  builder.PrependUint8(0)
  buffer1_data_offset = builder.EndVector()
  schema_fb.BufferStart(builder)
  schema_fb.BufferAddData(builder, buffer1_data_offset)
  buffer1_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStart(builder)
  buffer2_offset = schema_fb.BufferEnd(builder)

  schema_fb.ModelStartBuffersVector(builder, 3)
  builder.PrependUOffsetTRelative(buffer2_offset)
  builder.PrependUOffsetTRelative(buffer1_offset)
  builder.PrependUOffsetTRelative(buffer0_offset)
  buffers_offset = builder.EndVector()

  string0_offset = builder.CreateString('input_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape0_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string0_offset)
  schema_fb.TensorAddShape(builder, shape0_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 0)
  tensor0_offset = schema_fb.TensorEnd(builder)

  schema_fb.QuantizationParametersStartMinVector(builder, 5)
  builder.PrependFloat32(0.5)
  builder.PrependFloat32(2.0)
  builder.PrependFloat32(5.0)
  builder.PrependFloat32(10.0)
  builder.PrependFloat32(20.0)
  quant1_min_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartMaxVector(builder, 5)
  builder.PrependFloat32(10.0)
  builder.PrependFloat32(20.0)
  builder.PrependFloat32(-50.0)
  builder.PrependFloat32(1.0)
  builder.PrependFloat32(2.0)
  quant1_max_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartScaleVector(builder, 5)
  builder.PrependFloat32(3.0)
  builder.PrependFloat32(4.0)
  builder.PrependFloat32(5.0)
  builder.PrependFloat32(6.0)
  builder.PrependFloat32(7.0)
  quant1_scale_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartZeroPointVector(builder, 5)
  builder.PrependInt64(1)
  builder.PrependInt64(2)
  builder.PrependInt64(3)
  builder.PrependInt64(-1)
  builder.PrependInt64(-2)
  quant1_zero_point_offset = builder.EndVector()

  schema_fb.QuantizationParametersStart(builder)
  schema_fb.QuantizationParametersAddMin(builder, quant1_min_offset)
  schema_fb.QuantizationParametersAddMax(builder, quant1_max_offset)
  schema_fb.QuantizationParametersAddScale(builder, quant1_scale_offset)
  schema_fb.QuantizationParametersAddZeroPoint(
      builder, quant1_zero_point_offset
  )
  quantization1_offset = schema_fb.QuantizationParametersEnd(builder)

  string1_offset = builder.CreateString('constant_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape1_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string1_offset)
  schema_fb.TensorAddShape(builder, shape1_offset)
  schema_fb.TensorAddType(builder, schema_fb.TensorType.UINT8)
  schema_fb.TensorAddBuffer(builder, 1)
  schema_fb.TensorAddQuantization(builder, quantization1_offset)
  tensor1_offset = schema_fb.TensorEnd(builder)

  string2_offset = builder.CreateString('output_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape2_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string2_offset)
  schema_fb.TensorAddShape(builder, shape2_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 2)
  tensor2_offset = schema_fb.TensorEnd(builder)

  schema_fb.SubGraphStartTensorsVector(builder, 3)
  builder.PrependUOffsetTRelative(tensor2_offset)
  builder.PrependUOffsetTRelative(tensor1_offset)
  builder.PrependUOffsetTRelative(tensor0_offset)
  tensors_offset = builder.EndVector()

  schema_fb.SubGraphStartInputsVector(builder, 1)
  builder.PrependInt32(0)
  inputs_offset = builder.EndVector()

  schema_fb.SubGraphStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  outputs_offset = builder.EndVector()

  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder, schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddDeprecatedBuiltinCode(
      builder, schema_fb.BuiltinOperator.ADD
  )
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code0_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(
      builder, schema_fb.BuiltinOperator.VAR_HANDLE
  )
  schema_fb.OperatorCodeAddDeprecatedBuiltinCode(
      builder, schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
  )
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code1_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.ModelStartOperatorCodesVector(builder, 2)
  builder.PrependUOffsetTRelative(code1_offset)
  builder.PrependUOffsetTRelative(code0_offset)
  codes_offset = builder.EndVector()

  schema_fb.OperatorStartInputsVector(builder, 2)
  builder.PrependInt32(0)
  builder.PrependInt32(1)
  op_inputs_offset = builder.EndVector()

  schema_fb.OperatorStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  op_outputs_offset = builder.EndVector()

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 0)
  schema_fb.OperatorAddInputs(builder, op_inputs_offset)
  schema_fb.OperatorAddOutputs(builder, op_outputs_offset)
  op0_offset = schema_fb.OperatorEnd(builder)

  shared_name = builder.CreateString('var')
  schema_fb.VarHandleOptionsStart(builder)
  schema_fb.VarHandleOptionsAddSharedName(builder, shared_name)
  var_handle_options_offset = schema_fb.VarHandleOptionsEnd(builder)

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 1)
  schema_fb.OperatorAddBuiltinOptionsType(
      builder, schema_fb.BuiltinOptions.VarHandleOptions
  )
  schema_fb.OperatorAddBuiltinOptions(builder, var_handle_options_offset)
  op1_offset = schema_fb.OperatorEnd(builder)

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddBuiltinOptionsType(
      builder, schema_fb.BuiltinOptions.VarHandleOptions
  )
  schema_fb.OperatorAddBuiltinOptions(builder, var_handle_options_offset)
  op2_offset = schema_fb.OperatorEnd(builder)

  schema_fb.SubGraphStartOperatorsVector(builder, 3)
  builder.PrependUOffsetTRelative(op2_offset)
  builder.PrependUOffsetTRelative(op1_offset)
  builder.PrependUOffsetTRelative(op0_offset)
  ops_offset = builder.EndVector()

  string3_offset = builder.CreateString('subgraph_name')
  schema_fb.SubGraphStart(builder)
  schema_fb.SubGraphAddName(builder, string3_offset)
  schema_fb.SubGraphAddTensors(builder, tensors_offset)
  schema_fb.SubGraphAddInputs(builder, inputs_offset)
  schema_fb.SubGraphAddOutputs(builder, outputs_offset)
  schema_fb.SubGraphAddOperators(builder, ops_offset)
  subgraph_offset = schema_fb.SubGraphEnd(builder)

  schema_fb.ModelStartSubgraphsVector(builder, 1)
  builder.PrependUOffsetTRelative(subgraph_offset)
  subgraphs_offset = builder.EndVector()

  signature_key = builder.CreateString('my_key')
  input_tensor_string = builder.CreateString('input_tensor')
  output_tensor_string = builder.CreateString('output_tensor')

  # Signature Inputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, input_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 1)
  input_tensor = schema_fb.TensorMapEnd(builder)

  # Signature Outputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, output_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 2)
  output_tensor = schema_fb.TensorMapEnd(builder)

  schema_fb.SignatureDefStartInputsVector(builder, 1)
  builder.PrependUOffsetTRelative(input_tensor)
  signature_inputs_offset = builder.EndVector()
  schema_fb.SignatureDefStartOutputsVector(builder, 1)
  builder.PrependUOffsetTRelative(output_tensor)
  signature_outputs_offset = builder.EndVector()

  schema_fb.SignatureDefStart(builder)
  schema_fb.SignatureDefAddSignatureKey(builder, signature_key)
  schema_fb.SignatureDefAddInputs(builder, signature_inputs_offset)
  schema_fb.SignatureDefAddOutputs(builder, signature_outputs_offset)
  signature_offset = schema_fb.SignatureDefEnd(builder)
  schema_fb.ModelStartSignatureDefsVector(builder, 1)
  builder.PrependUOffsetTRelative(signature_offset)
  signature_defs_offset = builder.EndVector()

  string4_offset = builder.CreateString('model_description')
  schema_fb.ModelStart(builder)
  schema_fb.ModelAddVersion(builder, _TFLITE_SCHEMA_VERSION)
  schema_fb.ModelAddOperatorCodes(builder, codes_offset)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddDescription(builder, string4_offset)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSignatureDefs(builder, signature_defs_offset)
  model_offset = schema_fb.ModelEnd(builder)
  builder.Finish(model_offset)
  model = builder.Output()

  return model


def build_operator_with_options() -> schema_fb.Operator:
  """Builds an operator with the given options."""
  builder = flatbuffers.Builder(1024)
  schema_fb.StableHLOCompositeOptionsStart(builder)
  schema_fb.StableHLOCompositeOptionsAddDecompositionSubgraphIndex(builder, 10)
  opts = schema_fb.StableHLOCompositeOptionsEnd(builder)
  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddBuiltinOptions2(builder, opts)
  schema_fb.OperatorAddBuiltinOptions2Type(
      builder, schema_fb.BuiltinOptions2.StableHLOCompositeOptions
  )
  op_offset = schema_fb.OperatorEnd(builder)
  builder.Finish(op_offset)
  return schema_fb.Operator.GetRootAs(builder.Output())


def load_model_from_flatbuffer(flatbuffer_model):
  """Loads a model as a python object from a flatbuffer model."""
  model = schema_fb.Model.GetRootAsModel(flatbuffer_model, 0)
  model = schema_fb.ModelT.InitFromObj(model)
  return model


def build_mock_model():
  """Creates an object containing an example model."""
  model = build_mock_flatbuffer_model()
  return load_model_from_flatbuffer(model)


class WriteReadModelTest(googletest.TestCase):

  def testWriteReadModel(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()
    # Define temporary files
    tmp_dir = self.create_tempdir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')

    # 2. INVOKE
    # Invoke the write_model and read_model functions
    flatbuffer_utils.write_model(initial_model, model_filename)
    final_model = flatbuffer_utils.read_model(model_filename)

    # 3. VALIDATE
    # Validate that the initial and final models are the same
    # Validate the description
    self.assertEqual(initial_model.description, final_model.description)
    # Validate the main subgraph's name, inputs, outputs, operators and tensors
    initial_subgraph = initial_model.subgraphs[0]
    final_subgraph = final_model.subgraphs[0]
    self.assertEqual(initial_subgraph.name, final_subgraph.name)
    for i in range(len(initial_subgraph.inputs)):
      self.assertEqual(initial_subgraph.inputs[i], final_subgraph.inputs[i])
    for i in range(len(initial_subgraph.outputs)):
      self.assertEqual(initial_subgraph.outputs[i], final_subgraph.outputs[i])
    for i in range(len(initial_subgraph.operators)):
      self.assertEqual(
          initial_subgraph.operators[i].opcodeIndex,
          final_subgraph.operators[i].opcodeIndex,
      )
    initial_tensors = initial_subgraph.tensors
    final_tensors = final_subgraph.tensors
    for i in range(len(initial_tensors)):
      self.assertEqual(initial_tensors[i].name, final_tensors[i].name)
      self.assertEqual(initial_tensors[i].type, final_tensors[i].type)
      self.assertEqual(initial_tensors[i].buffer, final_tensors[i].buffer)
      for j in range(len(initial_tensors[i].shape)):
        self.assertEqual(initial_tensors[i].shape[j], final_tensors[i].shape[j])
    # Validate the first valid buffer (index 0 is always None)
    initial_buffer = initial_model.buffers[1].data
    final_buffer = final_model.buffers[1].data
    for i in range(initial_buffer.size):
      self.assertEqual(initial_buffer.data[i], final_buffer.data[i])


class StripStringsTest(googletest.TestCase):

  def testStripStrings(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()
    final_model = copy.deepcopy(initial_model)

    # 2. INVOKE
    # Invoke the strip_strings function
    flatbuffer_utils.strip_strings(final_model)

    # 3. VALIDATE
    # Validate that the initial and final models are the same except strings
    # Validate the description
    self.assertIsNotNone(initial_model.description)
    self.assertIsNone(final_model.description)
    self.assertIsNotNone(initial_model.signatureDefs)
    self.assertIsNone(final_model.signatureDefs)

    # Validate the main subgraph's name, inputs, outputs, operators and tensors
    initial_subgraph = initial_model.subgraphs[0]
    final_subgraph = final_model.subgraphs[0]
    self.assertIsNotNone(initial_model.subgraphs[0].name)
    self.assertIsNone(final_model.subgraphs[0].name)
    for i in range(len(initial_subgraph.inputs)):
      self.assertEqual(initial_subgraph.inputs[i], final_subgraph.inputs[i])
    for i in range(len(initial_subgraph.outputs)):
      self.assertEqual(initial_subgraph.outputs[i], final_subgraph.outputs[i])
    for i in range(len(initial_subgraph.operators)):
      self.assertEqual(
          initial_subgraph.operators[i].opcodeIndex,
          final_subgraph.operators[i].opcodeIndex,
      )
    initial_tensors = initial_subgraph.tensors
    final_tensors = final_subgraph.tensors
    for i in range(len(initial_tensors)):
      self.assertIsNotNone(initial_tensors[i].name)
      self.assertIsNone(final_tensors[i].name)
      self.assertEqual(initial_tensors[i].type, final_tensors[i].type)
      self.assertEqual(initial_tensors[i].buffer, final_tensors[i].buffer)
      for j in range(len(initial_tensors[i].shape)):
        self.assertEqual(initial_tensors[i].shape[j], final_tensors[i].shape[j])
    # Validate the first valid buffer (index 0 is always None)
    initial_buffer = initial_model.buffers[1].data
    final_buffer = final_model.buffers[1].data
    for i in range(initial_buffer.size):
      self.assertEqual(initial_buffer.data[i], final_buffer.data[i])


class RandomizeWeightsTest(googletest.TestCase):

  def testRandomizeWeights(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()
    final_model = copy.deepcopy(initial_model)

    # 2. INVOKE
    # Invoke the randomize_weights function
    flatbuffer_utils.randomize_weights(final_model)

    # 3. VALIDATE
    # Validate that the initial and final models are the same, except that
    # the weights in the model buffer have been modified (i.e, randomized)
    # Validate the description
    self.assertEqual(initial_model.description, final_model.description)
    # Validate the main subgraph's name, inputs, outputs, operators and tensors
    initial_subgraph = initial_model.subgraphs[0]
    final_subgraph = final_model.subgraphs[0]
    self.assertEqual(initial_subgraph.name, final_subgraph.name)
    for i in range(len(initial_subgraph.inputs)):
      self.assertEqual(initial_subgraph.inputs[i], final_subgraph.inputs[i])
    for i in range(len(initial_subgraph.outputs)):
      self.assertEqual(initial_subgraph.outputs[i], final_subgraph.outputs[i])
    for i in range(len(initial_subgraph.operators)):
      self.assertEqual(
          initial_subgraph.operators[i].opcodeIndex,
          final_subgraph.operators[i].opcodeIndex,
      )
    initial_tensors = initial_subgraph.tensors
    final_tensors = final_subgraph.tensors
    for i in range(len(initial_tensors)):
      self.assertEqual(initial_tensors[i].name, final_tensors[i].name)
      self.assertEqual(initial_tensors[i].type, final_tensors[i].type)
      self.assertEqual(initial_tensors[i].buffer, final_tensors[i].buffer)
      for j in range(len(initial_tensors[i].shape)):
        self.assertEqual(initial_tensors[i].shape[j], final_tensors[i].shape[j])
    # Validate the first valid buffer (index 0 is always None)
    initial_buffer = initial_model.buffers[1].data
    final_buffer = final_model.buffers[1].data
    for j in range(initial_buffer.size):
      self.assertNotEqual(initial_buffer.data[j], final_buffer.data[j])

  def testRandomizeSomeWeights(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()
    final_model = copy.deepcopy(initial_model)

    # 2. INVOKE
    # Invoke the randomize_weights function, but skip the first buffer
    flatbuffer_utils.randomize_weights(
        final_model, buffers_to_skip=[_SKIPPED_BUFFER_INDEX]
    )

    # 3. VALIDATE
    # Validate that the initial and final models are the same, except that
    # the weights in the model buffer have been modified (i.e, randomized)
    # Validate the description
    self.assertEqual(initial_model.description, final_model.description)
    # Validate the main subgraph's name, inputs, outputs, operators and tensors
    initial_subgraph = initial_model.subgraphs[0]
    final_subgraph = final_model.subgraphs[0]
    self.assertEqual(initial_subgraph.name, final_subgraph.name)
    for i, _ in enumerate(initial_subgraph.inputs):
      self.assertEqual(initial_subgraph.inputs[i], final_subgraph.inputs[i])
    for i, _ in enumerate(initial_subgraph.outputs):
      self.assertEqual(initial_subgraph.outputs[i], final_subgraph.outputs[i])
    for i, _ in enumerate(initial_subgraph.operators):
      self.assertEqual(
          initial_subgraph.operators[i].opcodeIndex,
          final_subgraph.operators[i].opcodeIndex,
      )
    initial_tensors = initial_subgraph.tensors
    final_tensors = final_subgraph.tensors
    for i, _ in enumerate(initial_tensors):
      self.assertEqual(initial_tensors[i].name, final_tensors[i].name)
      self.assertEqual(initial_tensors[i].type, final_tensors[i].type)
      self.assertEqual(initial_tensors[i].buffer, final_tensors[i].buffer)
      for j in range(len(initial_tensors[i].shape)):
        self.assertEqual(initial_tensors[i].shape[j], final_tensors[i].shape[j])
    # Validate that the skipped buffer is unchanged.
    initial_buffer = initial_model.buffers[_SKIPPED_BUFFER_INDEX].data
    final_buffer = final_model.buffers[_SKIPPED_BUFFER_INDEX].data
    for j in range(initial_buffer.size):
      self.assertEqual(initial_buffer.data[j], final_buffer.data[j])


class XxdOutputToBytesTest(googletest.TestCase):

  def testXxdOutputToBytes(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()
    initial_bytes = flatbuffer_utils.convert_object_to_bytearray(initial_model)

    # Define temporary files
    tmp_dir = self.create_tempdir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')

    # 2. Write model to temporary file (will be used as input for xxd)
    flatbuffer_utils.write_model(initial_model, model_filename)

    # 3. DUMP WITH xxd
    input_cc_file = os.path.join(tmp_dir, 'model.cc')

    command = 'xxd -i {} > {}'.format(model_filename, input_cc_file)
    subprocess.call(command, shell=True)

    # 4. VALIDATE
    final_bytes = flatbuffer_utils.xxd_output_to_bytes(input_cc_file)
    if sys.byteorder == 'big':
      final_bytes = flatbuffer_utils.byte_swap_tflite_buffer(
          final_bytes, 'little', 'big'
      )

    # Validate that the initial and final bytearray are the same
    self.assertEqual(initial_bytes, final_bytes)


class CountResourceVariablesTest(googletest.TestCase):

  def testCountResourceVariables(self):
    # 1. SETUP
    # Define the initial model
    initial_model = build_mock_model()

    # 2. Confirm that resource variables for mock model is 1
    # The mock model is created with two VAR HANDLE ops, but with the same
    # shared name.
    self.assertEqual(
        flatbuffer_utils.count_resource_variables(initial_model), 1
    )


class GetOptionsTest(googletest.TestCase):

  op: schema_fb.Operator
  op_t: schema_fb.OperatorT

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.op = build_operator_with_options()
    cls.op_t = schema_fb.OperatorT.InitFromObj(cls.op)

  def test_get_options(self):
    ty = schema_fb.StableHLOCompositeOptionsT
    opts = flatbuffer_utils.get_options_as(self.op, ty)
    self.assertIsNotNone(opts)
    self.assertIsInstance(opts, ty)
    self.assertEqual(opts.decompositionSubgraphIndex, 10)

  def test_get_options_obj(self):
    ty = schema_fb.StableHLOCompositeOptionsT
    opts = flatbuffer_utils.get_options_as(self.op_t, ty)
    self.assertIsNotNone(opts)
    self.assertIsInstance(opts, ty)
    self.assertEqual(opts.decompositionSubgraphIndex, 10)

  def test_get_options_not_schema_type_raises(self):
    with self.assertRaises(ValueError):
      flatbuffer_utils.get_options_as(self.op, int)

  def test_get_options_not_object_type_raises(self):
    with self.assertRaises(ValueError):
      flatbuffer_utils.get_options_as(
          self.op, schema_fb.StableHLOCompositeOptions
      )

  def test_get_options_op_type_does_not_match(self):
    ty = schema_fb.Conv2DOptionsT
    opts = flatbuffer_utils.get_options_as(self.op, ty)
    self.assertIsNone(opts)


if __name__ == '__main__':
  googletest.main()
