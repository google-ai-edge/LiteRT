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
"""Utility functions for FlatBuffers.

All functions that are commonly used to work with FlatBuffers.
"""

from collections.abc import Mapping
import copy
import logging
import mmap
import os
import pathlib
import random
import re
import struct
import sys
from typing import Any, Literal, Protocol, Type, TypeVar, overload

import flatbuffers
import numpy as np

import os # import gfile
from litert.python import schema_py_generated as schema_fb  # pylint:disable=g-direct-tensorflow-import

# Types imported from `schema_py_generated`.
ActivationFunctionType = schema_fb.ActivationFunctionType
BlockwiseQuantizationT = schema_fb.BlockwiseQuantizationT
Buffer = schema_fb.Buffer
BufferT = schema_fb.BufferT
BuiltinOperator = schema_fb.BuiltinOperator
BuiltinOptions = schema_fb.BuiltinOptions
BuiltinOptions2 = schema_fb.BuiltinOptions2
FullyConnectedOptionsT = schema_fb.FullyConnectedOptionsT
Model = schema_fb.Model
ModelT = schema_fb.ModelT
Operator = schema_fb.Operator
OperatorCode = schema_fb.OperatorCode
OperatorCodeT = schema_fb.OperatorCodeT
OperatorT = schema_fb.OperatorT
QuantizationDetails = schema_fb.QuantizationDetails
QuantizationParametersT = schema_fb.QuantizationParametersT
StableHLOCompositeOptions = schema_fb.StableHLOCompositeOptions
StableHLOCompositeOptionsT = schema_fb.StableHLOCompositeOptionsT
SubGraph = schema_fb.SubGraph
SubGraphT = schema_fb.SubGraphT
Tensor = schema_fb.Tensor
TensorT = schema_fb.TensorT
TensorType = schema_fb.TensorType

# Local convenience types.
Path = str | pathlib.Path
BufferType = bytes | bytearray | memoryview | mmap.mmap
Endiness = Literal['little', 'big']


class Packed(Protocol):

  @classmethod
  def GetRootAs(cls, buf: BufferType, offset: int = 0):  # pylint: disable=invalid-name
    ...


class Packable(Protocol):

  def Pack(self, builder: flatbuffers.Builder):  # pylint: disable=invalid-name
    ...

  @classmethod
  def InitFromObj(cls, operatorCode: Packed):  # pylint: disable=invalid-name
    ...


_TFLITE_FILE_IDENTIFIER = b'TFL3'

_TENSOR_TYPE_TO_NAME = {v: k for k, v in TensorType.__dict__.items()}


def update_packed_buffer(
    packed_buffer: Buffer, offset: int | None = None, size: int | None = None
):
  """Sets the `offset` and `size` values in a packed `Buffer` object.

  This is based on the implementation of `Buffer.Offset` and `Buffer.Size` in
  `schema_py_generated.py`.

  The `Bufffer` is declared as follows in the TFLite schema:
  ```
  table Buffer {
    data:[ubyte] (force_align: 16);
    offset: ulong;
    size: ulong;
  }
  ```
  i.e. the `offset` and `size` fields are of type `uint64` and reside at offsets
  6 and 8 of the `Buffer` vtable, respectively.

  Note that for this to work, the original `Buffer` object has to have had
  non-default values for `offset` and `size` set.

  Args:
    packed_buffer: The packed `Buffer` object to modify.
    offset: Integer offset value.
    size: Integer size value.

  Raises:
    `ValueError` if the original `Buffer` object has no spece reserved for
    either `offset` or `size`.
  """
  table = packed_buffer._tab  # pylint: disable=protected-access
  packer_type = flatbuffers.number_types.Uint64Flags.packer_type
  if offset is not None:
    vtable_offset = flatbuffers.number_types.UOffsetTFlags.py_type(
        table.Offset(6)
    )
    if vtable_offset == 0:
      raise ValueError('Failed to set `offset`, no buffer reserved for it.')
    flatbuffers.encode.Write(
        packer_type,
        table.Bytes,
        vtable_offset + table.Pos,
        offset,
    )
  if size is not None:
    vtable_offset = flatbuffers.number_types.UOffsetTFlags.py_type(
        table.Offset(8)
    )
    if vtable_offset == 0:
      raise ValueError('Failed to set `size`, no buffer reserved for it.')
    flatbuffers.encode.Write(
        packer_type,
        table.Bytes,
        vtable_offset + table.Pos,
        size,
    )


def get_builtin_code_from_operator_code(
    opcode: OperatorCode | OperatorCodeT,
) -> int:
  """Return the builtin code of the given operator code.

  The following method is introduced to resolve op builtin code shortage
  problem. The new builtin operator will be assigned to the extended builtin
  code field in the flatbuffer schema. Those methods helps to hide builtin code
  details.

  Args:
    opcode: Operator code.

  Returns:
    The builtin code of the given operator code.
  """
  # Access BuiltinCode() method first if available.
  if isinstance(opcode, OperatorCode):
    return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())
  return max(opcode.builtinCode, opcode.deprecatedBuiltinCode)


def convert_bytearray_to_object(model_bytearray: BufferType) -> ModelT:
  """Converts a packed TFLite model from a buffer to a `ModelT` object."""
  model_object = Model.GetRootAs(model_bytearray, 0)
  return ModelT.InitFromObj(model_object)


def read_model(input_tflite_file: Path) -> ModelT:
  """Reads a tflite model as a python object.

  Args:
    input_tflite_file: Full path name to the input tflite file.

  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """

  model_bytearray = None

  # Try to mmap the file first if it is local.
  if (fd := os.open(input_tflite_file, os.O_RDONLY)) >= 0:
    try:
      model_bytearray = mmap.mmap(
          fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ
      )
    except IOError as e:
      logging.info(
          'Mapping model file "%s" failed with exception: %s.',
          input_tflite_file,
          e,
      )
    os.close(fd)

  if not model_bytearray:
    if not os.path.exists(input_tflite_file):
      raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
    with open(input_tflite_file, 'rb') as input_file_handle:
      model_bytearray = bytearray(input_file_handle.read())

  return read_model_from_bytearray(model_bytearray)


T = TypeVar('T')


@overload
def _ndarrays_to_lists(value: np.ndarray) -> np.ndarray | list[int]:
  ...


@overload
def _ndarrays_to_lists(value: list) -> list:
  ...


@overload
def _ndarrays_to_lists(value: T) -> T:
  ...


def _ndarrays_to_lists(value: Any) -> Any:
  """Recursively convert `np.ndarray`s of `np.int32` to `list`.

  If the input is a `list`, this function recurses over its elements. If the
  input has a `__dict__` attribute, this function recurses over its non-`None`
  entries.

  Args:
    value: The object in which to replace `np.ndarray`s with `list`s.

  Returns:
    The modified value.
  """
  if isinstance(value, np.ndarray):
    return value.tolist() if value.dtype == np.int32 else value
  if isinstance(value, list):
    return [_ndarrays_to_lists(v) for v in value]
  if hasattr(value, '__dict__'):
    for k, v in value.__dict__.items():
      if v is not None:
        value.__dict__[k] = _ndarrays_to_lists(v)
  return value


def read_model_from_bytearray(model_bytearray: BufferType) -> ModelT:
  """Reads a tflite model as a python object.

  This function also does the following:
   * Resolves `Buffer`s specified by their offset/size,
   * Resolves `Operator.CustomOptions` specified by their offset/size,
   * Converts non-`Buffer` `np.ndarray` objects (things like `op.inputs`) to
     `list` since if the underlying buffer is not mutable, neither will the
     `np.ndarray`s.

  Args:
    model_bytearray: TFLite model in bytearray format.

  Returns:
    A python object corresponding to the input tflite file.
  """
  model = convert_bytearray_to_object(model_bytearray)
  if sys.byteorder == 'big':
    byte_swap_tflite_model_obj(model, 'little', 'big')

  # Offset handling for buffers.
  for buffer in model.buffers:
    if buffer.offset:
      buffer.data = model_bytearray[buffer.offset : buffer.offset + buffer.size]
      buffer.offset = 0
      buffer.size = 0

  # Offset handling for `Operator.CustomOptions`.
  for subgraph in model.subgraphs:
    for op in subgraph.operators:
      if op.largeCustomOptionsOffset:
        op.customOptions = model_bytearray[
            op.largeCustomOptionsOffset : op.largeCustomOptionsOffset
            + op.largeCustomOptionsSize
        ]
        op.largeCustomOptionsOffset = 0
        op.largeCustomOptionsSize = 0

  # Convert any non-buffer `np.ndarray`s to `list` to ensure they are mutable.
  buffers = model.buffers
  model.buffers = None
  _ndarrays_to_lists(model)
  model.buffers = buffers

  return model


def read_model_with_mutable_tensors(
    input_tflite_file: Path,
) -> ModelT:
  """Reads a tflite model as a python object with mutable tensors.

  Similar to read_model() with the addition that the returned object has
  mutable tensors (read_model() returns an object with immutable tensors).

  NOTE: This API only works for TFLite generated with
  _experimental_use_buffer_offset=false

  Args:
    input_tflite_file: Full path name to the input tflite file

  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A mutable python object corresponding to the input tflite file.
  """
  return copy.deepcopy(read_model(input_tflite_file))


def convert_object_to_bytearray(
    model_object: Packable, extra_buffer: BufferType = b''
) -> bytearray:
  """Converts a tflite model from an object to a immutable bytearray."""
  # Initial size of the buffer, which will grow automatically if needed
  builder = flatbuffers.Builder(1024)
  model_offset = model_object.Pack(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  model_bytearray = builder.Output()
  if extra_buffer:
    model_bytearray = model_bytearray + extra_buffer
  return model_bytearray


def write_model(model_object: ModelT, output_tflite_file: Path):
  """Writes the tflite model, a python object, into the output file.

  NOTE: This API only works for TFLite generated with
  _experimental_use_buffer_offset=false

  Args:
    model_object: A tflite model as a python object
    output_tflite_file: Full path name to the output tflite file.

  Raises:
    IOError: If output_tflite_file path is invalid or cannot be opened.
  """
  if sys.byteorder == 'big':
    model_object = copy.deepcopy(model_object)
    byte_swap_tflite_model_obj(model_object, 'big', 'little')
  model_bytearray = convert_object_to_bytearray(model_object)
  with open(output_tflite_file, 'wb') as output_file_handle:
    output_file_handle.write(model_bytearray)


def strip_strings(model: ModelT):
  """Strips all nonessential strings from the model to reduce model size.

  We remove the following strings:
  (find strings by searching ":string" in the tensorflow lite flatbuffer schema)
  1. Model description
  2. SubGraph name
  3. Tensor names
  We retain OperatorCode custom_code and Metadata name.

  Args:
    model: The model from which to remove nonessential strings.
  """

  model.description = None
  for subgraph in model.subgraphs:
    subgraph.name = None
    for tensor in subgraph.tensors:
      tensor.name = None
  # We clear all signature_def structure, since without names it is useless.
  model.signatureDefs = None


def type_to_name(tensor_type: int) -> str:
  """Converts a numerical enum value to a readable tensor type."""
  return _TENSOR_TYPE_TO_NAME[tensor_type]


def randomize_weights(
    model: ModelT,
    random_seed: int = 0,
    buffers_to_skip: list[int] | None = None,
):
  """Randomize weights in a model.

  Args:
    model: The model in which to randomize weights.
    random_seed: The input to the random number generator (default value is 0).
    buffers_to_skip: The list of buffer indices to skip. The weights in these
      buffers are left unmodified.
  """

  # The input to the random seed generator. The default value is 0.
  random.seed(random_seed)

  # Parse model buffers which store the model weights
  buffers = model.buffers
  buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None
  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]

  buffer_types = {}
  for graph in model.subgraphs:
    for op in graph.operators:
      if op.inputs is None:
        break
      for input_idx in op.inputs:
        tensor = graph.tensors[input_idx]
        buffer_types[tensor.buffer] = type_to_name(tensor.type)

  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size
    if buffer_i_size == 0:
      continue

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # For floats, we need to generate a valid float and then pack it into
    # the raw bytes in place.
    buffer_type = buffer_types.get(i, 'INT8')
    if buffer_type and buffer_type.startswith('FLOAT'):
      format_code = 'e' if buffer_type == 'FLOAT16' else 'f'
      for offset in range(0, buffer_i_size, struct.calcsize(format_code)):
        value = random.uniform(-0.5, 0.5)  # See http://b/152324470#comment2
        struct.pack_into(format_code, buffer_i_data, offset, value)
    else:
      for j in range(buffer_i_size):
        buffer_i_data[j] = random.randint(0, 255)


def rename_custom_ops(model: ModelT, map_custom_op_renames: Mapping[str, str]):
  """Rename custom ops so they use the same naming style as builtin ops.

  Args:
    model: The input tflite model.
    map_custom_op_renames: A mapping from old to new custom op names.
  """
  for op_code in model.operatorCodes:
    if op_code.customCode:
      op_code_str = op_code.customCode.decode('ascii')
      if op_code_str in map_custom_op_renames:
        op_code.customCode = map_custom_op_renames[op_code_str].encode('ascii')


def opcode_to_name(model: ModelT, op_code: int) -> str | None:
  """Converts a TFLite op_code to the human readable name.

  Args:
    model: The input tflite model.
    op_code: The op_code to resolve to a readable name.

  Returns:
    A string containing the human readable op name, or None if not resolvable.
  """
  op: OperatorCodeT = model.operatorCodes[op_code]
  code = get_builtin_code_from_operator_code(op)
  for name, value in vars(BuiltinOperator).items():
    if value == code:
      return name
  return None


def xxd_output_to_bytes(input_cc_file: Path) -> bytes:
  """Converts xxd output C++ source file to bytes (immutable).

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A bytearray corresponding to the input cc file array.
  """
  # Match hex values in the string with comma as separator
  pattern = re.compile(r'\W*(0x[0-9a-fA-F,x ]+).*')

  model_bytearray = bytearray()

  with open(input_cc_file) as file_handle:
    for line in file_handle:
      values_match = pattern.match(line)

      if values_match is None:
        continue

      # Match in the parentheses (hex array only)
      list_text = values_match.group(1)

      # Extract hex values (text) from the line
      # e.g. 0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c,
      values_text = filter(None, list_text.split(','))

      # Convert to hex
      values = [int(x, base=16) for x in values_text]
      model_bytearray.extend(values)

  return bytes(model_bytearray)


def xxd_output_to_object(input_cc_file: Path) -> ModelT:
  """Converts xxd output C++ source file to object.

  Args:
    input_cc_file: Full path name to th C++ source file dumped by xxd

  Raises:
    RuntimeError: If input_cc_file path is invalid.
    IOError: If input_cc_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """
  model_bytes = xxd_output_to_bytes(input_cc_file)
  return convert_bytearray_to_object(model_bytes)


def byte_swap_buffer_content(
    buffer: BufferT,
    chunksize: int,
    from_endiness: Endiness,
    to_endiness: Endiness,
):
  """Helper function for byte-swapping the buffers field."""
  to_swap = [
      buffer.data[i : i + chunksize]
      for i in range(0, len(buffer.data), chunksize)
  ]
  buffer.data = b''.join([
      int.from_bytes(byteswap, from_endiness).to_bytes(chunksize, to_endiness)
      for byteswap in to_swap
  ])


def byte_swap_string_content(
    buffer: BufferT,
    from_endiness: Endiness,
    to_endiness: Endiness,
):
  """Helper function for byte-swapping the string buffer.

  Args:
    buffer: TFLite string buffer of from_endiness format.
    from_endiness: The original endianness format of the string buffer.
    to_endiness: The destined endianness format of the string buffer.
  """
  num_of_strings = int.from_bytes(buffer.data[0:4], from_endiness)
  string_content = bytearray(buffer.data[4 * (num_of_strings + 2) :])
  prefix_data = b''.join([
      int.from_bytes(buffer.data[i : i + 4], from_endiness).to_bytes(
          4, to_endiness
      )
      for i in range(0, (num_of_strings + 1) * 4 + 1, 4)
  ])
  buffer.data = prefix_data + string_content


def byte_swap_tflite_model_obj(
    model: ModelT, from_endiness: Endiness, to_endiness: Endiness
):
  """Byte swaps the buffers field in a TFLite model.

  Args:
    model: TFLite model object of from_endiness format.
    from_endiness: The original endianness format of the buffers in model.
    to_endiness: The destined endianness format of the buffers in model.
  """
  if model is None:
    return
  # Get all the constant buffers, byte swapping them as per their data types
  buffer_swapped = set()
  types_of_16_bits = [
      TensorType.FLOAT16,
      TensorType.INT16,
      TensorType.UINT16,
  ]
  types_of_32_bits = [
      TensorType.FLOAT32,
      TensorType.INT32,
      TensorType.COMPLEX64,
      TensorType.UINT32,
  ]
  types_of_64_bits = [
      TensorType.INT64,
      TensorType.FLOAT64,
      TensorType.COMPLEX128,
      TensorType.UINT64,
  ]
  for subgraph in model.subgraphs:
    for tensor in subgraph.tensors:
      if (
          tensor.buffer > 0
          and tensor.buffer < len(model.buffers)
          and tensor.buffer not in buffer_swapped
          and model.buffers[tensor.buffer].data is not None
      ):
        if tensor.type == TensorType.STRING:
          byte_swap_string_content(
              model.buffers[tensor.buffer], from_endiness, to_endiness
          )
        elif tensor.type in types_of_16_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 2, from_endiness, to_endiness
          )
        elif tensor.type in types_of_32_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 4, from_endiness, to_endiness
          )
        elif tensor.type in types_of_64_bits:
          byte_swap_buffer_content(
              model.buffers[tensor.buffer], 8, from_endiness, to_endiness
          )
        else:
          continue
        buffer_swapped.add(tensor.buffer)


def byte_swap_tflite_buffer(
    tflite_model: BufferType, from_endiness: Endiness, to_endiness: Endiness
):
  """Generates a new model byte array after byte swapping its buffers field.

  Args:
    tflite_model: TFLite flatbuffer in a byte array.
    from_endiness: The original endianness format of the buffers in
      tflite_model.
    to_endiness: The destined endianness format of the buffers in tflite_model.

  Returns:
    TFLite flatbuffer in a byte array, after being byte swapped to to_endiness
    format.
  """
  if tflite_model is None:
    return None
  # Load TFLite Flatbuffer byte array into an object.
  model = convert_bytearray_to_object(tflite_model)

  # Byte swapping the constant buffers as per their data types
  byte_swap_tflite_model_obj(model, from_endiness, to_endiness)

  # Return a TFLite flatbuffer as a byte array.
  return convert_object_to_bytearray(model)


def count_resource_variables(model: ModelT | BufferType) -> int:
  """Calculates the number of unique resource variables in a model.

  Args:
    model: the input tflite model, either as bytearray or object.

  Returns:
    An integer number representing the number of unique resource variables.
  """
  if not isinstance(model, ModelT):
    model = convert_bytearray_to_object(model)
  unique_shared_names = set()
  for subgraph in model.subgraphs:
    if subgraph.operators is None:
      continue
    for op in subgraph.operators:
      builtin_code = get_builtin_code_from_operator_code(
          model.operatorCodes[op.opcodeIndex]
      )
      if builtin_code == BuiltinOperator.VAR_HANDLE:
        unique_shared_names.add(op.builtinOptions.sharedName)
  return len(unique_shared_names)


OptsT = TypeVar('OptsT')


def get_options_as(
    op: Operator | OperatorT, opts_type: Type[OptsT]
) -> OptsT | None:
  """Get the options of an operator as the specified type.

  Requested type must be an object-api type (ends in 'T').

  Args:
    op: The operator to get the options from.
    opts_type: The type of the options to get.

  Returns:
    The options as the specified type, or None if the options are not of the
    specified type.

  Raises:
    ValueError: If the specified type is not a valid options type.
  """

  err = ValueError(f'Unsupported options type: {opts_type}')
  type_name: str = opts_type.__name__
  if not type_name.endswith('T'):
    raise err
  base_type_name = type_name.removesuffix('T')
  is_opt_1_type = hasattr(BuiltinOptions, base_type_name)
  if not is_opt_1_type and not hasattr(BuiltinOptions2, base_type_name):
    raise err

  if isinstance(op, Operator):
    if not is_opt_1_type:
      enum_val = getattr(BuiltinOptions2, base_type_name)
      opts_creator = schema_fb.BuiltinOptions2Creator
      raw_ops = op.BuiltinOptions2()
      actual_enum_val = op.BuiltinOptions2Type()
    else:
      enum_val = getattr(BuiltinOptions, base_type_name)
      opts_creator = schema_fb.BuiltinOptionsCreator
      raw_ops = op.BuiltinOptions()
      actual_enum_val = op.BuiltinOptionsType()
    if raw_ops is None or actual_enum_val != enum_val:
      return None
    return opts_creator(enum_val, raw_ops)

  elif isinstance(op, OperatorT):
    if is_opt_1_type:
      raw_ops_t = op.builtinOptions
    else:
      raw_ops_t = op.builtinOptions2
    if raw_ops_t is None or not isinstance(raw_ops_t, opts_type):
      return None
    return raw_ops_t

  else:
    return None
