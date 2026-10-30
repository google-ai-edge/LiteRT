# Copyright 2024 Google LLC
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

"""Utility functions for graph transformations."""

from collections.abc import MutableMapping
import dataclasses
from typing import Optional

import numpy as np

from litert_quantizer import qtyping


@dataclasses.dataclass
class TransformationInput:
  """Standard input for a graph transformation.

  Attributes:
    tensor_id: the tensor index to insert dequant op after
    model: The original TFlite model for buffer quantization
    subgraph: flatbuffer subgraph object which the tensor resides.
    producer: op id for the producer of the tensor.
    consumers: op ids for consumers of the new dequant op.
    quant_params: quantization parameters to be applied on the orignal tensor
    buffer_origin: A mapping of buffer indices to the quantization parameters
      used to populate them, used to avoid duplicate (and detect conflicting)
      writes.
  """

  tensor_id: int
  model: qtyping.ModelT
  subgraph: qtyping.SubGraphT
  producer: int
  consumers: list[int]
  quant_params: qtyping.UniformQuantParams | qtyping.NonLinearQuantParams
  buffer_origin: MutableMapping[
      int, qtyping.UniformQuantParams | qtyping.NonLinearQuantParams
  ] = dataclasses.field(default_factory=dict)


class HashableMemoryView:
  """Hashable wrapper class for a `memoryview`."""

  _mv: memoryview

  def __init__(self, value):
    if isinstance(value, np.ndarray):
      value = value.reshape(-1).view(np.uint8)
    self._mv = memoryview(value)

  def __hash__(self):
    # There are two things going on here:
    # * We only hash the shape, format, and first 16 entries to avoid traversing
    #   the entire buffer. Note that if two hashes match, the objects still
    #   have to test equal.
    # * We pack the first 16 entries of the buffer into a `tuple` because
    #   `memoryview.__hash__` delegates to the underlying buffer, which may be
    #   unhashable, e.g. a `np.ndarray`.
    return hash((self._mv.shape, self._mv.format, tuple(self._mv[:16])))

  def __eq__(self, value, /):
    if isinstance(value, self.__class__):
      return self._mv == value._mv
    return self._mv == value


def add_op_code(
    op_code: qtyping.OperatorCodeT,
    model_op_codes: list[qtyping.OperatorCodeT],
    custom_op_name: Optional[str] = None,
) -> int:
  """Add an op code into a model if it's not present.

  Args:
    op_code: The op code to be added.
    model_op_codes: The op codes of the model.
    custom_op_name: The custom string of the op code. If None, the op code will
      be added as a builtin op code.

  Returns:
    The index of the op code in the model.
  """
  if op_code == qtyping.BuiltinOperator.CUSTOM and custom_op_name is None:
    raise ValueError('Custom string is required for custom op code.')

  for i, model_op_code in enumerate(model_op_codes):
    # If the model already has the op code, just return the index.
    if model_op_code.builtinCode == op_code:
      if custom_op_name is not None:
        if model_op_code.customCode == custom_op_name:
          return i
      else:
        # Built-in op
        return i

  model_op_codes.append(qtyping.OperatorCodeT())
  model_op_codes[-1].builtinCode = op_code
  if custom_op_name is not None:
    model_op_codes[-1].customCode = custom_op_name
  return len(model_op_codes) - 1


def get_constant_buffer(
    data: np.ndarray | qtyping.BufferType,
    model: qtyping.ModelT,
    force_duplicate_buffer: bool = False,
) -> int:
  """Get the index of the constant buffer that contains the given data.

  creating new buffer if provided data is not found in buffers list.

  Args:
    data: The data of the new tensor.
    model: The model holding the buffers.
    force_duplicate_buffer: Whether to add a new buffer even if the same buffer
      already exists.

  Returns:
    The index of the new buffer in the model.
  """
  # Convert the data to a 1-D `memoryview`.
  if isinstance(data, np.ndarray):
    # in the case where the data is passed from quantization_params.
    data = np.ravel(data.view(np.uint8))

  # Check if the model has a buffer lookup mapping, and if not, add one.
  if not (id_for_buffer_data := getattr(model, '_id_for_buffer_data', None)):  # pyrefly: ignore[bad-assignment]
    id_for_buffer_data: dict[HashableMemoryView, int] = {}
    for buffer_id, buffer in enumerate(model.buffers):
      if (buffer_data := buffer.data) is not None:
        id_for_buffer_data[HashableMemoryView(buffer_data)] = buffer_id
    model._id_for_buffer_data = id_for_buffer_data  # pylint: disable=protected-access

  # Check if we already have a buffer for this data.
  if not force_duplicate_buffer and (
      index := id_for_buffer_data.get(HashableMemoryView(data))
  ) is not None:
    return index

  # Create a new `qtyping.BufferT` object.
  new_buffer = qtyping.BufferT()
  new_buffer.data = data
  new_buffer.offset = 0
  new_buffer.size = 0

  # Store the new buffer.
  new_buffer_id = len(model.buffers)
  model.buffers.append(new_buffer)
  id_for_buffer_data[HashableMemoryView(data)] = new_buffer_id

  return new_buffer_id


def add_new_constant_tensor(
    tensor_name: str,
    data: np.ndarray,
    tensor_type: qtyping.TensorType,
    subgraph: qtyping.SubGraphT,
    model: qtyping.ModelT,
    tensor_shape: Optional[list[int]] = None,
    force_duplicate_buffer: bool = False,
    quantization: qtyping.QuantizationParametersT | None = None,
    allow_tensor_sharing: bool = False,
) -> int:
  """Add a new constant tensor to the model.

  Args:
    tensor_name: The name of the new tensor.
    data: The data of the new tensor.
    tensor_type: The type of the new tensor.
    subgraph: The subgraph where the new tensor is added.
    model: The model holding the buffers.
    tensor_shape: The shape of the new tensor. If not provided, the shape of the
      data will be used.
    force_duplicate_buffer: Whether to add a new buffer even if the same buffer
      already exists.
    quantization: Optional `QuantizationParametersT` describing the quantization
      of this tensor.
    allow_tensor_sharing: Whether to allow sharing an existing tensor with the
      same data, shape, type, and quantization.

  Returns:
    The index of the new tensor in the subgraph.
  """
  new_buffer_id = get_constant_buffer(data, model, force_duplicate_buffer)

  if allow_tensor_sharing and not force_duplicate_buffer:
    expected_shape = (
        list(tensor_shape) if tensor_shape is not None else list(data.shape)
    )

    # To avoid O(N) linear scans when adding many shared constant tensors
    # (e.g. identical Hadamard matrices), we lazily initialize an O(1)
    # lookup cache on the subgraph.
    # The cache mapping is:
    #   Key: (buffer_id, shape_tuple, tensor_type, has_quantization)
    #   Value: tensor_index in the subgraph
    if not (tensor_lookup := getattr(subgraph, '_tensor_lookup', None)):
      tensor_lookup = {}
      for i, t in enumerate(subgraph.tensors):
        key = (t.buffer, tuple(t.shape), t.type, t.quantization is not None)
        tensor_lookup[key] = i
      subgraph._tensor_lookup = tensor_lookup

    key = (
        new_buffer_id,
        tuple(expected_shape),
        tensor_type,
        quantization is not None,
    )
    if (existing_idx := tensor_lookup.get(key)) is not None:
      return existing_idx

  new_tensor = qtyping.TensorT()
  if tensor_shape is None:
    tensor_shape = data.shape  # pyrefly: ignore[bad-assignment]
  new_tensor.shape = tensor_shape
  new_tensor.buffer = new_buffer_id
  new_tensor.type = tensor_type
  new_tensor.name = tensor_name
  new_tensor.quantization = quantization
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)
  if (tensor_lookup := getattr(subgraph, '_tensor_lookup', None)) is not None:
    key = (
        new_buffer_id,
        tuple(new_tensor.shape),  # pyrefly: ignore[bad-argument-type]
        new_tensor.type,
        new_tensor.quantization is not None,
    )
    tensor_lookup[key] = new_tensor_id
  return new_tensor_id


def add_new_activation_tensor(
    tensor_name: str,
    shape: list[int],
    tensor_type: qtyping.TensorType,
    subgraph: qtyping.SubGraphT,
    quantization: qtyping.QuantizationParametersT | None = None,
) -> int:
  """Add a new activation tensor to the model.

  Args:
    tensor_name: The name of the new tensor.
    shape: The shape of the new tensor.
    tensor_type: The type of the new tensor.
    subgraph: The subgraph where the new tensor is added.
    quantization: Optional `QuantizationParametersT` describing the quantization
      of this tensor.

  Returns:
    The index of the new tensor in the subgraph.
  """
  new_tensor = qtyping.TensorT()
  # If there's a dynamic shape, we need to read from the shapeSignature field
  # instead of shape. Shape should contain just 1 for the dynamic dimension but
  # shapeSignature should contain the true shape.
  if -1 in shape:
    new_tensor.shapeSignature = shape
    new_tensor.shape = [1 if i == -1 else i for i in shape]
  else:
    new_tensor.shape = shape
  new_tensor.type = tensor_type
  new_tensor.name = tensor_name
  new_tensor.quantization = quantization
  new_tensor.buffer = 0
  new_tensor_id = len(subgraph.tensors)
  subgraph.tensors.append(new_tensor)
  return new_tensor_id


def raise_deprecated_error(_: TransformationInput):
  raise NotImplementedError(
      'This transformation is deprecated. Please contact LiteRT Quantizer team'
      ' if you see this error.'
  )


def pack_data(bitwidth: int, data: np.ndarray) -> np.ndarray:
  """Pack the data to the corresponding bit width.

  Currently only support 2 and 4 bits. If no packing is needed, the original
  data is returned.

  Args:
    bitwidth: Bit width from NonLinearQuantParams.
    data: The data to be packed.

  Returns:
    Packed data.
  """
  # Flatten the data without copying it.
  data = data.reshape(-1)
  if bitwidth in(2, 4):
    # Some useful constants.
    values_per_byte = 8 // bitwidth
    mask = (1 << bitwidth) - 1

    # Slice the buffers into `values_per_bytes` interleaved views (no copies are
    # made at this point).
    *buffers, last = [
        data[offset::values_per_byte]  # Creates a view, no copy.
        for offset in range(values_per_byte)
    ]

    # Convert the last of the buffers to `np.uint8` making a copy of it. This
    # is the buffer we will pack our data into.
    last = last.astype(np.uint8)  # Makes a copy.
    last &= mask  # Mask values (in-place).

    # Loop over the remaining buffers in reverse order, i.e. last-1 to first.
    # Note that since we only ever make a copy of the data in each `buffer`
    # inside of this loop, only at most len(data) // values_per_byte * 2 bytes
    # are allocated (`last` and `buffer`).
    for buffer in reversed(buffers):
      # Shift the values (in-place, no copy) to the left to make room for the
      # next chunk of bits.
      last <<= bitwidth

      # Convert the buffer values to `np.uint8` (makes a copy) and mask out any
      # extra bits.
      buffer = buffer.astype(np.uint8)  # Makes a copy.
      buffer &= mask  # Mask values (in-place).

      # If `buffer` is larger than `last` (happens if `len(data)` is not an
      # integer multiple of `values_per_buffer`), swap the buffers to pack the
      # data into the larger of the two. This avoids explicitly padding
      # `last`, which creates an extra copy.
      if len(buffer) > len(last):
        buffer, last = last, buffer

      # Merge `last` and `buffer`, keeping in mind that `last` may be larger.
      last[: len(buffer)] |= buffer  # Merge with previous bits (in-place).

    return last

  else:
    # Nothing to pack.
    return data


def get_producer_schema_op_id(
    transformation: TransformationInput,
) -> int:
  """Checks if the tensor's producer matches the given op.

  Args:
    transformation: The transformation input to check the producer of.

  Returns:
    The schema op id of the producer op. E.g.
    qtyping.BuiltinOperator.FULLY_CONNECTED.
  """
  if transformation.producer == -1:
    return False
  else:
    return transformation.model.operatorCodes[
        transformation.subgraph.operators[transformation.producer].opcodeIndex
    ].builtinCode


def get_schema_op_id(
    transformation: TransformationInput, op_id: int
) -> bool:
  """Returns the schema op id of the given op.

  Args:
    transformation: The transformation input to check the consumers of.
    op_id: The op id in the list of operators to check for.

  Returns:
    The schema op id of the given op.
  """
  return transformation.model.operatorCodes[
      transformation.subgraph.operators[op_id].opcodeIndex
  ].builtinCode


def update_fully_connected_consumers(
    transformation: TransformationInput,
    new_tensor_id: int,
) -> bool:
  """Updates the fully connected op(s) to use the new tensor.

  Since the new tensor is inserted to the fully_connected's input, we need to
  scan each consumer (in case of multiple fully_connected ops), and update
  the input tensor to the new tensor.

  Args:
    transformation: The transformation input to update the consumers of.
    new_tensor_id: The new tensor id to use as the input to the fully connected
      consumers.

  Returns:
    True if the fully connected op(s) were updated to use the new tensor.
  """
  updated = False
  for consumer in transformation.consumers:
    if (
        get_schema_op_id(transformation, consumer)
        == qtyping.BuiltinOperator.FULLY_CONNECTED
    ):
      transformation.subgraph.operators[consumer].inputs[0] = new_tensor_id
      updated = True
  return updated
