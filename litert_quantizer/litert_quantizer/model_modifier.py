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

"""Model Modifier class that produce the final quantized TFlite model."""

import logging
import mmap
from typing import TypeVar

import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_litert.tools import mmap_utils
from litert_quantizer import qtyping
from litert_quantizer import transformation_instruction_generator
from litert_quantizer import transformation_performer
from litert_quantizer.utils import tfl_flatbuffer_utils


_DEQUANT_SUFFIX = "_dequant"
_QUANT_SUFFIX = "_quantized"


def _round_up_16(offset: int) -> int:
  """Round the given `offset` to the next multiple of 16."""
  return (offset + 15) & ~15


class _PackedBufferData:
  """Holds a `ModelT`'s buffer data for packing."""

  data_size: int
  packed_size: int
  data_for_buffer_id: dict[int, memoryview]

  def __init__(self, model: qtyping.ModelT, min_size_bytes: int = 1024):
    self.data_for_buffer_id = {}
    self.data_size = 0
    self.packed_size = 0
    for buffer_id, buffer in enumerate(model.buffers):
      if buffer.data is not None and min_size_bytes <= len(buffer.data):
        # Convert the buffer data to a `memoryview` of its bytes.
        buffer_data = buffer.data
        if not isinstance(buffer_data, np.ndarray):
          buffer_data = np.array(buffer_data, dtype=np.uint8)
        buffer_data = memoryview(np.ravel(buffer_data).view(np.uint8))

        # Add this buffer to the list of buffers.
        self.data_for_buffer_id[buffer_id] = buffer_data
        self.data_size += len(buffer_data)
        self.packed_size = _round_up_16(self.packed_size + len(buffer_data))
    logging.debug(
        "Created _PackedBufferData with %d buffers, data_size=%d,"
        " packed_size=%d.",
        len(self.data_for_buffer_id),
        self.data_size,
        self.packed_size,
    )


T = TypeVar("T")


def _copy_with_views(value: T) -> T:
  """Recursively convert a nested structure without copying `np.ndarray`s.

  If the input is a `list`, this function recurses over its elements. If the
  input has a `__dict__` attribute, this function recurses over its non-`None`
  entries.

  Args:
    value: The object in which to replace `np.ndarray`s with `list`s.

  Returns:
    The modified value.
  """
  if isinstance(value, np.ndarray):
    return value.view()
  if isinstance(value, list):
    return [_copy_with_views(v) for v in value]  # pyrefly: ignore[bad-return]
  if hasattr(value, "__dict__"):
    return type(value)(**{
        k: _copy_with_views(v)
        for k, v in value.__dict__.items()
        if v is not None
    })
  return value


class ModelModifier:
  """Model Modifier class that produce the final quantized TFlite model."""

  def __init__(self, float_model: qtyping.ModelT):
    """Constructor.

    Args:
      float_model: the original TFlite model.
    """
    self._model: qtyping.ModelT = float_model

    self._transformation_instruction_generator = (
        transformation_instruction_generator.TransformationInstructionsGenerator()
    )
    self._transformation_performer = (
        transformation_performer.TransformationPerformer()
    )

  def _get_tensor_processing_order(
      self,
      tensor_names: set[str],
      flatbuffer_model: qtyping.ModelT,
  ) -> list[str]:
    """Get the tensor processing order obtained from `buffer_to_tensors`.

    The processing order is used to ensure that last tensor in a buffer is
    processed the last. This is required for the correctness of buffer
    duplication, as the last tensor in a buffer won't be duplicated.

    Args:
      tensor_names: Names of the tensors that need to be processed.
      flatbuffer_model: TFlite model.

    Returns:
      A list of tensor names in the processing order.
    """
    buffer_to_tensors = tfl_flatbuffer_utils.buffer_to_tensors(flatbuffer_model)

    processing_order = []
    for buffer_tensors in buffer_to_tensors.values():
      for tensor in buffer_tensors:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(tensor)
        if tensor_name in tensor_names:
          processing_order.append(tensor_name)

    return processing_order

  def modify_model(
      self,
      params: dict[str, qtyping.TensorTransformationParams],
      serialize_to_path: qtyping.Path | None = None,
      enable_progress_bar: bool | None = None,
  ) -> qtyping.BufferType:
    """Modify the model.

    Args:
      params: a dictionary with tensor name and a list of tensor transformation
        params
      serialize_to_path: If set, the quantized model will be serialized to this
        path.
      enable_progress_bar: Whether to enable the progress bar. By default, it is
        disabled for smaller models and enabled for larger models.

    Returns:
      a byte buffer that represents the serialized tflite model.
    """
    # Make a copy of the model, but don't duplicate the buffer data.
    quantized_model = _copy_with_views(self._model)

    instructions = self._transformation_instruction_generator.quant_params_to_transformation_insts(
        params, quantized_model, enable_progress_bar
    )

    tensor_processing_order = self._get_tensor_processing_order(
        set(instructions.keys()), quantized_model
    )

    self._transformation_performer.transform_graph(
        instructions, quantized_model, tensor_processing_order,
        enable_progress_bar,
    )
    del tensor_processing_order

    # Update signature defs if dequant is inserted before output.
    if self._has_transform_before_output(
        instructions, qtyping.QuantTransformation.ADD_DEQUANTIZE
    ):
      quantized_model = self._update_signature_defs(
          quantized_model, _DEQUANT_SUFFIX
      )

    # Update signature defs if quant is inserted before output.
    if self._has_transform_before_output(
        instructions, qtyping.QuantTransformation.ADD_QUANTIZE
    ):
      quantized_model = self._update_signature_defs(
          quantized_model, _QUANT_SUFFIX
      )
    del instructions

    logging.info("Serializing model...")
    packed_buffer_data = _PackedBufferData(quantized_model)
    if packed_buffer_data.packed_size < 256 * 1024:
      serialized_quantized_model = self._serialize_small_model(quantized_model)
      if serialize_to_path:
        mmap_utils.set_file_contents(
            serialize_to_path, serialized_quantized_model
        )
    else:
      serialized_quantized_model = self._serialize_model(
          quantized_model,
          packed_buffer_data,
          serialize_to_path=serialize_to_path,
      )

    return serialized_quantized_model

  def _update_signature_defs(
      self,
      model: qtyping.ModelT,
      suffix: str,
  ) -> qtyping.ModelT:
    """Updates the signature definitions in the model.

    This function is called when a transformation (quantize or dequantize)
    is inserted before an output tensor. It updates the tensor index in the
    signature definitions to point to the newly inserted output tensor.

    Args:
      model: The TFlite ModelT object.
      suffix: The suffix to append to the tensor name.

    Returns:
      The updated TFlite ModelT object.
    """
    for signature_def in model.signatureDefs:
      signature_key = signature_def.signatureKey.decode("utf-8")
      logging.info("Signature = %s", signature_key)
      subgraph_idx = signature_def.subgraphIndex
      subgraph = model.subgraphs[subgraph_idx]
      graph_info = qtyping.GraphInfo(subgraph.tensors, model.buffers)

      for output in subgraph.outputs:
        tensor_name = tfl_flatbuffer_utils.get_tensor_name(
            graph_info.subgraph_tensors[output]
        )
        logging.info("\tOutput tensor = `%s`", tensor_name)

        # for signature_name, tensor_details in output_details.items():
        for signature_item in signature_def.outputs:
          output_tensor = subgraph.tensors[signature_item.tensorIndex]
          output_tensor_name = tfl_flatbuffer_utils.get_tensor_name(
              output_tensor
          )
          if output_tensor_name + suffix == tensor_name:
            logging.info(
                "\t\tfound tensor mapping: `%s`->`%s` for signature name: `%s`",
                output_tensor_name,
                tensor_name,
                signature_item.name,
            )
            logging.info(
                "\t\t\tswapping tensor index: %s->%s",
                signature_item.tensorIndex,
                output,
            )
            signature_item.tensorIndex = output
            break

    return model

  def _has_transform_before_output(
      self,
      instructions: dict[str, qtyping.TensorTransformationInsts],
      transformation: qtyping.QuantTransformation,
  ) -> bool:
    """Check if the model has transformation insert to output."""
    for tensor_name, tensor_trans_insts in instructions.items():
      for instr in tensor_trans_insts.instructions:  # pyrefly: ignore[not-iterable]
        if transformation == instr.transformation and instr.consumers == [-1]:
          logging.info(
              "Found %s insert to output for tensor: %s",
              transformation,
              tensor_name,
          )
          return True
    return False

  def _serialize_model(
      self,
      quantized_model: qtyping.ModelT,
      packed_buffer_data: _PackedBufferData,
      serialize_to_path: qtyping.Path | None = None,
  ) -> qtyping.BufferType:
    """Serialize a model using external buffers.

    Args:
      quantized_model: a quantized TFlite ModelT.
      packed_buffer_data: a `_PackedBufferData` created from `quantized_model`.
      serialize_to_path: If set, the quantized model will be serialized to this
        path.

    Returns:
      a `bytearray` that represents the serialized tflite model
    """
    # Clear the buffer data and set the offset and size to a non-zero value so
    # that they are still packed into the flatbuffer.
    for buffer_id, buffer in enumerate(quantized_model.buffers):
      if buffer_id in packed_buffer_data.data_for_buffer_id:
        buffer.data = None
        buffer.offset = 1
        buffer.size = 1

    # Serialize the model.
    model_buffer = flatbuffer_utils.convert_object_to_bytearray(quantized_model)

    # Always round the buffer length up to the next multiple of 16 bytes so that
    # the buffer data is 16-byte aligned for potential faster reading with SIMD
    # instructions.
    buffer_data_offset = _round_up_16(len(model_buffer))

    # Resize the model_buffer to accommodate for the buffer data. Note that we
    # do it this way since `bytearray.resize` is only available as of
    # Python 3.14.
    if (
        serialize_to_path
        and (
            combined_buffer := mmap_utils.get_mapped_buffer_or_none(
                serialize_to_path,
                buffer_data_offset + packed_buffer_data.packed_size,
            )
        )
        is not None
    ):
      mmap_utils.advise_sequential(combined_buffer)
    else:
      combined_buffer = bytearray(
          buffer_data_offset + packed_buffer_data.packed_size
      )
    combined_buffer[: len(model_buffer)] = model_buffer
    model_buffer = combined_buffer

    # Get the model's serialized representation.
    model_packed = qtyping.Model.GetRootAs(model_buffer)

    # Pack the buffer contents at the end of the model_buffer, setting the
    # correct sizes and offsets in the original quantized_model.
    for buffer_id, buffer_data in packed_buffer_data.data_for_buffer_id.items():
      # Update the packed buffer with the correct size and offset.
      buffer = quantized_model.buffers[buffer_id]
      buffer.offset = buffer_data_offset
      buffer.size = len(buffer_data)
      flatbuffer_utils.update_packed_buffer(
          model_packed.Buffers(buffer_id),
          offset=buffer.offset,
          size=buffer.size,
      )

      # Pack the buffer at the end of the model_buffer.
      model_buffer[
          buffer_data_offset : buffer_data_offset + len(buffer_data)
      ] = memoryview(buffer_data)

      # Increment the offset by the amount of data that we added, plus padding.
      buffer_data_offset = _round_up_16(buffer_data_offset + len(buffer_data))
      mmap_utils.advise_dont_need(buffer_data)
      if isinstance(model_buffer, mmap.mmap):
        mmap_utils.advise_dont_need(model_buffer, 0, buffer_data_offset)

    # Clean up and write the buffer to a file if requested/needed.
    if isinstance(model_buffer, mmap.mmap):
      model_buffer.flush()
    elif serialize_to_path:
      mmap_utils.set_file_contents(serialize_to_path, model_buffer)

    return model_buffer

  def _serialize_small_model(
      self,
      quantized_model: qtyping.ModelT,
  ) -> bytearray:
    """serialize models < 2GB.

    Args:
      quantized_model: a quantized TFlite ModelT

    Returns:
      a byte buffer that represents the serialized tflite model
    """
    model_bytearray = flatbuffer_utils.convert_object_to_bytearray(
        quantized_model
    )
    return model_bytearray
