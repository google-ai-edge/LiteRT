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
"""Externalize supported TFLite weight buffers into one weight blob.

This FlatBuffer-level tool moves only large constants that LiteRT-LM's scoped
external-weight path can consume: weights at input 1 of FullyConnected, Conv2D,
DepthwiseConv2D, and EmbeddingLookup. Other constants remain embedded so they do
not become delegate partition inputs.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import flatbuffers
import numpy as np

from litert.python import schema_py_generated as schema  # pylint:disable=g-direct-tensorflow-import

_LITERT_HOST_MEMORY_BUFFER_ALIGNMENT = 64


def _enum_value(enum_obj, name: str) -> int:
  enum_cls = getattr(enum_obj, enum_obj.__name__, enum_obj)
  return int(getattr(enum_cls, name))


def _builtin_code(operator_code) -> int:
  builtin = int(operator_code.builtinCode)
  if builtin == 0 and int(operator_code.deprecatedBuiltinCode) != 0:
    return int(operator_code.deprecatedBuiltinCode)
  return builtin


def _tensor_element_count(tensor) -> int:
  if tensor.shape is None:
    return 0
  shape = np.asarray(tensor.shape, dtype=np.int64)
  if shape.size == 0 or np.any(shape < 0):
    return 0
  return int(np.prod(shape, dtype=np.int64))


def _buffer_bytes(buffer) -> memoryview | None:
  if buffer.data is None:
    return None
  data = np.asarray(buffer.data, dtype=np.uint8)
  if data.size == 0:
    return None
  return memoryview(data)


def _bias_tensor_indices(model) -> set[tuple[int, int]]:
  bias_ops = {
      _enum_value(schema.BuiltinOperator, "FULLY_CONNECTED"),
      _enum_value(schema.BuiltinOperator, "CONV_2D"),
      _enum_value(schema.BuiltinOperator, "DEPTHWISE_CONV_2D"),
  }
  bias_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for op in subgraph.operators or []:
      if op.inputs is None or len(op.inputs) <= 2:
        continue
      opcode = model.operatorCodes[int(op.opcodeIndex)]
      if _builtin_code(opcode) in bias_ops:
        bias_idx = int(op.inputs[2])
        if bias_idx >= 0:
          bias_tensors.add((sg_idx, bias_idx))
  return bias_tensors


def _shareable_weight_tensor_indices(model) -> set[tuple[int, int]]:
  weight_ops = {
      _enum_value(schema.BuiltinOperator, "EMBEDDING_LOOKUP"),
      _enum_value(schema.BuiltinOperator, "FULLY_CONNECTED"),
      _enum_value(schema.BuiltinOperator, "CONV_2D"),
      _enum_value(schema.BuiltinOperator, "DEPTHWISE_CONV_2D"),
  }
  weight_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for op in subgraph.operators or []:
      if op.inputs is None or len(op.inputs) <= 1:
        continue
      opcode = model.operatorCodes[int(op.opcodeIndex)]
      if _builtin_code(opcode) in weight_ops:
        weight_idx = int(op.inputs[1])
        if weight_idx >= 0:
          weight_tensors.add((sg_idx, weight_idx))
  return weight_tensors


def _subgraph_input_indices(model) -> set[tuple[int, int]]:
  input_tensors: set[tuple[int, int]] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    if subgraph.inputs is None:
      continue
    for tensor_idx in subgraph.inputs:
      if int(tensor_idx) >= 0:
        input_tensors.add((sg_idx, int(tensor_idx)))
  return input_tensors


def _referenced_buffers(model, externalized: set[tuple[int, int]]) -> set[int]:
  referenced: set[int] = set()
  for sg_idx, subgraph in enumerate(model.subgraphs or []):
    for tensor_idx, tensor in enumerate(subgraph.tensors or []):
      if (sg_idx, tensor_idx) not in externalized:
        referenced.add(int(tensor.buffer))
  return referenced


def _ensure_group(model, group_name: str) -> int:
  if model.externalBufferGroups is None:
    model.externalBufferGroups = []
  if not model.externalBufferGroups:
    placeholder = schema.ExternalBufferGroupT()
    placeholder.name = ""
    model.externalBufferGroups.append(placeholder)
  for idx, group in enumerate(model.externalBufferGroups):
    name = (
        group.name.decode("utf-8")
        if isinstance(group.name, bytes)
        else group.name
    )
    if name == group_name:
      return idx
  group = schema.ExternalBufferGroupT()
  group.name = group_name
  model.externalBufferGroups.append(group)
  return len(model.externalBufferGroups) - 1


def _make_external_buffer_id(model) -> int:
  # Match LiteRT flatbuffer_export.cc: external buffer IDs have the MSB set so
  # they cannot be confused with ordinary TFLite buffer indices.
  return (1 << 31) | len(model.externalBuffers or [])


def externalize(
    input_model: Path,
    output_dir: Path,
    group_name: str,
    num_elements_threshold: int,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  output_model = output_dir / "model.tflite"
  output_weights = output_dir / group_name

  input_bytes = input_model.read_bytes()
  model = schema.ModelT.InitFromPackedBuf(input_bytes, 0)
  if model.externalBuffers is None:
    model.externalBuffers = []

  group_idx = _ensure_group(model, group_name)
  bias_tensors = _bias_tensor_indices(model)
  shareable_weight_tensors = _shareable_weight_tensor_indices(model)
  input_tensors = _subgraph_input_indices(model)
  externalized_tensors: set[tuple[int, int]] = set()
  candidate_buffers: set[int] = set()
  bytes_cache: dict[tuple[int, bytes], tuple[int, int]] = {}
  total_externalized_bytes = 0

  with output_weights.open("wb") as weights_file:
    for sg_idx, subgraph in enumerate(model.subgraphs or []):
      for tensor_idx, tensor in enumerate(subgraph.tensors or []):
        tensor_key = (sg_idx, tensor_idx)
        if tensor_key not in shareable_weight_tensors:
          continue
        if tensor_key in input_tensors or tensor_key in bias_tensors:
          continue
        if int(tensor.externalBuffer) != 0:
          continue
        if tensor.isVariable:
          continue
        if _tensor_element_count(tensor) <= num_elements_threshold:
          continue

        buffer_idx = int(tensor.buffer)
        if buffer_idx <= 0 or buffer_idx >= len(model.buffers or []):
          continue
        data = _buffer_bytes(model.buffers[buffer_idx])
        if data is None:
          continue

        digest = hashlib.sha256(data).digest()
        cache_key = (len(data), digest)
        if cache_key in bytes_cache:
          offset, length = bytes_cache[cache_key]
        else:
          padding = (
              -weights_file.tell()
          ) % _LITERT_HOST_MEMORY_BUFFER_ALIGNMENT
          if padding:
            weights_file.write(b"\0" * padding)
          offset = weights_file.tell()
          length = len(data)
          weights_file.write(data)
          bytes_cache[cache_key] = (offset, length)
          total_externalized_bytes += length

        external_buffer_id = _make_external_buffer_id(model)
        external_buffer = schema.ExternalBufferT()
        external_buffer.id = external_buffer_id
        external_buffer.group = group_idx
        external_buffer.offset = offset
        external_buffer.length = length
        external_buffer.packing = "unpacked"
        model.externalBuffers.append(external_buffer)

        tensor.externalBuffer = external_buffer_id
        candidate_buffers.add(buffer_idx)
        tensor.buffer = 0
        externalized_tensors.add(tensor_key)

  still_referenced = _referenced_buffers(model, externalized_tensors)
  cleared_buffers = 0
  for buffer_idx in candidate_buffers:
    if buffer_idx not in still_referenced and 0 <= buffer_idx < len(
        model.buffers
    ):
      model.buffers[buffer_idx].data = None
      model.buffers[buffer_idx].offset = 0
      model.buffers[buffer_idx].size = 0
      cleared_buffers += 1

  builder = flatbuffers.Builder(1024)
  root = model.Pack(builder)
  builder.Finish(root, file_identifier=b"TFL3")
  output_model.write_bytes(bytes(builder.Output()))

  print(f"Reading model from {input_model}")
  print(f"Writing model to {output_model}")
  print(
      f"Writing weight file {output_weights}: {total_externalized_bytes:,}"
      " bytes"
  )
  print("======")
  print(f"Externalized tensors: {len(externalized_tensors):,}")
  print(f"Cleared buffers: {cleared_buffers:,}")
  print(
      f"Original model size: {input_model.stat().st_size / 1024 / 1024:.2f}MB"
  )
  print(
      f"Processed model size: {output_model.stat().st_size / 1024 / 1024:.2f}MB"
  )
  print(
      "Processed total weight files size:"
      f" {output_weights.stat().st_size / 1024 / 1024:.2f}MB"
  )
  print("======")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_model", type=Path, required=True)
  parser.add_argument("--output_dir", type=Path, required=True)
  parser.add_argument("--group_name", default="tflite_weights")
  parser.add_argument("--num_elements_threshold", type=int, default=256)
  args = parser.parse_args()
  externalize(
      input_model=args.input_model,
      output_dir=args.output_dir,
      group_name=args.group_name,
      num_elements_threshold=args.num_elements_threshold,
  )


if __name__ == "__main__":
  main()
