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

"""Utilities to read and parse LiteRT-LM files."""

from collections.abc import Mapping
import copy
import pathlib
import struct
from typing import Any, TypeVar

import flatbuffers

import os
import io
from ai_edge_litert.tools import flatbuffer_utils
from ai_edge_litert.tools import mmap_utils
from litert_lm_builder import litertlm_builder
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_header_schema_py_generated as schema
from litert_quantizer import qtyping

# Exported types.
SectionObjectT = schema.SectionObjectT
AnySectionDataType = schema.AnySectionDataType

# Internal types.
_Path = str | pathlib.Path
T = TypeVar('T')


# TODO: b/495756579 - Fix a mis-named function in the generated schema.
schema.VDataCreator = schema.VdataCreator  # pyrefly: ignore[missing-attribute]


def _bytes_to_str(val: bytes | T) -> str | T:
  """Converts the input to `str` if it is `bytes`."""
  return str(val, encoding='utf-8') if isinstance(val, bytes) else val


def _from_key_value_pair(kvp: Any) -> litertlm_builder.Metadata:
  """Helper to construct Metadata from key-value pair with fallback."""
  if hasattr(litertlm_builder.Metadata, 'from_key_value_pair'):
    return litertlm_builder.Metadata.from_key_value_pair(kvp)
  key = kvp.key.decode('utf-8') if isinstance(kvp.key, bytes) else kvp.key
  val_obj = kvp.value
  if hasattr(val_obj, 'value'):
    val_obj = val_obj.value
  value = val_obj.decode('utf-8') if isinstance(val_obj, bytes) else val_obj
  if hasattr(litertlm_builder, 'DType'):
    return litertlm_builder.Metadata(
        key=key, value=value, dtype=getattr(litertlm_builder, 'DType').STRING
    )
  return litertlm_builder.Metadata(key=key, value=value)  # pyrefly: ignore[missing-argument]


class LiteRTLMFile:
  """Represents a LiteRT-LM file, which may contain one or more TFLite models.

  On initialization, the LiteRTLM file header is read and parsed, and a list
   sections (offsets in the LiteRTLM file) is created.

  Subsequent calls to `get_section_type` or `read_model` will use this list to
  extract data from the LiteRT-LM file.
  """

  _path: _Path
  _metadata: schema.LiteRTLMMetaDataT
  _sections: list[SectionObjectT]

  def __init__(self, path: _Path):
    self._path = path
    self._metadata = self._get_metadata()
    self._sections = self._get_sections()

  def _get_metadata(self) -> schema.LiteRTLMMetaDataT:
    """Extracts the packed metadata from the file header."""
    # TODO: b/495763732 - This should be part of litertlm_core.

    # Read and check the file header to extract the size of the serializaed
    # metadata.
    header_bytes = mmap_utils.get_file_contents(
        self._path, size=litertlm_core.HEADER_BEGIN_BYTE_OFFSET
    )
    assert header_bytes[:8] == litertlm_core.HEADER_MAGIC_BYTES

    # Get the end offset of the serialized metadata.
    header_end_offset = struct.unpack(
        '<Q',
        header_bytes[litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET :][:8],
    )[0]

    # Extract and parse the metadata flatbuffer.
    metadata_buffer = mmap_utils.get_file_contents(
        self._path,
        offset=litertlm_core.HEADER_BEGIN_BYTE_OFFSET,
        size=header_end_offset - litertlm_core.HEADER_BEGIN_BYTE_OFFSET,
    )
    return schema.LiteRTLMMetaDataT.InitFromPackedBuf(metadata_buffer)

  def _get_sections(self) -> list[SectionObjectT]:
    """Extracts the mapping of `model_type` to `SectionObjectT`."""
    # Get the SectionObjects from the metadata.
    if (
        (metadata := self._metadata)
        and (section_metadata := metadata.sectionMetadata)
        and (objects := section_metadata.objects)
    ):
      return objects[:]
    return []

  @property
  def sections(self) -> list[SectionObjectT]:
    """Returns a list of `SectionObjectT` contained in this LiteRT-LM file."""
    return self._sections

  def get_system_metadata(self) -> dict[str, Any]:
    """Returns a `dict` of key/value pairs corresponding to the system metadata of this LiteRT-LM file."""
    if (system_metadata := self._metadata.systemMetadata) and (
        entries := system_metadata.entries
    ):
      return {
          _bytes_to_str(entry.key): _bytes_to_str(entry.value.value)
          for entry in entries
      }
    return {}

  def get_section_metadata(self, section_id: int) -> dict[str, Any]:
    """Returns a `dict` of key/value pairs corresponding to the system metadata of the given section of the LiteRT-LM file."""
    if items := self._sections[section_id].items:
      return {
          _bytes_to_str(item.key): _bytes_to_str(item.value.value)
          for item in items
      }
    return {}

  def get_model_type(self, section_id: int) -> str | None:
    """Returns the model_type of the given section of the LiteRT-LM file."""
    for key, val in self.get_section_metadata(section_id).items():
      if key == 'model_type':
        return _bytes_to_str(val)
    return None

  def read_model(self, section_id: int) -> qtyping.ModelT | None:
    """Extracts a `qtyping.ModelT` from the given section of this LiteRT-LM file."""
    obj = self._sections[section_id]
    if obj.dataType == AnySectionDataType.TFLiteModel:
      return flatbuffer_utils.read_model_from_bytearray(
          mmap_utils.get_file_contents(
              self._path,
              offset=obj.beginOffset,
              size=obj.endOffset - obj.beginOffset,
          )
      )
    return None

  def get_section_buffer(self, section_id: int) -> qtyping.BufferType | None:
    """Extracts the raw contents of the given section of this LiteRT-LM file."""
    obj = self._sections[section_id]
    return mmap_utils.get_file_contents(
        self._path,
        offset=obj.beginOffset,
        size=obj.endOffset - obj.beginOffset,
    )

  def serialize(
      self,
      path: _Path,
      section_data_overrides: Mapping[int, qtyping.BufferType],
  ) -> int:
    """Serializes the LiteRTLMFile to the given path.

    Args:
      path: Location at which to create the LiteRT-LM file.
      section_data_overrides: An optional mapping of `int` to
        `qtyping.BufferType` where the value will replace the contents of the
        section matching the key.

    Returns:
      The number of bytes written to the specified file.
    """
    # Compute the length and offset of each section. Note that there is no
    # guarantee that the sections are ordered by their location in the
    # LiteRT-LM file!
    section_offsets = [
        min(self._sections, key=lambda s: s.beginOffset).beginOffset
    ]
    section_lengths = []
    for section_id, section in enumerate(self._sections):
      if buff := section_data_overrides.get(section_id):
        section_length = len(buff)
      else:
        section_length = section.endOffset - section.beginOffset
      section_lengths.append(section_length)
      section_offsets.append(
          (section_offsets[-1] + section_length + litertlm_core.BLOCK_SIZE - 1)
          & ~(litertlm_core.BLOCK_SIZE - 1)
      )

    # Make a copy of the metadata with the modified section offsets.
    metadata = copy.deepcopy(self._metadata)
    assert metadata.sectionMetadata and metadata.sectionMetadata.objects
    for section_id, section in enumerate(metadata.sectionMetadata.objects):
      section.beginOffset = section_offsets[section_id]
      section.endOffset = section.beginOffset + section_lengths[section_id]

    # Update the system metadata where needed.
    if not metadata.systemMetadata:
      metadata.systemMetadata = schema.SystemMetadataT(entries=[])
    system_metadata = metadata.systemMetadata
    system_metadata.entries = [
        m.to_key_value_pair()
        for m in litertlm_builder.populate_system_metadata(
            [_from_key_value_pair(kvp) for kvp in system_metadata.entries]  # pyrefly: ignore[not-iterable]
        )
    ]

    # Pack the modified metadata.
    metadata_builder = flatbuffers.Builder(1024)
    metadata_builder.Finish(metadata.Pack(metadata_builder))
    assert metadata_builder.Offset() <= section_offsets[0]

    # Create a memory-mapped file in which we will pack the LiteRTLM file.
    if (
        buff := mmap_utils.get_mapped_buffer_or_none(path, section_offsets[-1])
    ) is None:
      buff = open(path, 'wb')

    # Write the header and metadata. Note that we don't use `buff.seek` to avoid
    # failing on file systems that don't support it.
    # TODO: b/495763732 - This should be part of litertlm_core.
    buff_offset = 0
    buff_offset += buff.write(litertlm_core.HEADER_MAGIC_BYTES)
    buff_offset += buff.write(
        litertlm_core.LITERTLM_MAJOR_VERSION.to_bytes(4, 'little')
    )
    buff_offset += buff.write(
        litertlm_core.LITERTLM_MINOR_VERSION.to_bytes(4, 'little')
    )
    buff_offset += buff.write(
        litertlm_core.LITERTLM_PATCH_VERSION.to_bytes(4, 'little')
    )
    buff_offset += buff.write(int(0).to_bytes(4, 'little'))  # Zero padding.

    # Write the offset of the end of the serialized metadata flatbuffer.
    assert buff_offset == litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET
    buff_offset += buff.write(
        (
            litertlm_core.HEADER_BEGIN_BYTE_OFFSET + metadata_builder.Offset()
        ).to_bytes(8, 'little')
    )

    # Write the serialized metadata flatbuffer.
    assert buff_offset == litertlm_core.HEADER_BEGIN_BYTE_OFFSET
    buff_offset += buff.write(metadata_builder.Output())

    # Write the section data.
    for section_id, section in enumerate(metadata.sectionMetadata.objects):
      # Move to the aligned offset of the next section.
      buff_offset += buff.write(b'\x00' * (section.beginOffset - buff_offset))

      # Write the section data.
      if (section_data := section_data_overrides.get(section_id)) is None:
        section_data = self.get_section_buffer(section_id)
      buff_offset += buff.write(section_data)  # pyrefly: ignore[bad-argument-type]

    # Close the output file/mmap.
    buff.close()

    # Return the number of bytes written.
    return buff_offset
