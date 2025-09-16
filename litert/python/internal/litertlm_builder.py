# Copyright 2025 The ODML Authors.
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


"""Builder class for LiteRT-LM files.

Example usage:
```
builder = litertlm_builder.LitertLmFileBuilder()
builder.add_system_metadata(
    litertlm_builder.Metadata(
        key="Authors",
        value="The ODML Authors",
        dtype=litertlm_builder.DType.STRING,
    )
)
builder.add_tflite_model(
    model_path,
    litertlm_builder.TfLiteModelType.PREFILL_DECODE,
)
builder.add_sentencepiece_tokenizer(tokenizer_path)
builder.add_llm_metadata(llm_metadata_path)
with litertlm_core.open_file(output_path, "wb") as f:
  builder.build(f)
```
"""

# TODO(b/445163709): Remove this module once litert_lm publishes a pypi package.

import dataclasses
import enum
import os  # pylint: disable=unused-import
from typing import Any, BinaryIO, Callable, IO, Optional, TypeVar, Union
import zlib
import flatbuffers
from google.protobuf import message
from google.protobuf import text_format
from litert.python.internal import litertlm_core
from litert.python.internal import litertlm_header_schema_py_generated as schema
from litert.python.internal import llm_metadata_pb2


@enum.unique
class DType(enum.Enum):
  """DType enum.

  This enum maps to the data types defined in the LiteRT-LM flatbuffers schema.
  """

  INT8 = "int8"
  INT16 = "int16"
  INT32 = "int32"
  INT64 = "int64"
  UINT8 = "uint8"
  UINT16 = "uint16"
  UINT32 = "uint32"
  UINT64 = "uint64"
  FLOAT32 = "float32"
  DOUBLE = "double"
  BOOL = "bool"
  STRING = "string"


@dataclasses.dataclass
class Metadata:
  """Metadata class."""

  key: str
  value: Any
  dtype: DType


@enum.unique
class TfLiteModelType(enum.Enum):
  """TfLiteModelType enum.

  This enum maps to the model types defined in the LiteRT-LM flatbuffers schema.
  """

  PREFILL_DECODE = "tf_lite_prefill_decode"

  EMBEDDER = "tf_lite_embedder"
  PER_LAYER_EMBEDDER = "tf_lite_per_layer_embedder"

  AUX = "tf_lite_aux"

  AUDIO_FRONTEND = "tf_lite_audio_frontend"
  AUDIO_ENCODER_HW = "tf_lite_audio_encoder_hw"
  AUDIO_ADAPTER = "tf_lite_audio_adapter"
  END_OF_AUDIO = "tf_lite_end_of_audio"

  VISION_ENCODER = "tf_lite_vision_encoder"
  VISION_ADAPTER = "tf_lite_vision_adapter"

  @classmethod
  def get_enum_from_tf_free_value(cls, tf_free_value: str) -> "TfLiteModelType":
    """A helper method to get the enum value from the TF-free value."""
    value = "tf_lite_" + tf_free_value.lower()
    return cls(value)


@dataclasses.dataclass
class _SectionObject:
  # Metadata for the section.
  metadata: list[Metadata]
  # The data type of the section.
  data_type: schema.AnySectionDataType | int
  # The data reader for the section. This should return the data as a byte
  # string.
  data_reader: Callable[[], bytes]


LitertLmFileBuilderT = TypeVar(
    "LitertLmFileBuilderT", bound="LitertLmFileBuilder"
)


class LitertLmFileBuilder:
  """LitertLmFileBuilder class.

  This is the primary entry point for building a LiteRT-LM file. It provides
  methods to add system metadata, sections, and llm metadata to the file.

  Example usage:
  ```
    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="The ODML Authors",
            dtype=litertlm_builder.DType.STRING,
        )
    )
    builder.add_tflite_model(
        model_path,
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
    )
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    builder.add_llm_metadata(llm_metadata_path)
    with litertlm_core.open_file(output_path, "wb") as f:
      builder.build(f)
  ```
  """

  def __init__(self):
    self._system_metadata: list[Metadata] = []
    self._sections: list[_SectionObject] = []
    self._has_llm_metadata = False
    self._has_tokenizer = False

  def add_system_metadata(
      self,
      metadata: Metadata,
  ) -> LitertLmFileBuilderT:
    """Adds system level metadata to the litertlm file."""
    for existing_metadata in self._system_metadata:
      if existing_metadata.key == metadata.key:
        raise ValueError(
            f"System metadata already exists for key: {metadata.key}"
        )
    self._system_metadata.append(metadata)
    return self

  def add_llm_metadata(
      self,
      llm_metadata_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds llm metadata to the litertlm file.

    Args:
      llm_metadata_path: The path to the llm metadata file. Can be binary or
        textproto format.
      additional_metadata: Additional metadata to add to the llm metadata.

    Returns:
      The currentLitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the llm metadata file is not found.
    """
    assert not self._has_llm_metadata, "Llm metadata already added."
    self._has_llm_metadata = True
    if not litertlm_core.path_exists(llm_metadata_path):
      raise FileNotFoundError(
          f"Llm metadata file not found: {llm_metadata_path}"
      )

    if _is_binary_proto(llm_metadata_path):

      def data_reader():
        with litertlm_core.open_file(llm_metadata_path, "rb") as f:
          return f.read()

    else:

      def data_reader():
        with litertlm_core.open_file(llm_metadata_path, "r") as f:
          return text_format.Parse(
              f.read(), llm_metadata_pb2.LlmMetadata()
          ).SerializeToString()

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.LlmMetadataProto,
        data_reader=data_reader,
    )
    self._sections.append(section_object)
    return self

  def add_tflite_model(
      self,
      tflite_model_path: str,
      model_type: TfLiteModelType,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a tflite model to the litertlm file.

    Args:
      tflite_model_path: The path to the tflite model file.
      model_type: The type of the tflite model.
      additional_metadata: Additional metadata to add to the tflite model.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the tflite model file is not found.
      ValueError: If the model type metadata is overridden.
    """
    if not litertlm_core.path_exists(tflite_model_path):
      raise FileNotFoundError(
          f"Tflite model file not found: {tflite_model_path}"
      )
    metadata = [
        Metadata(key="model_type", value=model_type.value, dtype=DType.STRING)
    ]
    if additional_metadata:
      for metadata_item in additional_metadata:
        if metadata_item.key == "model_type":
          raise ValueError("Model type metadata cannot be overridden.")
      metadata.extend(additional_metadata)

    def data_reader():
      with litertlm_core.open_file(tflite_model_path, "rb") as f:
        return f.read()

    section_object = _SectionObject(
        metadata=metadata,
        data_type=schema.AnySectionDataType.TFLiteModel,
        data_reader=data_reader,
    )
    self._sections.append(section_object)
    return self

  def add_sentencepiece_tokenizer(
      self,
      sp_tokenizer_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a sentencepiece tokenizer to the litertlm file.

    Args:
      sp_tokenizer_path: The path to the sentencepiece tokenizer file.
      additional_metadata: Additional metadata to add to the sentencepiece
        tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the sentencepiece tokenizer file is not found.
    """
    assert not self._has_tokenizer, "Tokenizer already added."
    self._has_tokenizer = True
    if not litertlm_core.path_exists(sp_tokenizer_path):
      raise FileNotFoundError(
          f"Sentencepiece tokenizer file not found: {sp_tokenizer_path}"
      )

    def data_reader():
      with litertlm_core.open_file(sp_tokenizer_path, "rb") as f:
        return f.read()

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.SP_Tokenizer,
        data_reader=data_reader,
    )
    self._sections.append(section_object)
    return self

  def add_hf_tokenizer(
      self,
      hf_tokenizer_path: str,
      additional_metadata: Optional[list[Metadata]] = None,
  ) -> LitertLmFileBuilderT:
    """Adds a hf tokenizer to the litertlm file.

    Args:
      hf_tokenizer_path: The path to the hf tokenizer `tokenizer.json` file.
      additional_metadata: Additional metadata to add to the hf tokenizer.

    Returns:
      The current LitertLmFileBuilder object.

    Raises:
      FileNotFoundError: If the hf tokenizer file is not found.
    """
    assert not self._has_tokenizer, "Tokenizer already added."
    self._has_tokenizer = True
    if not litertlm_core.path_exists(hf_tokenizer_path):
      raise FileNotFoundError(
          f"HF tokenizer file not found: {hf_tokenizer_path}"
      )

    def read_and_compress(path: str) -> bytes:
      with litertlm_core.open_file(path, "rb") as f:
        content = f.read()
        uncompressed_size = len(content)
        compressed_content = zlib.compress(content)
        return uncompressed_size.to_bytes(8, "little") + compressed_content

    section_object = _SectionObject(
        metadata=additional_metadata if additional_metadata else [],
        data_type=schema.AnySectionDataType.HF_Tokenizer_Zlib,
        data_reader=lambda: read_and_compress(hf_tokenizer_path),
    )
    self._sections.append(section_object)
    return self

  def build(
      self,
      stream: BinaryIO | IO[Union[bytes, str]],
  ) -> None:
    """Builds the litertlm into the given stream."""
    stream.seek(0)
    # To simplify the build logic, we reserved the first block for the header.
    # This translates to the first block will be padded to `BLOCK_SIZE`.
    # TODO(b/413978412): support headers > 16KB.
    stream.write(b"\0" * litertlm_core.BLOCK_SIZE)

    # Write sections
    offsets = []
    for section in self._sections:
      start_offset = stream.tell()
      stream.write(section.data_reader())
      end_offset = stream.tell()
      offsets.append((start_offset, end_offset))
      _write_padding(stream, litertlm_core.BLOCK_SIZE)

    # write header
    self._write_header(stream, offsets)

  def _write_header(
      self, stream: BinaryIO, offsets: list[tuple[int, int]]
  ) -> None:
    """Writes the header to the stream."""
    assert self._system_metadata, "System metadata is empty."

    stream.seek(0)
    stream.write(b"LITERTLM")
    stream.write(litertlm_core.LITERTLM_MAJOR_VERSION.to_bytes(4, "little"))
    stream.write(litertlm_core.LITERTLM_MINOR_VERSION.to_bytes(4, "little"))
    stream.write(litertlm_core.LITERTLM_PATCH_VERSION.to_bytes(4, "little"))
    _write_padding(stream, litertlm_core.HEADER_BEGIN_BYTE_OFFSET)
    stream.write(self._get_header_data(offsets))
    header_end_offset = stream.tell()
    if header_end_offset > litertlm_core.BLOCK_SIZE:
      raise ValueError("Header size exceeds 16KB limit.")
    stream.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
    stream.write(header_end_offset.to_bytes(8, "little"))

  def _get_header_data(self, offsets: list[tuple[int, int]]) -> bytearray:
    builder = flatbuffers.Builder(1024)
    system_metadata_offset = self._write_system_metadata(builder)
    section_metadata_offset = self._write_section_metadata(builder, offsets)
    schema.LiteRTLMMetaDataStart(builder)
    schema.LiteRTLMMetaDataAddSystemMetadata(builder, system_metadata_offset)
    schema.LiteRTLMMetaDataAddSectionMetadata(builder, section_metadata_offset)
    root = schema.LiteRTLMMetaDataEnd(builder)
    builder.Finish(root)
    return builder.Output()

  def _write_system_metadata(self, builder: flatbuffers.Builder) -> int:
    """Writes the system metadata to the builder."""
    system_metadata_offsets = [
        _write_metadata(builder, m) for m in self._system_metadata
    ]
    schema.SystemMetadataStartEntriesVector(
        builder, len(system_metadata_offsets)
    )
    for offsets in reversed(system_metadata_offsets):
      builder.PrependUOffsetTRelative(offsets)
    entries_vec = builder.EndVector()
    schema.SystemMetadataStart(builder)
    schema.SystemMetadataAddEntries(builder, entries_vec)
    return schema.SystemMetadataEnd(builder)

  def _write_section_metadata(
      self, builder: flatbuffers.Builder, offsets: list[tuple[int, int]]
  ) -> int:
    """Writes the section metadata to the builder."""
    assert len(self._sections) == len(offsets)

    section_objects_offsets = []
    for section, offset in zip(self._sections, offsets):
      section_objects_offsets.append(
          _write_section_object(
              builder, section.metadata, offset, section.data_type
          )
      )

    schema.SectionMetadataStartObjectsVector(
        builder, len(section_objects_offsets)
    )
    for obj in reversed(section_objects_offsets):
      builder.PrependUOffsetTRelative(obj)
    objects_vec = builder.EndVector()

    schema.SectionMetadataStart(builder)
    schema.SectionMetadataAddObjects(builder, objects_vec)
    return schema.SectionMetadataEnd(builder)


def _is_binary_proto(filepath: str) -> bool:
  """Checks if a file is a binary protobuf or a textproto version of LlmMetadata.

  Args:
      filepath (str): The path to the file.

  Returns:
      bool: True if the file is a binary protobuf, False if it's a textproto.
      TextProto.
  """
  assert litertlm_core.path_exists(filepath), f"File {filepath} does not exist."

  try:
    with litertlm_core.open_file(filepath, "rb") as f:
      content = f.read()
      msg = llm_metadata_pb2.LlmMetadata()
      msg.ParseFromString(content)
      if msg.IsInitialized():
        return True
  except message.DecodeError:
    # This is expected if the file is in text format. We'll just pass and try
    # the next format.
    pass

  try:
    with litertlm_core.open_file(filepath, "r") as f:
      content = f.read()
      msg = text_format.Parse(content, llm_metadata_pb2.LlmMetadata())
      if msg.IsInitialized():
        return False
  except (text_format.ParseError, UnicodeDecodeError) as e:
    raise ValueError(f"Failed to parse LlmMetadata from {filepath}.") from e


def _write_padding(stream: BinaryIO, block_size: int) -> None:
  """Writes zero padding to align to the next block size."""
  current_pos = stream.tell()
  padding_needed = (block_size - (current_pos % block_size)) % block_size
  if padding_needed > 0:
    stream.write(b"\0" * padding_needed)


def _write_metadata(builder: flatbuffers.Builder, metadata: Metadata) -> int:
  """Writes a FlatBuffers KeyValuePair."""
  key_offset = builder.CreateString(metadata.key)

  if metadata.dtype == DType.BOOL:
    schema.BoolStart(builder)
    schema.BoolAddValue(builder, metadata.value)
    value_offset = schema.BoolEnd(builder)
    value_type = schema.VData.Bool
  elif metadata.dtype == DType.INT8:
    schema.Int8Start(builder)
    schema.Int8AddValue(builder, metadata.value)
    value_offset = schema.Int8End(builder)
    value_type = schema.VData.Int8
  elif metadata.dtype == DType.INT16:
    schema.Int16Start(builder)
    schema.Int16AddValue(builder, metadata.value)
    value_offset = schema.Int16End(builder)
    value_type = schema.VData.Int16
  elif metadata.dtype == DType.INT32:
    schema.Int32Start(builder)
    schema.Int32AddValue(builder, metadata.value)
    value_offset = schema.Int32End(builder)
    value_type = schema.VData.Int32
  elif metadata.dtype == DType.INT64:
    schema.Int64Start(builder)
    schema.Int64AddValue(builder, metadata.value)
    value_offset = schema.Int64End(builder)
    value_type = schema.VData.Int64
  elif metadata.dtype == DType.UINT8:
    schema.UInt8Start(builder)
    schema.UInt8AddValue(builder, metadata.value)
    value_offset = schema.UInt8End(builder)
    value_type = schema.VData.UInt8
  elif metadata.dtype == DType.UINT16:
    schema.UInt16Start(builder)
    schema.UInt16AddValue(builder, metadata.value)
    value_offset = schema.UInt16End(builder)
    value_type = schema.VData.UInt16
  elif metadata.dtype == DType.UINT32:
    schema.UInt32Start(builder)
    schema.UInt32AddValue(builder, metadata.value)
    value_offset = schema.UInt32End(builder)
    value_type = schema.VData.UInt32
  elif metadata.dtype == DType.UINT64:
    schema.UInt64Start(builder)
    schema.UInt64AddValue(builder, metadata.value)
    value_offset = schema.UInt64End(builder)
    value_type = schema.VData.UInt64
  elif metadata.dtype == DType.FLOAT32:
    schema.Float32Start(builder)
    schema.Float32AddValue(builder, metadata.value)
    value_offset = schema.Float32End(builder)
    value_type = schema.VData.Float32
  elif metadata.dtype == DType.DOUBLE:
    schema.DoubleStart(builder)
    schema.DoubleAddValue(builder, metadata.value)
    value_offset = schema.DoubleEnd(builder)
    value_type = schema.VData.Double
  elif metadata.dtype == DType.STRING:
    value_offset_str = builder.CreateString(str(metadata.value))
    schema.StringValueStart(builder)
    schema.StringValueAddValue(builder, value_offset_str)
    value_offset = schema.StringValueEnd(builder)
    value_type = schema.VData.StringValue
  else:
    raise ValueError(f"Unsupported dtype: {metadata.dtype}")

  schema.KeyValuePairStart(builder)
  schema.KeyValuePairAddKey(builder, key_offset)
  schema.KeyValuePairAddValueType(builder, value_type)
  schema.KeyValuePairAddValue(builder, value_offset)
  return schema.KeyValuePairEnd(builder)


def _write_section_object(
    builder: flatbuffers.Builder,
    section_metadata: list[Metadata],
    section_offset: tuple[int, int],
    section_type: schema.AnySectionDataType,
) -> int:
  """Writes a FlatBuffers SectionObject."""
  section_metadata_offsets = [
      _write_metadata(builder, m) for m in section_metadata
  ]
  schema.SectionObjectStartItemsVector(builder, len(section_metadata_offsets))
  for offsets in reversed(section_metadata_offsets):
    builder.PrependUOffsetTRelative(offsets)
  items_vec = builder.EndVector()
  schema.SectionObjectStart(builder)
  schema.SectionObjectAddItems(builder, items_vec)
  schema.SectionObjectAddBeginOffset(builder, section_offset[0])
  schema.SectionObjectAddEndOffset(builder, section_offset[1])
  schema.SectionObjectAddDataType(builder, section_type)
  return schema.SectionObjectEnd(builder)
