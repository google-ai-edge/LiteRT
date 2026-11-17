// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/nvidia/tools/carve_container.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_model.h"

namespace litert::tools {
namespace {

constexpr char kMagic[] = "LITERTLM";
constexpr size_t kMagicSize = 8;
// The file starts with the magic, three 32-bit version numbers, four bytes of
// padding and the 64-bit offset of the header's end; the header follows.
constexpr uint64_t kMajorVersionOffset = 8;
constexpr uint64_t kHeaderEndOffset = 24;
constexpr uint64_t kHeaderOffset = 32;
constexpr uint32_t kMajorVersion = 1;
// A section starts on a multiple of this.
constexpr uint64_t kBlockSize = 16 * 1024;
// The position in the VData union of the string value of an item.
constexpr uint8_t kStringValue = 9;

constexpr size_t kWriteChunk = size_t{64} << 20;

// Reads the header, a flatbuffer, by the field numbers of its schema:
//   LiteRTLMMetaData { 0: system_metadata, 1: section_metadata }
//   SectionMetadata  { 0: objects:[SectionObject] }
//   SectionObject    { 0: items:[KeyValuePair], 1: begin_offset:ulong,
//                      2: end_offset:ulong, 3: data_type:ubyte }
//   KeyValuePair     { 0: key:string, 1: value type, 2: value (a table) }
//   StringValue      { 0: value:string }
// Positions are offsets in the file; a read outside the header fails the
// whole parse. Flatbuffers are little-endian, as are the hosts this runs on.
class HeaderReader {
 public:
  HeaderReader(const uint8_t* data, uint64_t size) : data_(data), size_(size) {}

  bool ok() const { return ok_; }

  template <class T>
  T Read(uint64_t position) {
    T value{};
    if (position > size_ || sizeof(T) > size_ - position) {
      ok_ = false;
    } else {
      std::memcpy(&value, data_ + position, sizeof(T));
    }
    return value;
  }

  // The position of field `index` of the table at `table`, 0 if the table
  // does not store it.
  uint64_t Field(uint64_t table, int index) {
    if (table == 0) return 0;
    const uint64_t vtable = table - Read<int32_t>(table);
    const uint64_t slot = 4 + 2 * static_cast<uint64_t>(index);
    if (slot + 2 > Read<uint16_t>(vtable)) return 0;
    const uint16_t offset = Read<uint16_t>(vtable + slot);
    return offset == 0 ? 0 : table + offset;
  }

  // The table, vector or string that the offset at `position` points to.
  uint64_t Follow(uint64_t position) {
    return position == 0 ? 0 : position + Read<uint32_t>(position);
  }

  // The elements of the vector of offsets at `vector`.
  std::vector<uint64_t> Elements(uint64_t vector) {
    std::vector<uint64_t> elements;
    if (vector == 0) return elements;
    const uint32_t count = Read<uint32_t>(vector);
    for (uint32_t i = 0; ok_ && i < count; ++i) {
      elements.push_back(Follow(vector + 4 + 4 * uint64_t{i}));
    }
    return elements;
  }

  std::string String(uint64_t string) {
    if (string == 0) return "";
    const uint32_t length = Read<uint32_t>(string);
    if (string + 4 > size_ || length > size_ - string - 4) {
      ok_ = false;
      return "";
    }
    return std::string(reinterpret_cast<const char*>(data_ + string + 4),
                       length);
  }

 private:
  const uint8_t* data_;
  const uint64_t size_;
  bool ok_ = true;
};

uint64_t RoundUpToBlock(uint64_t offset) {
  return (offset + kBlockSize - 1) / kBlockSize * kBlockSize;
}

// Writes to a file and keeps the first error.
class Writer {
 public:
  explicit Writer(const std::string& path)
      : path_(path),
        stream_(path, std::ios::binary | std::ios::out | std::ios::trunc) {}

  void Write(const uint8_t* data, uint64_t size, Sha256* hash = nullptr) {
    while (size > 0) {
      const size_t chunk =
          static_cast<size_t>(std::min<uint64_t>(size, kWriteChunk));
      stream_.write(reinterpret_cast<const char*>(data), chunk);
      if (hash != nullptr) hash->Update(data, chunk);
      data += chunk;
      size -= chunk;
    }
  }

  void WriteZeros(uint64_t size, Sha256* hash = nullptr) {
    const std::vector<uint8_t> zeros(
        static_cast<size_t>(std::min<uint64_t>(size, kBlockSize)));
    while (size > 0) {
      const uint64_t chunk = std::min<uint64_t>(size, zeros.size());
      Write(zeros.data(), chunk, hash);
      size -= chunk;
    }
  }

  void Write(const CarvedModel& model, Sha256* hash = nullptr) {
    // The const accessor is the base class's.
    const litert::BufferRef<uint8_t>& flatbuffer = model.flatbuffer;
    Write(flatbuffer.Data(), flatbuffer.Size(), hash);
    WriteZeros(model.gap, hash);
    Write(model.appended.Data(), model.appended.Size(), hash);
  }

  void Seek(uint64_t offset) {
    stream_.seekp(static_cast<std::streamoff>(offset));
  }

  absl::Status Close() {
    stream_.close();
    if (stream_.fail()) {
      return absl::DataLossError(absl::StrCat("could not write ", path_));
    }
    return absl::OkStatus();
  }

 private:
  const std::string path_;
  std::ofstream stream_;
};

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t RotateRight(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::Compress(const uint8_t* block) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = uint32_t{block[4 * i]} << 24 | uint32_t{block[4 * i + 1]} << 16 |
           uint32_t{block[4 * i + 2]} << 8 | uint32_t{block[4 * i + 3]};
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^
                        (w[i - 15] >> 3);
    const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^
                        (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 =
        RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
    const uint32_t choice = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + choice + kRoundConstants[i] + w[i];
    const uint32_t s0 =
        RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::Update(const uint8_t* data, size_t size) {
  total_size_ += size;
  if (block_size_ > 0) {
    const size_t take = std::min(size, sizeof(block_) - block_size_);
    std::memcpy(block_ + block_size_, data, take);
    block_size_ += take;
    data += take;
    size -= take;
    if (block_size_ < sizeof(block_)) return;
    Compress(block_);
    block_size_ = 0;
  }
  for (; size >= sizeof(block_);
       data += sizeof(block_), size -= sizeof(block_)) {
    Compress(data);
  }
  if (size > 0) std::memcpy(block_, data, size);
  block_size_ = size;
}

std::string Sha256::HexDigest() {
  // The message ends with a 1 bit, zeros up to 8 bytes before a block's end,
  // and its length in bits.
  const uint64_t bits = total_size_ * 8;
  uint8_t padding[72] = {0x80};
  const size_t padding_size = (block_size_ < 56 ? 56 : 120) - block_size_;
  Update(padding, padding_size);
  uint8_t length[8];
  for (int i = 0; i < 8; ++i) {
    length[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  }
  Update(length, sizeof(length));

  static constexpr char kDigits[] = "0123456789abcdef";
  std::string digest;
  for (const uint32_t word : state_) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      digest.push_back(kDigits[(word >> shift) & 0xf]);
    }
  }
  return digest;
}

bool IsContainer(litert::BufferRef<uint8_t> file) {
  return file.Size() >= kHeaderOffset &&
         std::memcmp(file.Data(), kMagic, kMagicSize) == 0;
}

absl::StatusOr<std::vector<ContainerSection>> ParseContainer(
    litert::BufferRef<uint8_t> file) {
  if (!IsContainer(file)) {
    return absl::InvalidArgumentError("not a LiteRT-LM container");
  }
  HeaderReader file_reader(file.Data(), file.Size());
  const uint32_t major_version =
      file_reader.Read<uint32_t>(kMajorVersionOffset);
  if (major_version != kMajorVersion) {
    return absl::UnimplementedError(
        absl::StrCat("LiteRT-LM container version ", major_version));
  }
  const uint64_t header_end = file_reader.Read<uint64_t>(kHeaderEndOffset);
  if (header_end < kHeaderOffset || header_end > file.Size()) {
    return absl::InvalidArgumentError("the container's header is cut off");
  }

  HeaderReader reader(file.Data(), header_end);
  const uint64_t root = reader.Follow(kHeaderOffset);
  const uint64_t section_metadata = reader.Follow(reader.Field(root, 1));
  std::vector<ContainerSection> sections;
  for (const uint64_t object :
       reader.Elements(reader.Follow(reader.Field(section_metadata, 0)))) {
    ContainerSection section;
    section.begin_field = reader.Field(object, 1);
    section.end_field = reader.Field(object, 2);
    if (section.begin_field == 0 || section.end_field == 0) {
      return absl::InvalidArgumentError(
          "a section of the container has no offsets");
    }
    section.begin = reader.Read<uint64_t>(section.begin_field);
    section.end = reader.Read<uint64_t>(section.end_field);
    if (const uint64_t data_type = reader.Field(object, 3); data_type != 0) {
      section.data_type = reader.Read<uint8_t>(data_type);
    }
    for (const uint64_t item :
         reader.Elements(reader.Follow(reader.Field(object, 0)))) {
      const uint64_t value_type = reader.Field(item, 1);
      if (value_type == 0 || reader.Read<uint8_t>(value_type) != kStringValue) {
        continue;
      }
      const std::string key =
          reader.String(reader.Follow(reader.Field(item, 0)));
      const uint64_t value =
          reader.Follow(reader.Field(reader.Follow(reader.Field(item, 2)), 0));
      const std::string text = reader.String(value);
      if (key == "model_type") {
        section.model_type = text;
      } else if (key == "sha256" && text.size() == 64) {
        section.sha256_field = value + 4;
      }
    }
    const uint64_t previous_end =
        sections.empty() ? header_end : sections.back().end;
    if (!reader.ok() || section.begin < previous_end ||
        section.end < section.begin || section.end > file.Size()) {
      return absl::InvalidArgumentError(
          "the sections of the container are not in file order inside the "
          "file");
    }
    sections.push_back(std::move(section));
  }
  if (!reader.ok() || sections.empty()) {
    return absl::InvalidArgumentError("cannot read the container's header");
  }
  return sections;
}

absl::StatusOr<std::vector<ContainerSection>> WriteContainer(
    litert::BufferRef<uint8_t> file,
    const std::vector<ContainerSection>& sections,
    const std::vector<std::optional<CarvedModel>>& replacements,
    const std::string& path) {
  if (sections.empty() || replacements.size() != sections.size()) {
    return absl::InvalidArgumentError("one replacement slot for each section");
  }
  // Everything before the first section: the header, patched below.
  std::vector<uint8_t> header(file.Data(), file.Data() + sections[0].begin);
  auto patch = [&](uint64_t field, const void* data, size_t size) {
    std::memcpy(header.data() + field, data, size);
  };

  Writer writer(path);
  writer.Write(header.data(), header.size());
  std::vector<ContainerSection> written = sections;
  uint64_t position = header.size();
  bool moved = false;
  for (size_t i = 0; i < sections.size(); ++i) {
    // The layout of LiteRT-LM's builder once a section has changed size; the
    // source's own layout up to there.
    const uint64_t begin = moved ? RoundUpToBlock(position) : sections[i].begin;
    writer.WriteZeros(begin - position);
    auto& section = written[i];
    section.begin = begin;
    if (replacements[i].has_value()) {
      Sha256 hash;
      writer.Write(*replacements[i], &hash);
      section.end = begin + replacements[i]->Size();
      if (section.sha256_field != 0) {
        const std::string digest = hash.HexDigest();
        patch(section.sha256_field, digest.data(), digest.size());
      }
      moved |= section.end != sections[i].end;
    } else {
      writer.Write(file.Data() + sections[i].begin,
                   sections[i].end - sections[i].begin);
      section.end = begin + (sections[i].end - sections[i].begin);
    }
    patch(section.begin_field, &section.begin, sizeof(section.begin));
    patch(section.end_field, &section.end, sizeof(section.end));
    position = section.end;
  }
  // What the source has behind its last section, if anything.
  writer.Write(file.Data() + sections.back().end,
               file.Size() - sections.back().end);
  writer.Seek(0);
  writer.Write(header.data(), header.size());
  if (auto status = writer.Close(); !status.ok()) return status;
  return written;
}

absl::Status WriteModel(const CarvedModel& model, const std::string& path) {
  Writer writer(path);
  writer.Write(model);
  return writer.Close();
}

absl::Status WriteFile(litert::BufferRef<uint8_t> bytes,
                       const std::string& path) {
  Writer writer(path);
  writer.Write(bytes.Data(), bytes.Size());
  return writer.Close();
}

}  // namespace litert::tools
