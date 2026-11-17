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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_CONTAINER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_CONTAINER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_model.h"

// Reading and rewriting the files the carve tool works on: a .tflite model,
// or a LiteRT-LM container (.litertlm), which is a header and sections
// (models, weights, a tokenizer, metadata) that start on 16 KiB boundaries.
// The header is a flatbuffer of LiteRT-LM's schema/core/
// litertlm_header_schema.fbs; rewriting a section changes only numbers and
// checksums of fixed width in it, so it is patched in place and keeps its
// size.
namespace litert::tools {

// The AnySectionDataType of a TFLite model section.
inline constexpr uint8_t kContainerTfliteModel = 3;

struct ContainerSection {
  // The section's bytes are [begin, end) of the file.
  uint64_t begin = 0;
  uint64_t end = 0;
  uint8_t data_type = 0;
  // The "model_type" item of the section, if it has one.
  std::string model_type;
  // Where the header holds `begin` and `end`, and the 64 hexadecimal digits
  // of the section's "sha256" item (0 if it has none), as offsets in the
  // file.
  uint64_t begin_field = 0;
  uint64_t end_field = 0;
  uint64_t sha256_field = 0;
};

bool IsContainer(litert::BufferRef<uint8_t> file);

// The sections of the container `file`, in file order.
absl::StatusOr<std::vector<ContainerSection>> ParseContainer(
    litert::BufferRef<uint8_t> file);

// Writes to `path` the container `file` with each section i for which
// `replacements[i]` holds a model replaced by that model. The other sections
// are copied byte for byte; a section moves only when one before it changed
// size. Returns the sections of the new file.
absl::StatusOr<std::vector<ContainerSection>> WriteContainer(
    litert::BufferRef<uint8_t> file,
    const std::vector<ContainerSection>& sections,
    const std::vector<std::optional<CarvedModel>>& replacements,
    const std::string& path);

// Writes `model` to `path` as a .tflite file.
absl::Status WriteModel(const CarvedModel& model, const std::string& path);

// Writes `bytes` to `path`.
absl::Status WriteFile(litert::BufferRef<uint8_t> bytes,
                       const std::string& path);

// SHA-256 (FIPS 180-4), for the checksums a container keeps of its sections.
class Sha256 {
 public:
  Sha256();
  void Update(const uint8_t* data, size_t size);
  // The digest as 64 lowercase hexadecimal digits. Ends the computation.
  std::string HexDigest();

 private:
  void Compress(const uint8_t* block);

  uint32_t state_[8];
  uint8_t block_[64];
  size_t block_size_ = 0;
  uint64_t total_size_ = 0;
};

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_CONTAINER_H_
