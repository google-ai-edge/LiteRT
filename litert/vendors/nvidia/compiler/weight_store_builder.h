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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_STORE_BUILDER_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_STORE_BUILDER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/vendors/nvidia/bytecode.h"

namespace litert::nvidia {

// Where the plugin of one launch reads its packed weights.
struct TensorRtWeightLocation {
  uint32_t segment = 0;  // index into TensorRtWeightStoreBuilder::segments()
  uint64_t offset = 0;   // of the weights in the segment
};

// Lays out the packed plugin weights of the partitions of one model as
// segments made of pieces of the model file, so that the TensorRT plans can
// be built without the weights and the engines can share them on the device
// (bytecode.h, TensorRtWeightStore).
//
// The weights of a launch are the concatenation of constant buffers of the
// model ("members"), which must be views of the memory-mapped model file
// that still hold what the file holds.
// Launches with the same members get the same location, whichever partition
// asks: the partitions of one model read the same segments wherever they use
// the same weights. A partition's segments are final once EndPartition() has
// closed the one it was filling; later partitions add segments of their own.
class TensorRtWeightStoreBuilder {
 public:
  struct Segment {
    // A multiple of the granule once the segment is closed.
    uint64_t size = 0;
    bool closed = false;
    // Of the source file and the pieces; set when the segment is closed.
    TensorRtArtifactFingerprint key;
    std::vector<TensorRtWeightPiece> pieces;
  };

  // The launches of a segment start at multiples of this many bytes.
  static constexpr uint64_t kEntryAlignment = 128;

  // `model_path` is the file the model was mapped from, `granule` the CUDA
  // virtual memory granule, and a segment is closed when the next launch
  // would take it past `segment_bytes` (a launch larger than that gets a
  // segment of its own). kLiteRtStatusErrorNotFound if no mapping of the file
  // is found in this process.
  static Expected<std::unique_ptr<TensorRtWeightStoreBuilder>> Create(
      const std::string& model_path, uint64_t granule, uint64_t segment_bytes);

  ~TensorRtWeightStoreBuilder();
  TensorRtWeightStoreBuilder(const TensorRtWeightStoreBuilder&) = delete;
  TensorRtWeightStoreBuilder& operator=(const TensorRtWeightStoreBuilder&) =
      delete;

  // kLiteRtStatusErrorNotFound if a member is not a view of the model file
  // or no longer holds the bytes of the file; the caller then keeps those
  // weights in the plan.
  Expected<TensorRtWeightLocation> Add(
      absl::Span<const absl::Span<const uint8_t>> members);

  // Closes the segment that is being filled.
  void EndPartition();

  const std::vector<Segment>& segments() const { return segments_; }
  uint64_t granule() const { return granule_; }
  const std::string& source_path() const { return source_path_; }
  uint64_t source_size() const { return source_size_; }
  const TensorRtAotFileIdentity& source_identity() const {
    return source_identity_;
  }

 private:
  struct Mapping {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    uint64_t file_offset = 0;
  };

  TensorRtWeightStoreBuilder() = default;

  // The offset in the model file of `bytes`, if they are a view of it that
  // holds what the file holds.
  Expected<uint64_t> SourceOffset(absl::Span<const uint8_t> bytes) const;
  // Whether the file holds `bytes` at `offset`.
  bool EqualsSource(absl::Span<const uint8_t> bytes, uint64_t offset) const;
  void CloseOpenSegment();

  std::string source_path_;
  int source_fd_ = -1;
  uint64_t source_size_ = 0;
  TensorRtAotFileIdentity source_identity_;
  uint64_t granule_ = 0;
  uint64_t segment_bytes_ = 0;
  std::vector<Mapping> mappings_;
  std::vector<Segment> segments_;
  // The segment being filled and its fill, if any.
  bool has_open_segment_ = false;
  uint64_t fill_ = 0;
  // Launches by their members' (source offset, size).
  std::map<std::vector<std::pair<uint64_t, uint64_t>>, TensorRtWeightLocation>
      locations_;
};

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_WEIGHT_STORE_BUILDER_H_
