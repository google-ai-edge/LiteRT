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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_pattern.h"
#include "tflite/schema/schema_generated.h"

// Rewrites a TFLite model: every region that matches a pattern becomes one
// STABLEHLO_COMPOSITE op whose decomposition subgraph holds the region's ops.
//
// The regions are found on LiteRT's model (the matcher of carve_pattern.h), and
// the file is rewritten on the TFLite schema's own objects, because the LiteRT
// model does not carry everything a model file holds (it drops, for one, the
// block shape of blockwise quantization, without which centered INT2 weights no
// longer run): what the rewrite does not touch is written back as it was read.
namespace litert::tools {

// The tensors a tensor's quantization names: the scales and zero points of
// blockwise and multi-axis quantization, as indices into the tensors of the
// same subgraph. An index of -1 is "none".
std::vector<int32_t*> QuantizationTensors(tflite::TensorT& tensor);

// A region of a subgraph in terms of the model file's indices.
struct CarveRegionSpec {
  std::string composite_name;
  // The operators of the region, ascending.
  std::vector<int32_t> ops;
  // The tensors that become the composite's inputs and outputs, in order.
  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  // The composite's attributes, a flexbuffer map.
  std::vector<uint8_t> attributes;
};

// Replaces the regions of subgraph `subgraph_index`, which must not overlap,
// by composites. The composite takes the place of its region's last operator;
// its inputs stay tensors of the subgraph (a constant among them stays a
// constant there and is an input of the decomposition), every other constant
// the region uses is copied into the decomposition, and the tensors that only
// the region used leave the subgraph.
absl::Status OutlineRegions(tflite::ModelT& model, size_t subgraph_index,
                            const std::vector<CarveRegionSpec>& regions);

// What was found in one subgraph for one pattern.
struct CarveReportEntry {
  struct NearMiss {
    std::string reason;
    // The number of places with this reason, and the first of them.
    size_t count = 0;
    std::string first_place;
  };
  size_t subgraph_index = 0;
  std::string subgraph_name;
  std::string pattern;
  size_t num_carved = 0;
  std::vector<NearMiss> near_misses;
};

struct CarveReport {
  // The pairs of subgraph and pattern with a region or a near miss.
  std::vector<CarveReportEntry> entries;
  size_t num_subgraphs = 0;
  // Decomposition subgraphs are not searched: a backend that takes a
  // composite does not look inside, and a carved model carved again must not
  // change.
  size_t num_decompositions_skipped = 0;
  size_t num_carved = 0;
};

// The bytes of a rewritten model: the new flatbuffer, zeros up to the place of
// the constant data, and the constant data the source keeps behind its
// flatbuffer, which stays where it is (a view of the source) until the model
// is written.
struct CarvedModel {
  litert::OwningBufferRef<uint8_t> flatbuffer;
  uint64_t gap = 0;
  litert::BufferRef<uint8_t> appended;

  uint64_t Size() const { return flatbuffer.Size() + gap + appended.Size(); }
};

// Carves the model in `source`. Returns nothing when no region matched: the
// model is then left as it is, byte for byte.
absl::StatusOr<std::optional<CarvedModel>> CarveModel(
    litert::BufferRef<uint8_t> source, absl::Span<const CarvePattern> patterns,
    CarveReport* report);

// Serializes `model`, an edited copy of the model in `source` with the same
// buffers. The offsets of the model's buffers move with the data behind the
// new flatbuffer.
absl::StatusOr<CarvedModel> SerializeCarvedModel(
    tflite::ModelT& model, litert::BufferRef<uint8_t> source);

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_TOOLS_CARVE_MODEL_H_
