// Copyright (c) 2025 MediaTek Inc.
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

#ifndef ODML_LITERT_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_
#define ODML_LITERT_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/mediatek/schema/neuron_schema_generated.h"

namespace neuron {

// Type aliases for Neuron SDK version components
using NeuronVersionMajor = uint32_t;
using NeuronVersionMinor = uint32_t;
using NeuronVersionPatch = uint32_t;
using NeuronVersionTuple =
    std::tuple<NeuronVersionMajor, NeuronVersionMinor, NeuronVersionPatch>;

inline bool IsNeuronSchema(const uint8_t* buffer, size_t size) {
  if (buffer == nullptr) {
    return false;
  }
  flatbuffers::Verifier verifier(buffer, size);
  return NeuronSchema::VerifyGraphsBuffer(verifier);
}

class CompiledGraph {
 public:
  CompiledGraph(const NeuronSchema::Graphs& g, const NeuronSchema::Subgraph& s)
      : graph_(g), subgraph_(s) {};

  litert::Expected<std::pair<const void*, size_t>> GetCompiledNetwork() {
    // Neuron Adapter doesn't support DLB for now.
    assert(GetCompiledType() != NeuronSchema::CompiledType_DLB);
    // TODO: Support the external buffer.
    assert(subgraph_.compiled_index_type() ==
           NeuronSchema::BufferIndicate_Index);
    auto index = subgraph_.compiled_index_as_Index();
    return GetBuffer(index->value());
  }

  NeuronSchema::CompiledType GetCompiledType() { return subgraph_.type(); }

  litert::Expected<std::pair<const void*, size_t>> GetBuffer(int32_t i) {
    auto array_size = graph_.data()->size();
    if (i < 0 || static_cast<size_t>(i) >= array_size) {
      return litert::Error(
          kLiteRtStatusErrorIndexOOB,
          absl::StrFormat("Buffer array index %d is OOB, the array size : %d",
                          i, array_size));
    }
    auto buffer = graph_.data()->Get(i);
    return std::pair<const void*, size_t>(buffer->data()->data(),
                                          buffer->data()->size());
  }

  // Returns the `(data, size)` of each buffer that holds externalized static
  // weights of this subgraph, in the order in which the compiled network
  // expects them as extra inputs. Returns an empty list if the subgraph has no
  // externalized weights.
  litert::Expected<std::vector<std::pair<const void*, size_t>>>
  GetWeightShareBuffers() {
    std::vector<std::pair<const void*, size_t>> buffers;
    const auto* types = subgraph_.weight_share_index_type();
    const auto* indices = subgraph_.weight_share_index();
    if (types == nullptr || indices == nullptr) {
      return buffers;
    }
    if (types->size() != indices->size()) {
      return litert::Error(
          kLiteRtStatusErrorInvalidFlatbuffer,
          "Mismatched weight_share_index_type and weight_share_index sizes");
    }
    buffers.reserve(indices->size());
    for (::flatbuffers::uoffset_t i = 0; i < indices->size(); ++i) {
      auto type = types->GetEnum<NeuronSchema::BufferIndicate>(i);
      if (type != NeuronSchema::BufferIndicate_Index) {
        return litert::Error(
            kLiteRtStatusErrorUnsupported,
            "Only BufferIndicate_Index is supported for weight_share_index");
      }
      auto buffer = GetBuffer(indices->GetAs<NeuronSchema::Index>(i)->value());
      if (!buffer) {
        return buffer.Error();
      }
      buffers.push_back(buffer.Value());
    }
    return buffers;
  }

 private:
  const NeuronSchema::Graphs& graph_;
  const NeuronSchema::Subgraph& subgraph_;
};

class SchemaResolver {
 public:
  SchemaResolver() = default;

  litert::Expected<bool> Initialize(const uint8_t* buffer, size_t size) {
    if (!IsNeuronSchema(buffer, size)) {
      return litert::Error(kLiteRtStatusErrorInvalidFlatbuffer,
                           "buffer is not a valid NeuronSchema");
    }
    graph_ = NeuronSchema::GetGraphs(buffer);

    auto subgraphs = graph_->subgraphs();
    for (const auto& subgraph : *subgraphs) {
      auto graph_name = subgraph->entry_point()->str();
      if (entry_points_.count(graph_name)) {
        // shouldn't have the same name between graphs.
        return false;
      } else {
        LITERT_LOG(LITERT_INFO, "Found graph: %s", graph_name.c_str());
        entry_points_[graph_name] = subgraph;
      }
    }
    LITERT_LOG(LITERT_INFO, "There are %u subgraphs in the bytecode",
               entry_points_.size());
    return true;
  }

  std::optional<CompiledGraph> GetCompiledGraph(const std::string& name) {
    if (entry_points_.count(name) == 0) {
      return std::nullopt;
    }
    return CompiledGraph(*graph_, *entry_points_[name]);
  };

  std::optional<NeuronVersionTuple> GetNeuronVersion() const {
    if (graph_ == nullptr) {
      return std::nullopt;
    }
    const auto* version = graph_->neuron_sdk_version();
    if (version == nullptr) {
      // Backward compatibility: old bytecode without version info
      return std::nullopt;
    }
    return NeuronVersionTuple{
        static_cast<NeuronVersionMajor>(version->major()),
        static_cast<NeuronVersionMinor>(version->minor()),
        static_cast<NeuronVersionPatch>(version->patch())};
  }

 private:
  const NeuronSchema::Graphs* graph_ = nullptr;

  std::unordered_map<std::string, NeuronSchema::Subgraph const*> entry_points_;
};

class BytecodeBuilder {
 public:
  BytecodeBuilder() = default;

  void SetNeuronVersion(NeuronVersionMajor major, NeuronVersionMinor minor,
                        NeuronVersionPatch patch) {
    neuron_version_ = NeuronSchema::NeuronVersion(major, minor, patch);
    has_neuron_version_ = true;
  }

  int32_t AddCompiledNetwork(
      const std::string& entry_point, NeuronSchema::CompiledType type,
      int32_t buffer_index,
      const std::vector<int32_t>& weight_share_indices = {}) {
    auto index = NeuronSchema::CreateIndex(fb_, buffer_index);
    ::flatbuffers::Offset<::flatbuffers::Vector<uint8_t>>
        weight_share_types_offset = 0;
    ::flatbuffers::Offset<::flatbuffers::Vector<::flatbuffers::Offset<void>>>
        weight_share_indices_offset = 0;
    if (!weight_share_indices.empty()) {
      std::vector<uint8_t> weight_share_types(
          weight_share_indices.size(), NeuronSchema::BufferIndicate_Index);
      std::vector<::flatbuffers::Offset<void>> weight_share_offsets;
      weight_share_offsets.reserve(weight_share_indices.size());
      for (int32_t weight_buffer_index : weight_share_indices) {
        weight_share_offsets.push_back(
            NeuronSchema::CreateIndex(fb_, weight_buffer_index).Union());
      }
      weight_share_types_offset = fb_.CreateVector(weight_share_types);
      weight_share_indices_offset = fb_.CreateVector(weight_share_offsets);
    }
    auto subgraph = NeuronSchema::CreateSubgraph(
        fb_, fb_.CreateString(entry_point), type,
        NeuronSchema::BufferIndicate_Index, index.Union(),
        weight_share_types_offset, weight_share_indices_offset);

    subgraphs_.push_back(subgraph);
    return subgraphs_count_++;
  };

  int32_t AddBuffer(const std::string& identifier,
                    const std::vector<int8_t>& data) {
    auto buffer =
        NeuronSchema::CreateBufferDirect(fb_, identifier.c_str(), &data);
    graph_data_.push_back(buffer);
    return buffer_count_++;
  }

  int32_t AddBuffer(const std::string& identifier, const int8_t* data,
                    size_t length) {
    return AddBufferWithData(identifier, fb_.CreateVector(data, length));
  }

  // Adds a buffer like `AddBuffer`, unless a buffer with identical contents
  // was already added through this method, in which case the index of that
  // buffer is returned and `identifier` is ignored.
  int32_t AddSharedWeightBuffer(const std::string& identifier,
                                const int8_t* data, size_t length) {
    const absl::string_view bytes(reinterpret_cast<const char*>(data), length);
    auto& candidates = shared_weight_buffers_[{
        length, absl::Hash<absl::string_view>{}(bytes)}];
    for (const SharedWeightBuffer& candidate : candidates) {
      const auto* stored =
          ::flatbuffers::GetTemporaryPointer(fb_, candidate.data);
      if (stored->size() == length &&
          (length == 0 || std::memcmp(stored->data(), data, length) == 0)) {
        return candidate.index;
      }
    }
    const auto data_offset = fb_.CreateVector(data, length);
    const int32_t index = AddBufferWithData(identifier, data_offset);
    candidates.push_back({index, data_offset});
    return index;
  }

  bool Finish() {
    auto graphs = NeuronSchema::CreateGraphsDirect(
        fb_, 1, &subgraphs_, &graph_data_, 0,
        has_neuron_version_ ? &neuron_version_ : nullptr);
    fb_.Finish(graphs);
    raw_buffer_ = {fb_.GetBufferPointer(), fb_.GetSize()};
    return true;
  }

  std::pair<uint8_t*, size_t> GetBytecode() {
    if (!raw_buffer_.has_value()) {
      return {nullptr, 0};
    }
    return raw_buffer_.value();
  }

 private:
  // A buffer added by `AddSharedWeightBuffer`.
  struct SharedWeightBuffer {
    int32_t index;
    ::flatbuffers::Offset<::flatbuffers::Vector<int8_t>> data;
  };

  int32_t AddBufferWithData(
      const std::string& identifier,
      ::flatbuffers::Offset<::flatbuffers::Vector<int8_t>> data_offset) {
    auto identifier_offset = fb_.CreateString(identifier);
    auto buffer =
        NeuronSchema::CreateBuffer(fb_, identifier_offset, data_offset);
    graph_data_.push_back(buffer);
    return buffer_count_++;
  }

  ::flatbuffers::FlatBufferBuilder fb_;

  std::optional<std::pair<uint8_t*, size_t>> raw_buffer_;

  std::vector<::flatbuffers::Offset<NeuronSchema::Subgraph>> subgraphs_;

  std::vector<::flatbuffers::Offset<NeuronSchema::Buffer>> graph_data_;

  // Buffers added by `AddSharedWeightBuffer`, keyed by length and content
  // hash. Candidates are compared byte by byte, so hash collisions are safe.
  absl::flat_hash_map<std::pair<size_t, size_t>,
                      std::vector<SharedWeightBuffer>>
      shared_weight_buffers_;

  int32_t subgraphs_count_ = 0;
  int32_t buffer_count_ = 0;
  NeuronSchema::NeuronVersion neuron_version_;
  bool has_neuron_version_ = false;
};

};  // namespace neuron

#endif  // ODML_LITERT_LITERT_VENDORS_MEDIATEK_SCHEMA_SCHEMA_RESOLVER_H_
