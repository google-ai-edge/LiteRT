// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_

#include <memory>
#include <string>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::intel_openvino {

/// @brief Defines the C++ wrapper for Intel OpenVINO-specific LiteRT options.
class IntelOpenVinoOptions {
 public:
  IntelOpenVinoOptions() = delete;

  // Non-copyable, only movable
  IntelOpenVinoOptions(const IntelOpenVinoOptions&) = delete;
  IntelOpenVinoOptions& operator=(const IntelOpenVinoOptions&) = delete;
  IntelOpenVinoOptions(IntelOpenVinoOptions&&) = default;
  IntelOpenVinoOptions& operator=(IntelOpenVinoOptions&&) = default;

  static Expected<IntelOpenVinoOptions> Create() {
    LrtIntelOpenVinoOptions options;
    LITERT_RETURN_IF_ERROR(LrtIntelOpenVinoOptionsCreate(&options));
    return IntelOpenVinoOptions(options);
  }

  static IntelOpenVinoOptions CreateFromOwnedHandle(
      LrtIntelOpenVinoOptions options) {
    return IntelOpenVinoOptions(options);
  }

  void SetPerformanceMode(LiteRtIntelOpenVinoPerformanceMode performance_mode) {
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsSetPerformanceMode(Get(), performance_mode));
  }

  LiteRtIntelOpenVinoPerformanceMode GetPerformanceMode() const {
    LiteRtIntelOpenVinoPerformanceMode performance_mode;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetPerformanceMode(Get(), &performance_mode));
    return performance_mode;
  }

  void SetConfigsMapOption(const char* key, const char* value) {
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsSetConfigsMapOption(Get(), key, value));
  }

  int GetNumConfigsMapOptions() const {
    int num_options;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(Get(), &num_options));
    return num_options;
  }

  std::pair<std::string, std::string> GetConfigsMapOption(int index) const {
    const char* key = nullptr;
    const char* value = nullptr;
    auto status =
        LrtIntelOpenVinoOptionsGetConfigsMapOption(Get(), index, &key, &value);

    if (status != kLiteRtStatusOk) {
      return std::make_pair(std::string(), std::string());
    }

    return std::make_pair(std::string(key), std::string(value));
  }

  // Per-graph (partition) backend overrides ----------------------------------

  // Set the OpenVINO graph type (target device) for a specific graph
  // (partition) index.  Each partition in the compiled model targets the
  // device configured here.
  void SetGraphBackend(int graph_index,
                    LiteRtIntelOpenVinoGraphBackend graph_backend) {
    LITERT_ABORT_IF_ERROR(LrtIntelOpenVinoOptionsSetGraphBackend(
        Get(), graph_index, graph_backend));
  }

  // Returns the graph type configured for `graph_index` if one is set,
  // otherwise an error with `kLiteRtStatusErrorNotFound`.
  Expected<LiteRtIntelOpenVinoGraphBackend> GetGraphBackend(
      int graph_index) const {
    LiteRtIntelOpenVinoGraphBackend graph_backend;
    LITERT_RETURN_IF_ERROR(LrtIntelOpenVinoOptionsGetGraphBackend(
        Get(), graph_index, &graph_backend));
    return graph_backend;
  }

  // Set a per-graph OpenVINO config map override.  Merged on top of the
  // model-wide configs_map at compile time.
  void SetGraphConfigsMapOption(int graph_index, const char* key,
                                const char* value) {
    LITERT_ABORT_IF_ERROR(LrtIntelOpenVinoOptionsSetGraphConfigsMapOption(
        Get(), graph_index, key, value));
  }

  // Total number of graphs that have at least one override set.
  int GetNumGraphOverrides() const {
    int num_overrides = 0;
    LITERT_ABORT_IF_ERROR(
        LrtIntelOpenVinoOptionsGetNumGraphOverrides(Get(), &num_overrides));
    return num_overrides;
  }

  // Returns the graph index at the given enumeration slot.
  int GetGraphOverrideIndex(int slot_index) const {
    int graph_index = 0;
    LITERT_ABORT_IF_ERROR(LrtIntelOpenVinoOptionsGetGraphOverrideIndex(
        Get(), slot_index, &graph_index));
    return graph_index;
  }

  int GetNumGraphConfigsMapOptions(int graph_index) const {
    int num_options = 0;
    LITERT_ABORT_IF_ERROR(LrtIntelOpenVinoOptionsGetNumGraphConfigsMapOptions(
        Get(), graph_index, &num_options));
    return num_options;
  }

  std::pair<std::string, std::string> GetGraphConfigsMapOption(
      int graph_index, int index) const {
    const char* key = nullptr;
    const char* value = nullptr;
    auto status = LrtIntelOpenVinoOptionsGetGraphConfigsMapOption(
        Get(), graph_index, index, &key, &value);
    if (status != kLiteRtStatusOk) {
      return std::make_pair(std::string(), std::string());
    }
    return std::make_pair(std::string(key), std::string(value));
  }

  LrtIntelOpenVinoOptions Get() const { return options_.get(); }

 private:
  explicit IntelOpenVinoOptions(LrtIntelOpenVinoOptions options)
      : options_(options) {}

  struct IntelOpenVinoOptionsDeleter {
    void operator()(LrtIntelOpenVinoOptions options) const {
      LrtDestroyIntelOpenVinoOptions(options);
    }
  };

  std::unique_ptr<LrtIntelOpenVinoOptionsT, IntelOpenVinoOptionsDeleter>
      options_;
};

}  // namespace litert::intel_openvino

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
