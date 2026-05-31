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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/c/options/litert_intel_openvino_options.h"

#include <stdlib.h>

#include <cstring>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/litert_toml_parser.h"

struct LrtIntelOpenVinoOptionsT {
  std::optional<LiteRtIntelOpenVinoPerformanceMode> performance_mode;
  // Store custom configuration options as key-value pairs
  std::vector<std::pair<std::string, std::string>> configs_map_options;

  // Per-graph (per-partition) overrides.  Ordered so that enumeration and
  // serialization produce a stable output.  A graph appears in the map if it
  // has either a graph_backend override or at least one configs map entry.
  struct GraphOverride {
    std::optional<LiteRtIntelOpenVinoGraphBackend> graph_backend;
    std::vector<std::pair<std::string, std::string>> configs_map_options;
  };
  std::map<int, GraphOverride> graph_overrides;
};

LiteRtStatus LrtIntelOpenVinoOptionsCreate(LrtIntelOpenVinoOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtIntelOpenVinoOptionsT{};
  return kLiteRtStatusOk;
}

void LrtDestroyIntelOpenVinoOptions(LrtIntelOpenVinoOptions options) {
  delete options;
}

static const char kIdentifier[] = "intel_openvino";

LiteRtStatus LrtGetOpaqueIntelOpenVinoOptionsData(
    LrtIntelOpenVinoOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*)) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::stringstream ss;
  if (options->performance_mode.has_value()) {
    ss << "performance_mode = " << options->performance_mode.value() << "\n";
  }
  for (const auto& pair : options->configs_map_options) {
    ss << "configs_map." << pair.first << " = \"" << pair.second << "\"\n";
  }
  for (const auto& [graph_index, override] : options->graph_overrides) {
    if (override.graph_backend.has_value()) {
      ss << "graph." << graph_index
         << ".graph_backend = " << override.graph_backend.value() << "\n";
    }
    for (const auto& pair : override.configs_map_options) {
      ss << "graph." << graph_index << ".configs_map." << pair.first << " = \""
         << pair.second << "\"\n";
    }
  }

  *identifier = LrtGetIntelOpenVinoOptionsIdentifier();
  std::string toml_str = ss.str();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);

  return kLiteRtStatusOk;
}

const char* LrtGetIntelOpenVinoOptionsIdentifier() { return kIdentifier; }

LiteRtStatus LrtCreateIntelOpenVinoOptionsFromToml(
    const char* payload, LrtIntelOpenVinoOptions* options) {
  if (options == nullptr || payload == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LrtIntelOpenVinoOptions local_options = nullptr;
  LITERT_RETURN_IF_ERROR(LrtIntelOpenVinoOptionsCreate(&local_options));

  absl::string_view payload_sv(payload);

  auto status = litert::internal::ParseToml(
      payload_sv,
      [local_options](absl::string_view key,
                      absl::string_view value) -> LiteRtStatus {
        if (key == "performance_mode") {
          auto parsed_int = litert::internal::ParseTomlInt(value);
          if (!parsed_int.HasValue()) {
            return litert::ToLiteRtStatus(parsed_int.Error().StatusCC());
          }
          return LrtIntelOpenVinoOptionsSetPerformanceMode(
              local_options,
              static_cast<LiteRtIntelOpenVinoPerformanceMode>(*parsed_int));
        } else if (absl::StartsWith(key, "configs_map.")) {
          std::string config_key(key.substr(12));
          std::string config_value(value);
          return LrtIntelOpenVinoOptionsSetConfigsMapOption(
              local_options, config_key.c_str(), config_value.c_str());
        } else if (absl::StartsWith(key, "graph.")) {
          // graph.<index>.graph_backend = N
          // graph.<index>.configs_map.<KEY> = "VALUE"
          absl::string_view rest = key.substr(6);
          auto dot_pos = rest.find('.');
          if (dot_pos == absl::string_view::npos) {
            return kLiteRtStatusErrorInvalidArgument;
          }
          int graph_index;
          if (!absl::SimpleAtoi(rest.substr(0, dot_pos), &graph_index)) {
            return kLiteRtStatusErrorInvalidArgument;
          }
          absl::string_view sub_key = rest.substr(dot_pos + 1);
          if (sub_key == "graph_backend") {
            auto parsed_int = litert::internal::ParseTomlInt(value);
            if (!parsed_int.HasValue()) {
              return litert::ToLiteRtStatus(parsed_int.Error().StatusCC());
            }
            return LrtIntelOpenVinoOptionsSetGraphBackend(
                local_options, graph_index,
                static_cast<LiteRtIntelOpenVinoGraphBackend>(*parsed_int));
          } else if (absl::StartsWith(sub_key, "configs_map.")) {
            std::string config_key(sub_key.substr(12));
            std::string config_value(value);
            return LrtIntelOpenVinoOptionsSetGraphConfigsMapOption(
                local_options, graph_index, config_key.c_str(),
                config_value.c_str());
          }
          return kLiteRtStatusOk;
        }
        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyIntelOpenVinoOptions(local_options);
    return status;
  }

  *options = local_options;
  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// performance_mode -----------------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->performance_mode = performance_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode* performance_mode) {
  if (options == nullptr || performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->performance_mode.has_value()) {
    *performance_mode = kLiteRtIntelOpenVinoPerformanceModeLatency;
  } else {
    *performance_mode = options->performance_mode.value();
  }
  return kLiteRtStatusOk;
}

// configs_map_options --------------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetConfigsMapOption(
    LrtIntelOpenVinoOptions options, const char* key, const char* value) {
  if (options == nullptr || key == nullptr || value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Check if the key already exists and update it, otherwise add new
  for (auto& pair : options->configs_map_options) {
    if (pair.first == key) {
      pair.second = value;
      return kLiteRtStatusOk;
    }
  }
  options->configs_map_options.emplace_back(key, value);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int* num_options) {
  if (options == nullptr || num_options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_options = static_cast<int>(options->configs_map_options.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetConfigsMapOption(
    LrtIntelOpenVinoOptions options, int index, const char** key,
    const char** value) {
  if (options == nullptr || key == nullptr || value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (index < 0 ||
      index >= static_cast<int>(options->configs_map_options.size())) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& pair = options->configs_map_options[index];
  *key = pair.first.c_str();
  *value = pair.second.c_str();
  return kLiteRtStatusOk;
}

// per-graph backend overrides ------------------------------------------------
LiteRtStatus LrtIntelOpenVinoOptionsSetGraphBackend(
    LrtIntelOpenVinoOptions options, int graph_index,
    LiteRtIntelOpenVinoGraphBackend graph_backend) {
  // graph_index == -1 is a wildcard meaning "default for all graphs that do
  // not have an explicit per-index override".
  if (options == nullptr || graph_index < -1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->graph_overrides[graph_index].graph_backend = graph_backend;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetGraphBackend(
    LrtIntelOpenVinoOptions options, int graph_index,
    LiteRtIntelOpenVinoGraphBackend* graph_backend) {
  if (options == nullptr || graph_backend == nullptr || graph_index < -1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Prefer the explicit per-index override.
  auto it = options->graph_overrides.find(graph_index);
  if (it != options->graph_overrides.end() &&
      it->second.graph_backend.has_value()) {
    *graph_backend = *it->second.graph_backend;
    return kLiteRtStatusOk;
  }
  // Fall back to the wildcard (graph_index = -1) entry if present.
  if (graph_index != -1) {
    auto wildcard = options->graph_overrides.find(-1);
    if (wildcard != options->graph_overrides.end() &&
        wildcard->second.graph_backend.has_value()) {
      *graph_backend = *wildcard->second.graph_backend;
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorNotFound;
}

LiteRtStatus LrtIntelOpenVinoOptionsSetGraphConfigsMapOption(
    LrtIntelOpenVinoOptions options, int graph_index, const char* key,
    const char* value) {
  if (options == nullptr || key == nullptr || value == nullptr ||
      graph_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& override = options->graph_overrides[graph_index];
  for (auto& pair : override.configs_map_options) {
    if (pair.first == key) {
      pair.second = value;
      return kLiteRtStatusOk;
    }
  }
  override.configs_map_options.emplace_back(key, value);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetNumGraphOverrides(
    LrtIntelOpenVinoOptions options, int* num_overrides) {
  if (options == nullptr || num_overrides == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_overrides = static_cast<int>(options->graph_overrides.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetGraphOverrideIndex(
    LrtIntelOpenVinoOptions options, int slot_index, int* graph_index) {
  if (options == nullptr || graph_index == nullptr || slot_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (slot_index >= static_cast<int>(options->graph_overrides.size())) {
    return kLiteRtStatusErrorIndexOOB;
  }
  auto it = options->graph_overrides.begin();
  std::advance(it, slot_index);
  *graph_index = it->first;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetNumGraphConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int graph_index, int* num_options) {
  if (options == nullptr || num_options == nullptr || graph_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->graph_overrides.find(graph_index);
  if (it == options->graph_overrides.end()) {
    *num_options = 0;
    return kLiteRtStatusOk;
  }
  *num_options = static_cast<int>(it->second.configs_map_options.size());
  return kLiteRtStatusOk;
}

LiteRtStatus LrtIntelOpenVinoOptionsGetGraphConfigsMapOption(
    LrtIntelOpenVinoOptions options, int graph_index, int index,
    const char** key, const char** value) {
  if (options == nullptr || key == nullptr || value == nullptr ||
      graph_index < 0 || index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto it = options->graph_overrides.find(graph_index);
  if (it == options->graph_overrides.end() ||
      index >= static_cast<int>(it->second.configs_map_options.size())) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& pair = it->second.configs_map_options[index];
  *key = pair.first.c_str();
  *value = pair.second.c_str();
  return kLiteRtStatusOk;
}
