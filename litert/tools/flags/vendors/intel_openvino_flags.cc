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

#include "litert/tools/flags/vendors/intel_openvino_flags.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/tools/flags/options_parser_registry.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

ABSL_FLAG(LiteRtIntelOpenVinoPerformanceMode, intel_openvino_performance_mode,
          kLiteRtIntelOpenVinoPerformanceModeLatency,
          "Performance mode for Intel OpenVINO inference (latency, throughput, "
          "cumulative_throughput).");

ABSL_FLAG(std::string, intel_openvino_configs_map, "",
          "Configuration options for Intel OpenVINO as comma-separated "
          "key=value pairs "
          "(e.g., 'INFERENCE_PRECISION_HINT=f16,CACHE_DIR=/tmp/cache').");

ABSL_FLAG(std::string, intel_openvino_graph_backends, "",
          "Per-graph (per-partition) OpenVINO target device.  Accepts either "
          "a bare backend name to apply to all graphs (e.g. 'npu' or 'gpu'), "
          "or semicolon-separated 'GRAPH_INDEX:GRAPH_BACKEND' entries for "
          "per-partition selection (e.g. '0:npu;1:cpu;2:gpu').  Backend is "
          "one of cpu, gpu, npu.  Partitions without an entry default "
          "to npu.");

ABSL_FLAG(std::string, intel_openvino_graph_configs_map, "",
          "Per-graph OpenVINO config map overrides as semicolon-separated "
          "'GRAPH_INDEX:KEY=VALUE' entries (e.g., "
          "'1:INFERENCE_PRECISION_HINT=f32;1:CACHE_DIR=/tmp/ov_g1'). "
          "Entries are merged on top of --intel_openvino_configs_map for "
          "the indicated graph.");

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoGraphBackend* options,
                   std::string* error) {
  if (text == "cpu") {
    *options = kLiteRtIntelOpenVinoGraphBackendCPU;
    return true;
  }

  if (text == "gpu") {
    *options = kLiteRtIntelOpenVinoGraphBackendGPU;
    return true;
  }

  if (text == "npu") {
    *options = kLiteRtIntelOpenVinoGraphBackendNPU;
    return true;
  }

  *error = "Unknown Intel OpenVINO graph type";
  return false;
}

std::string AbslUnparseFlag(LiteRtIntelOpenVinoGraphBackend options) {
  switch (options) {
    case kLiteRtIntelOpenVinoGraphBackendCPU:
      return "cpu";
    case kLiteRtIntelOpenVinoGraphBackendGPU:
      return "gpu";
    case kLiteRtIntelOpenVinoGraphBackendNPU:
      return "npu";
    case kLiteRtIntelOpenVinoGraphBackendMax:
      break;
  }
  return "npu";
}

bool AbslParseFlag(absl::string_view text,
                   LiteRtIntelOpenVinoPerformanceMode* options,
                   std::string* error) {
  if (text == "latency") {
    *options = kLiteRtIntelOpenVinoPerformanceModeLatency;
    return true;
  }
  if (text == "throughput") {
    *options = kLiteRtIntelOpenVinoPerformanceModeThroughput;
    return true;
  }
  if (text == "cumulative_throughput") {
    *options = kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput;
    return true;
  }

  *error = "Unknown Intel OpenVINO performance mode";
  return false;
}

std::string AbslUnparseFlag(LiteRtIntelOpenVinoPerformanceMode options) {
  switch (options) {
    case kLiteRtIntelOpenVinoPerformanceModeLatency:
      return "latency";
    case kLiteRtIntelOpenVinoPerformanceModeThroughput:
      return "throughput";
    case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
      return "cumulative_throughput";
  }
}

// NOLINTEND(*alien-types*)

namespace litert::intel_openvino {

Expected<void> UpdateIntelOpenVinoOptionsFromFlags(
    IntelOpenVinoOptions& options) {
  options.SetPerformanceMode(
      absl::GetFlag(FLAGS_intel_openvino_performance_mode));

  // Parse configs map options from the flag (comma-separated key=value pairs)
  const std::string configs_map_str =
      absl::GetFlag(FLAGS_intel_openvino_configs_map);
  if (!configs_map_str.empty()) {
    std::vector<std::string> config_pairs =
        absl::StrSplit(configs_map_str, ',');
    for (const auto& pair : config_pairs) {
      // Split on the first '=' only so values may themselves contain '='
      // (e.g. NPU_COMPILATION_MODE_PARAMS=enable-flash-sdpa-conversion=true).
      const std::pair<std::string, std::string> key_value =
          absl::StrSplit(pair, absl::MaxSplits('=', 1));
      if (!key_value.first.empty() && absl::StrContains(pair, '=')) {
        options.SetConfigsMapOption(key_value.first.c_str(),
                                    key_value.second.c_str());
      } else {
        LITERT_LOG(
            LITERT_WARNING,
            "Ignoring malformed config pair: '%s'. Expected format: key=value",
            pair.c_str());
      }
    }
  }

  // Per-graph backend (target device) selection.
  //
  // Accepts either:
  //   * a bare backend name (e.g. "npu") -> applied to all graphs via the
  //     wildcard graph_index = -1, or
  //   * semicolon-separated "GRAPH_INDEX:GRAPH_BACKEND" entries (e.g.
  //     "0:npu;1:cpu") -> per-partition selection.
  const std::string graph_backends_str =
      absl::GetFlag(FLAGS_intel_openvino_graph_backends);
  if (!graph_backends_str.empty()) {
    // Bare-backend form: no ':' and no ';'.  Apply to all graphs.
    if (!absl::StrContains(graph_backends_str, ':') &&
        !absl::StrContains(graph_backends_str, ';')) {
      LiteRtIntelOpenVinoGraphBackend graph_backend;
      std::string parse_error;
      if (AbslParseFlag(graph_backends_str, &graph_backend, &parse_error)) {
        options.SetGraphBackend(/*graph_index=*/-1, graph_backend);
      } else {
        LITERT_LOG(LITERT_WARNING,
                   "Ignoring --intel_openvino_graph_backends='%s': %s",
                   graph_backends_str.c_str(), parse_error.c_str());
      }
    } else {
      std::vector<std::string> entries =
          absl::StrSplit(graph_backends_str, ';', absl::SkipEmpty());
      for (const auto& entry : entries) {
        std::vector<std::string> parts = absl::StrSplit(entry, ':');
        if (parts.size() != 2) {
          LITERT_LOG(LITERT_WARNING,
                     "Ignoring malformed graph backend entry: '%s'. "
                     "Expected format: GRAPH_INDEX:GRAPH_BACKEND",
                     entry.c_str());
          continue;
        }
        int graph_index = -1;
        if (!absl::SimpleAtoi(parts[0], &graph_index) || graph_index < 0) {
          LITERT_LOG(LITERT_WARNING,
                     "Ignoring graph backend entry with invalid index: '%s'",
                     entry.c_str());
          continue;
        }
        LiteRtIntelOpenVinoGraphBackend graph_backend;
        std::string parse_error;
        if (!AbslParseFlag(parts[1], &graph_backend, &parse_error)) {
          LITERT_LOG(LITERT_WARNING,
                     "Ignoring graph backend entry with invalid type '%s': %s",
                     entry.c_str(), parse_error.c_str());
          continue;
        }
        options.SetGraphBackend(graph_index, graph_backend);
      }
    }
  }

  // Per-graph configs map overrides: "GRAPH_INDEX:KEY=VALUE;..."
  const std::string graph_configs_str =
      absl::GetFlag(FLAGS_intel_openvino_graph_configs_map);
  if (!graph_configs_str.empty()) {
    std::vector<std::string> entries =
        absl::StrSplit(graph_configs_str, ';', absl::SkipEmpty());
    for (const auto& entry : entries) {
      // Split on the first ':' to separate the index from the KEY=VALUE.
      auto colon = entry.find(':');
      if (colon == std::string::npos) {
        LITERT_LOG(LITERT_WARNING,
                   "Ignoring malformed graph config entry: '%s'. "
                   "Expected format: GRAPH_INDEX:KEY=VALUE",
                   entry.c_str());
        continue;
      }
      int graph_index = -1;
      if (!absl::SimpleAtoi(entry.substr(0, colon), &graph_index) ||
          graph_index < 0) {
        LITERT_LOG(LITERT_WARNING,
                   "Ignoring graph config entry with invalid index: '%s'",
                   entry.c_str());
        continue;
      }
      absl::string_view kv = absl::string_view(entry).substr(colon + 1);
      auto eq = kv.find('=');
      if (eq == absl::string_view::npos) {
        LITERT_LOG(LITERT_WARNING,
                   "Ignoring malformed graph config entry: '%s'. "
                   "Expected format: GRAPH_INDEX:KEY=VALUE",
                   entry.c_str());
        continue;
      }
      std::string key(kv.substr(0, eq));
      std::string value(kv.substr(eq + 1));
      options.SetGraphConfigsMapOption(graph_index, key.c_str(), value.c_str());
    }
  }

  return {};
}

}  // namespace litert::intel_openvino

namespace litert::intel_openvino {

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  LITERT_ASSIGN_OR_RETURN(auto& intel_openvino_opts,
                          options.GetIntelOpenVinoOptions());
  return UpdateIntelOpenVinoOptionsFromFlags(intel_openvino_opts);
});

}  // namespace litert::intel_openvino
