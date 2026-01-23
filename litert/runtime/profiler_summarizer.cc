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

#include "litert/runtime/profiler_summarizer.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tflite/core/api/profiler.h"
#include "tflite/core/interpreter.h"
#include "tflite/profiling/profile_buffer.h"

namespace litert {
namespace profiling {

namespace {

// Helper to update stats
void UpdateStat(OpStat& stat, int64_t time_us) {
  if (stat.count == 0) {
    stat.first_run_time_us = time_us;
    stat.min_time_us = time_us;
    stat.max_time_us = time_us;
  } else {
    stat.min_time_us = std::min(stat.min_time_us, time_us);
    stat.max_time_us = std::max(stat.max_time_us, time_us);
  }
  stat.total_time_us += time_us;
  stat.count++;
}

}  // namespace

LiteRtProfileSummarizer::LiteRtProfileSummarizer() = default;

void LiteRtProfileSummarizer::ProcessProfiles(
    const std::vector<const tflite::profiling::ProfileEvent*>& profile_stats,
    const tflite::Interpreter& interpreter) {
  for (auto event : profile_stats) {
    if (event->event_type ==
        tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      int node_index = event->event_metadata;
      int subgraph_index = event->extra_event_metadata;

      std::string type_in_stats(event->tag);
      const auto* node_and_reg =
          interpreter.node_and_registration(subgraph_index, node_index);
      if (node_and_reg) {
        const char* profiling_string = interpreter.OpProfilingString(
            node_and_reg->second, &node_and_reg->first);
        if (profiling_string) {
          type_in_stats += "/";
          type_in_stats += profiling_string;
        }
      }

      UpdateStat(stats_[type_in_stats], event->elapsed_time);

      // New node stats update
      std::pair<int, int> node_key = {subgraph_index, node_index};
      if (node_index_map_.find(node_key) == node_index_map_.end()) {
        ProfileNodeInfo node_info;
        node_info.node_type = type_in_stats;
        node_info.node_name = absl::StrFormat("[Unknown]:%d", node_index);
        node_info.first_start_time_us = event->begin_timestamp_us;

        node_stats_.push_back(node_info);
        node_index_map_[node_key] = node_stats_.size() - 1;
      }

      auto& node_stat = node_stats_[node_index_map_[node_key]];
      node_stat.total_time_us += event->elapsed_time;
      node_stat.count++;

    } else if (event->event_type ==
               tflite::Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      std::string op_name = event->tag;
      UpdateStat(delegate_stats_[op_name], event->elapsed_time);
    } else if (event->event_type ==
               tflite::Profiler::EventType::
                   DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT) {
      std::string op_name = "Delegate/" + event->tag;
      UpdateStat(delegate_stats_[op_name], event->elapsed_time);
    }
  }
}

std::string LiteRtProfileSummarizer::GetOutputString() const {
  std::stringstream ss;

  int64_t total_time_us = 0;
  for (const auto& node : node_stats_) {
    total_time_us += node.total_time_us;
  }
  double safe_total_time_us =
      total_time_us > 0 ? static_cast<double>(total_time_us) : 1.0;

  // 1. Run Order
  ss << "============================== Run Order "
        "==============================\n";
  ss << absl::StrFormat("%30s %10s %10s %10s %10s %10s %10s %10s\n",
                        "[node type]", "[first]", "[avg ms]", "[%]", "[cdf%]",
                        "[mem KB]", "[times called]", "[Name]");

  double cdf_us = 0;
  for (const auto& node : node_stats_) {
    cdf_us += node.total_time_us;
    double avg_ms =
        static_cast<double>(node.total_time_us) / node.count / 1000.0;
    double first_ms = static_cast<double>(node.first_start_time_us) / 1000.0;
    double pct =
        static_cast<double>(node.total_time_us) / safe_total_time_us * 100.0;
    double cdf_pct = cdf_us / safe_total_time_us * 100.0;

    ss << absl::StrFormat("%30s %10.3f %10.3f %9.3f%% %9.3f%% %10.3f %10d %s\n",
                          node.node_type, first_ms, avg_ms, pct, cdf_pct, 0.0,
                          node.count, node.node_name);
  }

  // 2. Top by Computation Time
  ss << "\n============================== Top by Computation Time "
        "==============================\n";
  ss << absl::StrFormat("%30s %10s %10s %10s %10s %10s %10s %10s\n",
                        "[node type]", "[first]", "[avg ms]", "[%]", "[cdf%]",
                        "[mem KB]", "[times called]", "[Name]");

  std::vector<ProfileNodeInfo> sorted_nodes = node_stats_;
  absl::c_stable_sort(sorted_nodes,
                   [](const ProfileNodeInfo& a, const ProfileNodeInfo& b) {
                     return a.total_time_us > b.total_time_us;
                   });

  cdf_us = 0;
  for (const auto& node : sorted_nodes) {
    cdf_us += node.total_time_us;
    double avg_ms =
        static_cast<double>(node.total_time_us) / node.count / 1000.0;
    double first_ms = static_cast<double>(node.first_start_time_us) / 1000.0;
    double pct =
        static_cast<double>(node.total_time_us) / safe_total_time_us * 100.0;
    double cdf_pct = cdf_us / safe_total_time_us * 100.0;

    ss << absl::StrFormat("%30s %10.3f %10.3f %9.3f%% %9.3f%% %10.3f %10d %s\n",
                          node.node_type, first_ms, avg_ms, pct, cdf_pct, 0.0,
                          node.count, node.node_name);
  }

  ss << "\nNumber of nodes executed: " << node_stats_.size() << "\n";

  // 3. Summary by node type
  ss << "============================== Summary by node type "
        "==============================\n";
  ss << absl::StrFormat("%30s %10s %10s %10s %10s %10s %10s\n", "[Node type]",
                        "[count]", "[avg ms]", "[avg %]", "[cdf %]", "[mem KB]",
                        "[times called]");

  std::vector<std::pair<std::string, OpStat>> sorted_stats(stats_.begin(),
                                                           stats_.end());
  absl::c_stable_sort(sorted_stats,
                   [](const auto& a, const auto& b) {
                     return a.second.total_time_us > b.second.total_time_us;
                   });

  cdf_us = 0;
  for (const auto& [name, stat] : sorted_stats) {
    cdf_us += stat.total_time_us;
    double avg_ms =
        static_cast<double>(stat.total_time_us) / stat.count / 1000.0;
    double pct =
        static_cast<double>(stat.total_time_us) / safe_total_time_us * 100.0;
    double cdf_pct = cdf_us / safe_total_time_us * 100.0;

    int node_count = 0;
    for (const auto& node : node_stats_) {
      if (node.node_type == name) node_count++;
    }

    ss << absl::StrFormat("%30s %10d %10.3f %9.3f%% %9.3f%% %10.3f %10d\n",
                          name, node_count, avg_ms, pct, cdf_pct, 0.0,
                          stat.count);
  }

  // 4. Timings
  ss << "\nTimings (microseconds): count=" << node_stats_.size()
     << " curr=" << total_time_us << "\n";
  ss << "Memory (bytes): count=0\n";
  ss << node_stats_.size() << " nodes observed\n";

  if (!delegate_stats_.empty()) {
    ss << "\nDelegate Statistics:\n";
    ss << absl::StrFormat("%-70s %10s %10s %10s %10s %10s\n", "Op Name",
                          "Count", "Avg(us)", "Min(us)", "Max(us)",
                          "Total(us)");
    for (const auto& [name, stat] : delegate_stats_) {
      double avg = static_cast<double>(stat.total_time_us) / stat.count;
      ss << absl::StrFormat("%-70s %10lld %10.2f %10lld %10lld %10lld\n", name,
                            stat.count, avg, stat.min_time_us, stat.max_time_us,
                            stat.total_time_us);
    }
  }

  return ss.str();
}

}  // namespace profiling
}  // namespace litert
