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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_PROFILER_SUMMARIZER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_PROFILER_SUMMARIZER_H_

#include <map>
#include <string>
#include <vector>

#include "tflite/core/interpreter.h"
#include "tflite/profiling/profile_buffer.h"

namespace litert {
namespace profiling {

struct OpStat {
  int64_t count = 0;
  int64_t total_time_us = 0;
  int64_t min_time_us = -1;
  int64_t max_time_us = -1;
  int64_t first_run_time_us = 0;
};

struct ProfileNodeInfo {
  std::string node_type;
  std::string node_name;
  int64_t first_start_time_us = 0;
  int64_t total_time_us = 0;
  int count = 0;
};

// Creates a summary of operator invocations in the interpreter.
class LiteRtProfileSummarizer {
 public:
  LiteRtProfileSummarizer();
  ~LiteRtProfileSummarizer() = default;

  // Process profile events to update statistics for operator invocations.
  void ProcessProfiles(
      const std::vector<const tflite::profiling::ProfileEvent*>& profile_stats,
      const tflite::Interpreter& interpreter);

  // Returns a string detailing the accumulated runtime stats.
  std::string GetOutputString() const;

  const std::map<std::string, OpStat>& GetStats() const { return stats_; }
  const std::map<std::string, OpStat>& GetDelegateStats() const {
    return delegate_stats_;
  }

 private:
  std::map<std::string, OpStat> stats_;
  std::map<std::string, OpStat> delegate_stats_;
  std::vector<ProfileNodeInfo> node_stats_;
  std::map<std::pair<int, int>, size_t> node_index_map_;
};

}  // namespace profiling
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_PROFILER_SUMMARIZER_H_
