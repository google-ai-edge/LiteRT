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

#include "litert/tools/gpu_numerics_threshold.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert {

double ComputeDiffRatio(const BufferDiffStats& stats) {
  if (stats.total_elements == 0) {
    return 0.0;
  }
  return static_cast<double>(stats.diff_elements) /
         static_cast<double>(stats.total_elements);
}

DiffThresholdEvaluation EvaluateDiffThresholds(
    absl::Span<const BufferDiffStats> stats,
    const DiffThresholdConfig& thresholds) {
  std::vector<std::string> violations;
  for (const auto& stat : stats) {
    if (stat.max_diff > thresholds.max_abs_diff_threshold) {
      violations.push_back(absl::StrFormat(
          "Output #%d max_abs_diff %.9g exceeds threshold %.9g",
          stat.buffer_idx, stat.max_diff, thresholds.max_abs_diff_threshold));
    }
    if (stat.mean_diff > thresholds.mean_abs_diff_threshold) {
      violations.push_back(absl::StrFormat(
          "Output #%d mean_abs_diff %.9g exceeds threshold %.9g",
          stat.buffer_idx, stat.mean_diff, thresholds.mean_abs_diff_threshold));
    }
    const double diff_ratio = ComputeDiffRatio(stat);
    if (diff_ratio > thresholds.diff_ratio_threshold) {
      violations.push_back(absl::StrFormat(
          "Output #%d diff_ratio %.9g exceeds threshold %.9g", stat.buffer_idx,
          diff_ratio, thresholds.diff_ratio_threshold));
    }
  }

  DiffThresholdEvaluation result;
  result.has_violation = !violations.empty();
  result.should_fail = result.has_violation && thresholds.fail_on_threshold;
  result.violation_messages = std::move(violations);
  return result;
}

}  // namespace litert
