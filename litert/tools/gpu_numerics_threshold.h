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

#ifndef ODML_LITERT_LITERT_TOOLS_GPU_NUMERICS_THRESHOLD_H_
#define ODML_LITERT_LITERT_TOOLS_GPU_NUMERICS_THRESHOLD_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl

namespace litert {

struct BufferDiffStats {
  // Index of output buffer.
  size_t buffer_idx;
  // Total number of elements in the buffer.
  size_t total_elements;
  // Number of elements with absolute difference greater than epsilon.
  size_t diff_elements;
  // Epsilon value used for comparison.
  double epsilon;
  // Maximum absolute difference between CPU and GPU values.
  double max_diff;
  // Minimum absolute difference between CPU and GPU values.
  double min_diff;
  // Mean absolute difference between CPU and GPU values.
  double mean_diff;
  // Mean squared error between CPU and GPU values.
  double mse;
};

struct DiffThresholdConfig {
  bool fail_on_threshold = true;
  double max_abs_diff_threshold = 1e-4;
  double mean_abs_diff_threshold = 1e-6;
  double diff_ratio_threshold = 1e-6;
};

struct DiffThresholdEvaluation {
  bool has_violation = false;
  bool should_fail = false;
  std::vector<std::string> violation_messages;
};

double ComputeDiffRatio(const BufferDiffStats& stats);

DiffThresholdEvaluation EvaluateDiffThresholds(
    absl::Span<const BufferDiffStats> stats,
    const DiffThresholdConfig& thresholds);

}  // namespace litert

#endif  // ODML_LITERT_LITERT_TOOLS_GPU_NUMERICS_THRESHOLD_H_
