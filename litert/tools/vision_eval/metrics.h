// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_TOOLS_VISION_EVAL_METRICS_H_
#define LITERT_TOOLS_VISION_EVAL_METRICS_H_

#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <vector>

#include "litert/tools/vision_eval/data.h"

namespace litert::vision {
// Metric resize is FP32 half-pixel bilinear, without image antialiasing.
Expected<std::vector<float>> ResizePlane(const float* source, int height,
                                         int width, int output_height,
                                         int output_width);
Expected<OutputMetrics> Score(const OutputSpec& spec,
                              const std::vector<float>& prediction,
                              const TargetFile* target, int label,
                              const std::filesystem::path& dataset_root,
                              const std::filesystem::path& checkpoint);

// Keep one confusion matrix per output and paths to normals error spools;
// predictions and sample records are not retained.
class MetricAccumulator {
 public:
  Expected<void> Add(const OutputMetrics& row);
  Expected<OutputMetrics> Result(const std::filesystem::path& scratch) const;

 private:
  OutputMetrics total_;
  std::vector<FileIdentity> normal_errors_;
};
Expected<OutputMetrics> Aggregate(const std::vector<OutputMetrics>& rows,
                                  const std::filesystem::path& scratch);
Expected<double> ExactMedian(
    const std::vector<FileIdentity>& files,
    const std::filesystem::path& scratch,
    size_t chunk_values = 64u * 1024u * 1024u / sizeof(float));
}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_METRICS_H_
