/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/core/c/common.h"

namespace tflite {
namespace optimize {
namespace calibration {

class MinMax {
 public:
  TfLiteStatus Update(const float* values, size_t tensor_size,
                      ErrorReporter* error_reporter);

  bool HasValues() const { return has_values_; }

  void InitializeHistogram(int num_bins = 1024, double lower_p = 0.05,
                           double upper_p = 99.95);

  TfLiteStatus Get(float* min_val, float* max_val) const;

 private:
  bool has_values_ = false;
  float min_ = std::numeric_limits<float>::max();
  float max_ = std::numeric_limits<float>::lowest();

  // Histogram fields for two-pass percentile calibration
  bool histogram_enabled_ = false;
  int num_bins_ = 1024;
  float bin_width_ = 0.0f;
  double lower_percentile_ = 0.05;
  double upper_percentile_ = 99.95;
  std::vector<uint64_t> bins_;
  uint64_t total_samples_ = 0;
};

// Captures min max values for tensors.
class Logger {
 public:
  // Log the value for tensor at |tensor_index| which has |tensor_values|
  TfLiteStatus LogTensorValue(int subgraph_index, int tensor_index,
                              const float* tensor_values, size_t tensor_size,
                              ErrorReporter* error_reporter) {
    std::tuple<int, int> key{subgraph_index, tensor_index};
    return tensor_id_to_stats_map_[key].Update(tensor_values, tensor_size,
                                               error_reporter);
  }

  // Initializes histograms across all logged tensors for Pass 2 percentile
  // calibration.
  void InitializeHistograms(int num_bins = 1024, double lower_p = 0.05,
                            double upper_p = 99.95) {
    for (auto& entry : tensor_id_to_stats_map_) {
      entry.second.InitializeHistogram(num_bins, lower_p, upper_p);
    }
  }

  // Returns a map from tensor_index -> observed min max values.
  const absl::flat_hash_map<std::tuple<int, int>, MinMax>&
  GetCalibrationValues() const {
    return tensor_id_to_stats_map_;
  }

 private:
  absl::flat_hash_map<std::tuple<int, int>, MinMax> tensor_id_to_stats_map_;
};

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATION_LOGGER_H_
