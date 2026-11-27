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

// Tracks the range of values seen for a single tensor.
//
// By default this records the global min and max of all values passed to
// Update(). It can optionally be switched into a second, histogram-based pass:
//
//   1. Call Update() over the calibration data to find the global [min, max].
//   2. Call InitializeHistogram(), which splits [min, max] (extended to include
//      0) into `num_bins` equal-width bins.
//   3. Call Update() over the calibration data again; values are now counted
//      into the histogram bins instead of updating min/max.
//   4. Get() returns a range clipped to the requested lower/upper percentiles
//      of the histogram, which reduces the effect of rare outliers.
class MinMax {
 public:
  // Upper limit for the number of histogram bins. Larger values are clamped to
  // this. Each tensor keeps one 64-bit counter per bin.
  static constexpr int kMaxHistogramBins = 1 << 20;

  TfLiteStatus Update(const float* values, size_t tensor_size,
                      ErrorReporter* error_reporter);

  bool HasValues() const { return has_values_; }

  // Switches to histogram accumulation using the current [min, max] range,
  // extended to include 0. `lower_percentile` and `upper_percentile` are in
  // [0, 100]. Invalid arguments are clamped: `num_bins` to
  // [1, kMaxHistogramBins], and the percentiles to [0, 100] with
  // `lower_percentile` <= `upper_percentile` (a NaN lower or upper percentile
  // is treated as 0 or 100). If the range is not finite, the tensor keeps
  // plain min/max tracking.
  void InitializeHistogram(int num_bins, double lower_percentile,
                           double upper_percentile);

  // Returns the calibrated range in `min_val` and `max_val`. Without a
  // histogram (or with an empty one) this is the min/max range. With a
  // histogram it is the percentile-clipped range. Once InitializeHistogram()
  // has been called, the returned range always includes 0. Returns
  // kTfLiteError if no values have been seen.
  TfLiteStatus Get(float* min_val, float* max_val) const;

 private:
  bool has_values_ = false;
  float min_ = std::numeric_limits<float>::max();
  float max_ = std::numeric_limits<float>::min();

  // State for the optional histogram pass.
  bool histogram_enabled_ = false;
  int num_bins_ = 0;
  float bin_width_ = 0.0f;
  double lower_percentile_ = 0.0;
  double upper_percentile_ = 100.0;
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

  // Switches every tensor logged so far into histogram accumulation mode. See
  // MinMax::InitializeHistogram(). Tensors first logged after this call keep
  // using plain min/max tracking.
  void InitializeHistograms(int num_bins, double lower_percentile,
                            double upper_percentile) {
    for (auto& [key, stats] : tensor_id_to_stats_map_) {
      stats.InitializeHistogram(num_bins, lower_percentile, upper_percentile);
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
