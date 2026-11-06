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
#include "tflite/tools/optimize/calibration/calibration_logger.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tflite/c/c_api_types.h"
#include "tflite/core/api/error_reporter.h"
#include "tflite/logger.h"
#include "tflite/minimal_logging.h"

namespace tflite {
namespace optimize {
namespace calibration {

TfLiteStatus MinMax::Update(const float* values, size_t tensor_size,
                            ErrorReporter* error_reporter) {
  if (tensor_size <= 0) return kTfLiteOk;

  // TODO(shashishekhar): Make it possible to use weighted/moving average.
  bool has_nan_value = false;
  for (size_t i = 0; i < tensor_size; ++i) {
    const float value = values[i];
    if (std::isnan(value)) {
      has_nan_value = true;
      continue;
    }
    has_values_ = true;
    if (!histogram_enabled_) {
      // Pass 1: Global Range Discovery
      min_ = std::min<float>(min_, value);
      max_ = std::max<float>(max_, value);
    } else {
      // Pass 2: Fixed-Width Histogram Accumulation
      int bin = static_cast<int>((value - min_) / bin_width_);
      bin = std::max(0, std::min(bin, num_bins_ - 1));
      bins_[bin]++;
      total_samples_++;
    }
  }
  if (has_nan_value) {
    TFLITE_LOG(TFLITE_LOG_WARNING,
               "Model resulted in Nan value during calibration. Please "
               "consider making sure that model results in all real-values "
               "during inference with provided dataset.");
  }
  return kTfLiteOk;
}

void MinMax::InitializeHistogram(int num_bins, double lower_p, double upper_p) {
  num_bins_ = num_bins;
  lower_percentile_ = lower_p;
  upper_percentile_ = upper_p;

  // Guarantee 0.0 is included in range (TFLite requirement)
  min_ = std::min(min_, 0.0f);
  max_ = std::max(max_, 0.0f);

  float range = max_ - min_;
  // Division-by-zero guard for constant/uninitialized tensors (min == max)
  bin_width_ = (range < 1e-7f) ? 1.0f : (range / num_bins_);
  bins_.assign(num_bins_, 0);
  total_samples_ = 0;
  histogram_enabled_ = true;
}

TfLiteStatus MinMax::Get(float* min_val, float* max_val) const {
  if (!has_values_) return kTfLiteError;

  if (!histogram_enabled_ || total_samples_ == 0) {
    *min_val = min_;
    *max_val = max_;
    return kTfLiteOk;
  }

  // Percentile calculation
  uint64_t lower_count =
      static_cast<uint64_t>((lower_percentile_ / 100.0) * total_samples_);
  uint64_t upper_count =
      static_cast<uint64_t>((upper_percentile_ / 100.0) * total_samples_);

  uint64_t accum_count = 0;
  int lower_bin = 0;
  int upper_bin = num_bins_ - 1;

  for (int b = 0; b < num_bins_; ++b) {
    accum_count += bins_[b];
    if (accum_count >= lower_count && lower_bin == 0) {
      lower_bin = b;
    }
    if (accum_count >= upper_count) {
      upper_bin = b;
      break;
    }
  }

  // Tail Clipping Policy: Unsigned (ReLU) vs Signed (SVDF) tensors
  if (min_ >= 0.0f) {
    // Unsigned / Non-Negative tensor (post-ReLU): lock lower bound at <= 0.0
    *min_val = std::min(min_, 0.0f);
  } else {
    // Signed / Bipolar tensor (SVDF): apply lower-tail percentile cutoff
    *min_val = min_ + lower_bin * bin_width_;
  }

  *max_val = min_ + (upper_bin + 1) * bin_width_;

  // Ensure min <= 0 <= max holds for TFLite quantization params
  *min_val = std::min(*min_val, 0.0f);
  *max_val = std::max(*max_val, 0.0f);

  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
