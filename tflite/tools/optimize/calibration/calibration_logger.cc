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
      min_ = std::min<float>(min_, value);
      max_ = std::max<float>(max_, value);
    } else {
      // Clamp before converting to int, so that values outside [min_, max_]
      // (including +/-inf) never overflow the cast. The clamp is done in double
      // because a float can't represent every int above 2^24, so a float upper
      // bound could round up to num_bins_ and index past the end of bins_.
      const float position = (value - min_) / bin_width_;
      const double clamped_position =
          std::clamp(static_cast<double>(position), 0.0,
                     static_cast<double>(num_bins_ - 1));
      ++bins_[static_cast<int>(clamped_position)];
      ++total_samples_;
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

void MinMax::InitializeHistogram(int num_bins, double lower_percentile,
                                 double upper_percentile) {
  // The quantized range must contain 0.
  min_ = std::min(min_, 0.0f);
  max_ = std::max(max_, 0.0f);

  const float range = max_ - min_;
  if (!std::isfinite(range)) {
    // A non-finite range (e.g. +/-inf seen in the first pass) can't be split
    // into bins, so keep using plain min/max for this tensor.
    TFLITE_LOG(TFLITE_LOG_WARNING,
               "Tensor range is not finite; skipping histogram calibration "
               "for this tensor.");
    return;
  }

  num_bins_ = std::clamp(num_bins, 1, kMaxHistogramBins);
  upper_percentile_ = std::clamp(upper_percentile, 0.0, 100.0);
  lower_percentile_ = std::clamp(lower_percentile, 0.0, upper_percentile_);

  // Avoid a zero bin width for constant or never-updated tensors.
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

  // Number of samples allowed below the lower cutoff and up to the upper
  // cutoff. The upper count is at least 1 so that an almost-empty histogram
  // doesn't stop at an empty first bin.
  const uint64_t lower_count =
      static_cast<uint64_t>((lower_percentile_ / 100.0) * total_samples_);
  const uint64_t upper_count = std::max<uint64_t>(
      1, static_cast<uint64_t>((upper_percentile_ / 100.0) * total_samples_));

  uint64_t accum_count = 0;
  int lower_bin = 0;
  bool found_lower_bin = false;
  int upper_bin = num_bins_ - 1;

  for (int b = 0; b < num_bins_; ++b) {
    accum_count += bins_[b];
    if (!found_lower_bin && accum_count >= lower_count) {
      lower_bin = b;
      found_lower_bin = true;
    }
    if (accum_count >= upper_count) {
      upper_bin = b;
      break;
    }
  }

  if (min_ >= 0.0f) {
    // Non-negative tensor (e.g. after ReLU): keep the lower bound at 0.
    *min_val = 0.0f;
  } else {
    // Signed tensor (e.g. SVDF): clip the lower tail as well.
    *min_val = min_ + lower_bin * bin_width_;
  }
  *max_val = min_ + (upper_bin + 1) * bin_width_;

  // Keep 0 inside the range, as required for TFLite quantization parameters.
  *min_val = std::min(*min_val, 0.0f);
  *max_val = std::max(*max_val, 0.0f);

  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
