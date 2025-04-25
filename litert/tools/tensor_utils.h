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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_TENSOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_TENSOR_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl

namespace litert {
namespace tensor_utils {

// Structure to hold tensor statistics
template <typename T>
struct TensorStats {
  T min_val;
  T max_val;
  double avg;
  size_t zero_count;
  float zero_percent;
  size_t nan_count;  // Used for float types
  size_t inf_count;  // Used for float types
};

// Helper function to detect if a value is "zero"
// For integer types, we do exact comparison
// For floating point types, we use a threshold
template <typename T>
bool IsZero(T val) {
  if constexpr (std::is_floating_point<T>::value) {
    constexpr T threshold = static_cast<T>(1e-7);
    return std::abs(val) < threshold;
  } else {
    return val == static_cast<T>(0);
  }
}

// Calculate tensor statistics for numeric types
template <typename T>
TensorStats<T> CalculateTensorStats(const std::vector<T>& data,
                                    size_t total_elements) {
  TensorStats<T> stats;
  stats.min_val = std::numeric_limits<T>::max();
  stats.max_val = std::numeric_limits<T>::lowest();
  double sum = 0.0;
  stats.zero_count = 0;
  stats.nan_count = 0;
  stats.inf_count = 0;

  for (size_t i = 0; i < total_elements; ++i) {
    T val = data[i];
    if (IsZero(val)) stats.zero_count++;

    // Update min, max, sum
    stats.min_val = std::min(stats.min_val, val);
    stats.max_val = std::max(stats.max_val, val);
    sum += static_cast<double>(val);
  }

  stats.avg = (total_elements > 0) ? sum / total_elements : 0.0;
  stats.zero_percent = (total_elements > 0)
                           ? (100.0f * stats.zero_count / total_elements)
                           : 0.0f;

  return stats;
}

// Specialization for float type to handle NaN and Inf
template <>
inline TensorStats<float> CalculateTensorStats(const std::vector<float>& data,
                                               size_t total_elements) {
  TensorStats<float> stats;
  stats.min_val = std::numeric_limits<float>::max();
  stats.max_val = std::numeric_limits<float>::lowest();
  double sum = 0.0;
  stats.zero_count = 0;
  stats.nan_count = 0;
  stats.inf_count = 0;

  for (size_t i = 0; i < total_elements; ++i) {
    float val = data[i];
    if (IsZero(val)) stats.zero_count++;
    if (std::isnan(val)) stats.nan_count++;
    if (std::isinf(val)) stats.inf_count++;

    // Only update min, max, sum for valid numbers
    if (!std::isnan(val) && !std::isinf(val)) {
      stats.min_val = std::min(stats.min_val, val);
      stats.max_val = std::max(stats.max_val, val);
      sum += val;
    }
  }

  stats.avg = (total_elements > 0) ? sum / total_elements : 0.0f;
  stats.zero_percent = (total_elements > 0)
                           ? (100.0f * stats.zero_count / total_elements)
                           : 0.0f;

  return stats;
}

// Function to print tensor statistics
template <typename T>
void PrintTensorStats(const TensorStats<T>& stats) {
  ABSL_LOG(INFO) << "  Stats:";
  ABSL_LOG(INFO) << "    Min: " << stats.min_val;
  ABSL_LOG(INFO) << "    Max: " << stats.max_val;
  ABSL_LOG(INFO) << "    Avg: " << stats.avg;
  ABSL_LOG(INFO) << "    Zeros: " << stats.zero_count << " ("
                 << stats.zero_percent << "%)";
}

// Specialization for uint8_t
template <>
inline void PrintTensorStats(const TensorStats<uint8_t>& stats) {
  ABSL_LOG(INFO) << "  Stats:";
  ABSL_LOG(INFO) << "    Min: " << static_cast<int>(stats.min_val);
  ABSL_LOG(INFO) << "    Max: " << static_cast<int>(stats.max_val);
  ABSL_LOG(INFO) << "    Avg: " << stats.avg;
  ABSL_LOG(INFO) << "    Zeros: " << stats.zero_count << " ("
                 << stats.zero_percent << "%)";
}

// Specialization for float to handle NaN and Inf
template <>
inline void PrintTensorStats(const TensorStats<float>& stats) {
  ABSL_LOG(INFO) << "  Stats:";
  ABSL_LOG(INFO) << "    Min: " << stats.min_val;
  ABSL_LOG(INFO) << "    Max: " << stats.max_val;
  ABSL_LOG(INFO) << "    Avg: " << stats.avg;
  ABSL_LOG(INFO) << "    Zeros: " << stats.zero_count << " ("
                 << stats.zero_percent << "%)";
  if (stats.nan_count > 0) ABSL_LOG(INFO) << "    NaNs: " << stats.nan_count;
  if (stats.inf_count > 0) ABSL_LOG(INFO) << "    Infs: " << stats.inf_count;
}

// Function to print tensor samples from beginning, middle and end
template <typename T>
void PrintTensorSamples(const std::vector<T>& data, size_t total_elements,
                        size_t sample_size) {
  // Helper for printing elements
  auto print_element = [&](size_t index) {
    if constexpr (std::is_same<T, uint8_t>::value) {
      ABSL_LOG(INFO) << "    " << index << ": "
                     << static_cast<int>(data[index]);
    } else {
      ABSL_LOG(INFO) << "    " << index << ": " << data[index];
    }
  };

  // Beginning samples
  size_t begin_size = std::min(sample_size, total_elements);
  if (begin_size > 0) {
    ABSL_LOG(INFO) << "  Beginning " << begin_size << " elements:";
    for (size_t i = 0; i < begin_size; ++i) {
      print_element(i);
    }
  }

  // Middle samples
  if (total_elements > 2 * sample_size) {
    size_t middle_idx = total_elements / 2 - sample_size / 2;
    size_t middle_size = std::min(sample_size, total_elements - middle_idx);
    ABSL_LOG(INFO) << "  Middle " << middle_size << " elements (starting at "
                   << middle_idx << "):";
    for (size_t i = 0; i < middle_size; ++i) {
      print_element(middle_idx + i);
    }
  }

  // Ending samples
  if (total_elements > sample_size) {
    size_t end_idx = total_elements - std::min(sample_size, total_elements);
    ABSL_LOG(INFO) << "  Ending " << (total_elements - end_idx) << " elements:";
    for (size_t i = end_idx; i < total_elements; ++i) {
      print_element(i);
    }
  }
}

}  // namespace tensor_utils
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_TENSOR_UTILS_H_
