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
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <numeric>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"

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

inline Expected<std::vector<char>> ReadTensorDataFromRawFile(
    absl::string_view file_path) {
  const std::string path_str(file_path);
  std::error_code ec;
  if (!std::filesystem::exists(path_str, ec) ||
      std::filesystem::is_directory(path_str, ec)) {
    return Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Failed to find input file %s.", file_path));
  }
  const auto file_size = std::filesystem::file_size(path_str, ec);
  if (ec) {
    return Unexpected(kLiteRtStatusErrorNotFound,
                      absl::StrFormat("Failed to get file size for %s: %s",
                                      file_path, ec.message()));
  }
  std::ifstream file(path_str, std::ios::binary);
  if (!file.is_open()) {
    return Unexpected(
        kLiteRtStatusErrorNotFound,
        absl::StrFormat("Failed to open input file %s.", file_path));
  }
  std::vector<char> input_data(file_size);
  if (file_size > 0) {
    file.read(input_data.data(), file_size);
    if (!file) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrFormat("Failed to read all bytes from %s.", file_path));
    }
  }
  return input_data;
}

template <typename T>
inline Expected<void> WriteBufferAs(TensorBuffer& buffer,
                                    const std::vector<char>& data) {
  return buffer.Write<T>(absl::MakeConstSpan(
      reinterpret_cast<const T*>(data.data()), data.size() / sizeof(T)));
}

// Fill tensor buffer with custom data
inline Expected<void> FillBufferWithCustomData(TensorBuffer& buffer,
                                               const std::vector<char>& data) {
  LITERT_ASSIGN_OR_RETURN(const auto buffer_size, buffer.Size());
  if (data.size() != buffer_size) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("Mismatched input size, input data size: %d bytes != "
                        "model buffer size: %d bytes.",
                        data.size(), buffer_size));
  }
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());

  switch (type.ElementType()) {
    case ElementType::Float32:
      return WriteBufferAs<float>(buffer, data);
    case ElementType::Int64:
      return WriteBufferAs<int64_t>(buffer, data);
    case ElementType::Int32:
      return WriteBufferAs<int32_t>(buffer, data);
    case ElementType::Int16:
      return WriteBufferAs<int16_t>(buffer, data);
    case ElementType::Int8:
      return WriteBufferAs<int8_t>(buffer, data);
    case ElementType::UInt8:
    case ElementType::Bool:
      return WriteBufferAs<uint8_t>(buffer, data);
    case ElementType::UInt16:
      return WriteBufferAs<uint16_t>(buffer, data);
    case ElementType::UInt32:
      return WriteBufferAs<uint32_t>(buffer, data);
    case ElementType::UInt64:
      return WriteBufferAs<uint64_t>(buffer, data);

    // Half-precision formats written as raw 16-bit payloads.
    case ElementType::Float16:
    case ElementType::BFloat16:
      return WriteBufferAs<uint16_t>(buffer, data);

    default:
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Unsupported element type.");
  }
}

// Fill tensor buffer with random data
inline Expected<void> FillBufferWithRandomData(TensorBuffer& buffer) {
  constexpr float kScale = 0.12345f;
  LITERT_ASSIGN_OR_RETURN(auto type, buffer.TensorType());
  const auto& layout = type.Layout();
  size_t total_elements =
      std::accumulate(layout.Dimensions().begin(), layout.Dimensions().end(), 1,
                      std::multiplies<size_t>());
  if (type.ElementType() == ElementType::Float16 ||
      type.ElementType() == ElementType::Float32 ||
      type.ElementType() == ElementType::BFloat16) {
    std::vector<float> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = std::sin(i * kScale);
    }
    buffer.Write<float>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::Int32) {
    std::vector<int32_t> data(total_elements);
    unsigned int seed = 7;
    for (size_t i = 0; i < total_elements; ++i) {
#if !defined(LITERT_WINDOWS_OS)
      data[i] = rand_r(&seed) % 1024 + 1;
#else   // !defined(LITERT_WINDOWS_OS)
      data[i] = rand() % 1024 + 1;
#endif  // !defined(LITERT_WINDOWS_OS)
    }
    buffer.Write<int32_t>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::Int16) {
    std::vector<int16_t> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = i % 2048;
    }
    buffer.Write<int16_t>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::Int64) {
    std::vector<int64_t> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = i % 2048;
    }
    buffer.Write<int64_t>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::Int8) {
    std::vector<int8_t> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = i % 256 - 128;
    }
    buffer.Write<int8_t>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::UInt8) {
    std::vector<uint8_t> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = i % 256;
    }
    buffer.Write<uint8_t>(absl::MakeConstSpan(data));
  } else if (type.ElementType() == ElementType::Bool) {
    std::vector<uint8_t> data(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
      data[i] = i % 2;
    }
    buffer.Write<uint8_t>(absl::MakeConstSpan(data));
  }
  return {};
}

// Helper function to quantize float data to an integral type using affine
// quantization.
template <typename T>
inline std::vector<T> QuantizeData(absl::Span<const float> float_data,
                                   float scale, int64_t zero_point) {
  static_assert(std::is_integral<T>::value, "Integral type required");
  std::vector<T> quantized_data(float_data.size());
  constexpr int64_t qmin = static_cast<int64_t>(std::numeric_limits<T>::min());
  constexpr int64_t qmax = static_cast<int64_t>(std::numeric_limits<T>::max());

  if (scale <= 0.0f || !std::isfinite(scale)) {
    const T clamped_zp = static_cast<T>(std::clamp(zero_point, qmin, qmax));
    std::fill(quantized_data.begin(), quantized_data.end(), clamped_zp);
    return quantized_data;
  }

  const double inv_scale = 1.0 / static_cast<double>(scale);
  const double zp = static_cast<double>(zero_point);
  const double min_bound = static_cast<double>(qmin);
  const double max_bound = static_cast<double>(qmax);

  for (size_t i = 0; i < float_data.size(); ++i) {
    const float val = float_data[i];
    if (std::isnan(val)) {
      quantized_data[i] = static_cast<T>(std::clamp(zero_point, qmin, qmax));
      continue;
    }
    double unclamped = std::round(static_cast<double>(val) * inv_scale) + zp;
    if (unclamped <= min_bound) {
      quantized_data[i] = static_cast<T>(qmin);
    } else if (unclamped >= max_bound) {
      quantized_data[i] = static_cast<T>(qmax);
    } else {
      quantized_data[i] = static_cast<T>(static_cast<int64_t>(unclamped));
    }
  }
  return quantized_data;
}

Expected<void> FillInputBuffersWithCustomData(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& input_buffers, absl::string_view input_dir,
    bool quantize_inputs = false);

// Writes output buffers to .raw files in the given directory, using the model's
// output signature names as filenames.
Expected<void> WriteOutputBuffersToFiles(
    const CompiledModel& compiled_model, size_t signature_index,
    std::vector<TensorBuffer>& output_buffers, absl::string_view output_dir);

}  // namespace tensor_utils
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_TENSOR_UTILS_H_
