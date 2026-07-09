/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_PADDING_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_PADDING_H_

// LINT.IfChange
#include <cstdint>
#include <limits>

#include "tflite/converter/core/c/builtin_op_data.h"

namespace tflite_migration {

inline int ClampPaddingSizeToInt(int64_t value) {
  if (value > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  if (value < std::numeric_limits<int>::min()) {
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(value);
}

inline int64_t ComputeEffectiveFilterSize(int filter_size,
                                          int dilation_rate) {
  return (static_cast<int64_t>(filter_size) - 1) * dilation_rate + 1;
}

// Matching GetWindowedOutputSize in TensorFlow.
inline int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate = 1) {
  const int64_t effective_filter_size =
      ComputeEffectiveFilterSize(filter_size, dilation_rate);

  // TODO(b/186448822): This uses 0 since the function has no other way to
  // report error case
  if (stride == 0) return 0;

  switch (padding) {
    case kTfLitePaddingSame:
      return ClampPaddingSizeToInt(
          (static_cast<int64_t>(image_size) + stride - 1) / stride);
    case kTfLitePaddingValid:
      return ClampPaddingSizeToInt(
          (static_cast<int64_t>(image_size) + stride - effective_filter_size) /
          stride);
    default:
      return 0;
  }
}

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
inline int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset) {
  if (offset == nullptr || in_size < 0 || filter_size <= 0 || out_size < 0 ||
      stride <= 0 || dilation_rate <= 0) {
    if (offset != nullptr) *offset = 0;
    return 0;
  }
  const int64_t effective_filter_size =
      ComputeEffectiveFilterSize(filter_size, dilation_rate);
  int64_t total_padding = ((static_cast<int64_t>(out_size) - 1) * stride +
                           effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = static_cast<int>(total_padding % 2);
  return ClampPaddingSizeToInt(total_padding / 2);
}

}  // namespace tflite_migration

// LINT.ThenChange(//tflite/kernels/padding.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_KERNELS_PADDING_H_
