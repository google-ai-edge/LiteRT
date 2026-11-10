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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_

#include <cstdint>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

// Returns true if the product of all positive dimensions (> 1) in `shape` fits
// in a signed 32-bit integer. Dynamic (-1) and zero (0) dimensions are skipped
// so that static sub-products are still validated even when another dimension
// is dynamic or zero.
inline bool CheckShapePositiveProductFitsInInt32(
    absl::Span<const int32_t> shape) {
  int32_t product = 1;
  for (int32_t dim : shape) {
    if (dim > 1 && !SafeMultiply(product, dim, product)) {
      return false;
    }
  }
  return true;
}

inline bool ComputeEffectiveFilterSize(int32_t filter_size,
                                       int32_t dilation_rate,
                                       int32_t& effective_filter_size) {
  if (filter_size <= 0 || dilation_rate <= 0) {
    return false;
  }
  int32_t dilated_filter_size = 0;
  return SafeSubtract(filter_size, 1, dilated_filter_size) &&
         SafeMultiply(dilated_filter_size, dilation_rate,
                      dilated_filter_size) &&
         SafeAdd(dilated_filter_size, 1, effective_filter_size);
}

// Computes the output size of a convolution or pooling operation.
inline int32_t ComputeOutputSize(tflite::Padding padding, int32_t image_size,
                                 int32_t filter_size, int32_t stride,
                                 int32_t dilation_rate = 1) {
  if (image_size == -1) return -1;
  if (image_size < 0 || filter_size <= 0 || stride <= 0 || dilation_rate <= 0) {
    return -1;
  }

  int32_t effective_filter_size = 0;
  if (!ComputeEffectiveFilterSize(filter_size, dilation_rate,
                                  effective_filter_size)) {
    return -1;
  }

  int32_t value = 0;
  switch (padding) {
    case tflite::Padding_SAME: {
      int32_t stride_minus_one = 0;
      if (!SafeSubtract(stride, 1, stride_minus_one) ||
          !SafeAdd(image_size, stride_minus_one, value) ||
          !SafeDivide(value, stride, value)) {
        return -1;
      }
      break;
    }
    case tflite::Padding_VALID:
      if (!SafeSubtract(image_size, effective_filter_size, value) ||
          !SafeAdd(value, stride, value) || !SafeDivide(value, stride, value)) {
        return -1;
      }
      break;
    default:
      return -1;
  }

  return value >= 0 ? value : -1;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_
