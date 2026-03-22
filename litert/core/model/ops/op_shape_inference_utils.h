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

#include "tflite/schema/schema_generated.h"

namespace litert::internal {

// Computes the output size of a convolution or pooling operation.
inline int32_t ComputeOutputSize(tflite::Padding padding, int32_t image_size,
                                 int32_t filter_size, int32_t stride,
                                 int32_t dilation_rate = 1) {
  if (image_size == -1) return -1;
  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding) {
    case tflite::Padding_SAME:
      return (image_size + stride - 1) / stride;
    case tflite::Padding_VALID:
      return (image_size - effective_filter_size + stride) / stride;
    default:
      return -1;
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_OP_SHAPE_INFERENCE_UTILS_H_
