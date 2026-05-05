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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_POOLING_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_POOLING_H_

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferPool2D(const LiteRtOpT& op,
                                absl::Span<const Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kPool2DMinArgs = 1;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kPool2DRank = 4;

  if (input_shapes.size() < kPool2DMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  if (input_shape.size() != kPool2DRank) {
    return kLiteRtStatusErrorInvalidArgument;  // Expect NHWC
  }

  const auto& opts = GetTflOptions(op);
  const auto* pool_opts = opts.AsPool2DOptions();
  if (!pool_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_h = pool_opts->stride_h;
  int32_t stride_w = pool_opts->stride_w;
  int32_t filter_h = pool_opts->filter_height;
  int32_t filter_w = pool_opts->filter_width;
  tflite::Padding padding = pool_opts->padding;

  int32_t out_h =
      ComputeOutputSize(padding, input_shape[1], filter_h, stride_h);
  int32_t out_w =
      ComputeOutputSize(padding, input_shape[2], filter_w, stride_w);

  output_shapes[0] = {input_shape[0], out_h, out_w, input_shape[3]};
  return kLiteRtStatusOk;
}

template <bool IsMax>
inline void ReferencePool2D(const float* input_data, float* output_data,
                            int batch, int in_h, int in_w, int in_c, int out_h,
                            int out_w, int filter_h, int filter_w, int stride_h,
                            int stride_w, int pad_t, int pad_l,
                            tflite::ActivationFunctionType faf) {
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int c = 0; c < in_c; ++c) {
          float res = IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;
          int count = 0;
          for (int kh = 0; kh < filter_h; ++kh) {
            for (int kw = 0; kw < filter_w; ++kw) {
              int ih = oh * stride_h + kh - pad_t;
              int iw = ow * stride_w + kw - pad_l;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                float val =
                    input_data[((b * in_h + ih) * in_w + iw) * in_c + c];
                if constexpr (IsMax) {
                  res = std::max(res, val);
                } else {
                  res += val;
                  count++;
                }
              }
            }
          }
          if constexpr (!IsMax) {
            if (count > 0) res /= count;
          }

          // Apply activation
          if (faf == tflite::ActivationFunctionType_RELU) {
            if (res < 0.0f) res = 0.0f;
          } else if (faf == tflite::ActivationFunctionType_RELU6) {
            if (res < 0.0f) res = 0.0f;
            if (res > 6.0f) res = 6.0f;
          }

          output_data[((b * out_h + oh) * out_w + ow) * in_c + c] = res;
        }
      }
    }
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_POOLING_H_
