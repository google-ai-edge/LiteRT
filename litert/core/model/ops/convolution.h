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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/op_shape_inference_utils.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {

inline LiteRtStatus InferConv2D(const LiteRtOpT& op,
                                absl::Span<Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kConv2DMinArgs = 2;
  constexpr size_t kConv2DRank = 4;

  if (input_shapes.size() < kConv2DMinArgs || output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kConv2DRank || filter_shape.size() != kConv2DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* conv_opts = opts.AsConv2DOptions();
  if (!conv_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_h = conv_opts->stride_h;
  int32_t stride_w = conv_opts->stride_w;
  int32_t dilation_h = conv_opts->dilation_h_factor;
  int32_t dilation_w = conv_opts->dilation_w_factor;
  tflite::Padding padding = conv_opts->padding;

  // Filter: [Out, H, W, In]
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  int32_t remainder = 0;
  int32_t groups = 0;
  if (filter_shape[3] <= 0 ||
      !SafeModulus(input_shape[3], filter_shape[3], remainder) ||
      remainder != 0 || !SafeDivide(input_shape[3], filter_shape[3], groups)) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D input channels (%d) must be a multiple of filter input "
               "channels (%d)",
               input_shape[3], filter_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (groups <= 0 || !SafeModulus(filter_shape[0], groups, remainder) ||
      remainder != 0) {
    LITERT_LOG(LITERT_ERROR,
               "Conv2D output channels (%d) must be a multiple of groups (%d)",
               filter_shape[0], groups);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid stride: %dx%d", stride_h,
               stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  constexpr int32_t kMaxStrideOrDilation = std::numeric_limits<int16_t>::max();

  if (stride_h > kMaxStrideOrDilation || stride_w > kMaxStrideOrDilation ||
      dilation_h > kMaxStrideOrDilation || dilation_w > kMaxStrideOrDilation) {
    LITERT_LOG(LITERT_ERROR, "Conv2D stride or dilation exceeds limit");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid dilation: %dx%d", dilation_h,
               dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[0];

  if (input_shapes.size() >= 3 && !input_shapes[2].empty()) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "Conv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "Conv2D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  int32_t out_h = ComputeOutputSize(padding, input_shape[1], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[2], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_h ||
        input_shape[2] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "Conv2D input size (%dx%d) too small for effective filter "
                 "size (%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], effective_filter_h,
                 effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv2D invalid output size: %dx%d", out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = {input_shape[0], out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR, "Conv2D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferDepthwiseConv2D(const LiteRtOpT& op,
                                         absl::Span<Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kDepthwiseConv2DMinArgs = 2;
  constexpr size_t kDepthwiseConv2DRank = 4;

  if (input_shapes.size() < kDepthwiseConv2DMinArgs ||
      output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kDepthwiseConv2DRank ||
      filter_shape.size() != kDepthwiseConv2DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* dw_opts = opts.AsDepthwiseConv2DOptions();
  if (!dw_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_h = dw_opts->stride_h;
  int32_t stride_w = dw_opts->stride_w;
  int32_t dilation_h = dw_opts->dilation_h_factor;
  int32_t dilation_w = dw_opts->dilation_w_factor;
  tflite::Padding padding = dw_opts->padding;

  // Filter: [1, H, W, OutChannels]
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  if (filter_shape[0] != 1) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D filter dim 0 must be 1, got %d",
               filter_shape[0]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[3];

  int32_t expected_out_channels = 0;
  if (!SafeMultiply(input_shape[3], dw_opts->depth_multiplier,
                    expected_out_channels) ||
      out_channels != expected_out_channels) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D out_channels mismatch: %d vs %d*%d",
               out_channels, input_shape[3], dw_opts->depth_multiplier);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid stride: %dx%d", stride_h,
               stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  constexpr int32_t kMaxStrideOrDilation = std::numeric_limits<int16_t>::max();

  if (stride_h > kMaxStrideOrDilation || stride_w > kMaxStrideOrDilation ||
      dilation_h > kMaxStrideOrDilation || dilation_w > kMaxStrideOrDilation) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D stride or dilation exceeds limit");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid dilation: %dx%d",
               dilation_h, dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() >= 3 && !input_shapes[2].empty()) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_h = ComputeOutputSize(padding, input_shape[1], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[2], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_h ||
        input_shape[2] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "DepthwiseConv2D input size (%dx%d) too small for effective "
                 "filter size (%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], effective_filter_h,
                 effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "DepthwiseConv2D invalid output size: %dx%d",
               out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = {input_shape[0], out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR,
               "DepthwiseConv2D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferTransposeConv(const LiteRtOpT& op,
                                       absl::Span<Dims> input_shapes,
                                       std::vector<Dims>& output_shapes) {
  constexpr size_t kOutputShapeArgIndex = 0;
  constexpr size_t kWeightsArgIndex = 1;
  constexpr size_t kInputArgIndex = 2;
  constexpr size_t kBiasArgIndex = 3;
  constexpr size_t kTransposeConvMinArgs = 3;
  constexpr size_t kTransposeConvRank = 4;
  constexpr size_t kShapeTensorSize = kTransposeConvRank * sizeof(int32_t);
  constexpr int32_t kMaxStride = std::numeric_limits<int16_t>::max();

  if (input_shapes.size() < kTransposeConvMinArgs ||
      output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& output_shape_arg = input_shapes[kOutputShapeArgIndex];
  const auto& weights_shape = input_shapes[kWeightsArgIndex];
  const auto& input_shape = input_shapes[kInputArgIndex];

  if (output_shape_arg.size() != 1 ||
      (output_shape_arg[0] != static_cast<int32_t>(kTransposeConvRank) &&
       output_shape_arg[0] != -1) ||
      weights_shape.size() != kTransposeConvRank ||
      input_shape.size() != kTransposeConvRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  // TFLite TransposeConv weights: [OutputDepth, H, W, InputDepth].
  if (weights_shape[0] <= 0 || weights_shape[1] <= 0 || weights_shape[2] <= 0 ||
      weights_shape[3] <= 0 ||
      !CheckShapePositiveProductFitsInInt32(weights_shape)) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv invalid weights shape");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shape[0] < -1 || (input_shape[1] <= 0 && input_shape[1] != -1) ||
      (input_shape[2] <= 0 && input_shape[2] != -1) ||
      (input_shape[3] <= 0 && input_shape[3] != -1) ||
      !CheckShapePositiveProductFitsInInt32(input_shape)) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv invalid input shape");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shape[3] != -1 && input_shape[3] != weights_shape[3]) {
    LITERT_LOG(LITERT_ERROR, "TransposeConv input channels mismatch: %d vs %d",
               input_shape[3], weights_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  // Validate col2im rows (in_h * in_w), columns (out_c * filter_h * filter_w),
  // and total col2im element count fit in int32_t.
  if (!CheckShapePositiveProductFitsInInt32({input_shape[1], input_shape[2],
                                             weights_shape[0], weights_shape[1],
                                             weights_shape[2]})) {
    LITERT_LOG(LITERT_ERROR,
               "TransposeConv col2im element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (input_shapes.size() > kBiasArgIndex &&
      !input_shapes[kBiasArgIndex].empty()) {
    const auto& bias_shape = input_shapes[kBiasArgIndex];
    if (bias_shape.size() != 1 ||
        (bias_shape[0] != weights_shape[0] && bias_shape[0] != -1)) {
      LITERT_LOG(LITERT_ERROR, "TransposeConv bias size mismatch: %d vs %d",
                 bias_shape[0], weights_shape[0]);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  const auto* tc_opts = GetTflOptions(op).AsTransposeConvOptions();
  if (tc_opts != nullptr) {
    if (tc_opts->stride_h <= 0 || tc_opts->stride_h > kMaxStride ||
        tc_opts->stride_w <= 0 || tc_opts->stride_w > kMaxStride) {
      LITERT_LOG(LITERT_ERROR, "TransposeConv invalid stride: %dx%d",
                 tc_opts->stride_h, tc_opts->stride_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  // If output_shape tensor is constant, validate and use it.
  if (op.Inputs().size() > kOutputShapeArgIndex &&
      op.Inputs()[kOutputShapeArgIndex] != nullptr) {
    const auto& output_size_tensor = op.Input(kOutputShapeArgIndex);
    auto buf = output_size_tensor.Weights().Buffer();
    if (buf.Size() > 0) {
      if (buf.Size() != kShapeTensorSize) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
      const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
      Dims out_shape(dims, dims + kTransposeConvRank);
      if (out_shape[0] < 0 || out_shape[1] <= 0 || out_shape[2] <= 0 ||
          out_shape[3] != weights_shape[0] ||
          (input_shape[0] != -1 && out_shape[0] != input_shape[0]) ||
          !CheckShapePositiveProductFitsInInt32(out_shape)) {
        LITERT_LOG(LITERT_ERROR, "TransposeConv invalid constant output shape");
        return kLiteRtStatusErrorShapeInferenceFailed;
      }

      if (tc_opts != nullptr) {
        if (tc_opts->padding == tflite::Padding_VALID &&
            (out_shape[1] < weights_shape[1] ||
             out_shape[2] < weights_shape[2])) {
          LITERT_LOG(LITERT_ERROR,
                     "TransposeConv output spatial size smaller than filter "
                     "with VALID padding");
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        const int32_t expected_in_h =
            ComputeOutputSize(tc_opts->padding, out_shape[1], weights_shape[1],
                              tc_opts->stride_h);
        const int32_t expected_in_w =
            ComputeOutputSize(tc_opts->padding, out_shape[2], weights_shape[2],
                              tc_opts->stride_w);
        if (expected_in_h <= 0 || expected_in_w <= 0 ||
            (input_shape[1] > 0 && input_shape[1] != expected_in_h) ||
            (input_shape[2] > 0 && input_shape[2] != expected_in_w)) {
          LITERT_LOG(LITERT_ERROR,
                     "TransposeConv input spatial dimensions (%dx%d) do not "
                     "match expected (%dx%d)",
                     input_shape[1], input_shape[2], expected_in_h,
                     expected_in_w);
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
      }

      output_shapes[0] = std::move(out_shape);
      return kLiteRtStatusOk;
    }
  }

  Dims out_shape = {input_shape[0], -1, -1, weights_shape[0]};
  if (!CheckShapePositiveProductFitsInInt32(out_shape)) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferConv3D(const LiteRtOpT& op,
                                absl::Span<Dims> input_shapes,
                                std::vector<Dims>& output_shapes) {
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kConv3DMinArgs = 2;
  constexpr size_t kConv3DRank = 5;

  if (input_shapes.size() < kConv3DMinArgs || output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& filter_shape = input_shapes[kFilterArgIndex];

  if (input_shape.size() != kConv3DRank || filter_shape.size() != kConv3DRank) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto& opts = GetTflOptions(op);
  const auto* conv_opts = opts.AsConv3DOptions();
  if (!conv_opts) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  int32_t stride_d = conv_opts->stride_d;
  int32_t stride_h = conv_opts->stride_h;
  int32_t stride_w = conv_opts->stride_w;
  int32_t dilation_d = conv_opts->dilation_d_factor;
  int32_t dilation_h = conv_opts->dilation_h_factor;
  int32_t dilation_w = conv_opts->dilation_w_factor;
  tflite::Padding padding = conv_opts->padding;

  int32_t filter_d = filter_shape[0];
  int32_t filter_h = filter_shape[1];
  int32_t filter_w = filter_shape[2];
  if (input_shape[4] != filter_shape[3]) {
    LITERT_LOG(LITERT_ERROR, "Conv3D input channels mismatch: %d vs %d",
               input_shape[4], filter_shape[3]);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (stride_d <= 0 || stride_h <= 0 || stride_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid stride: %dx%dx%d", stride_d,
               stride_h, stride_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  if (dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid dilation: %dx%dx%d", dilation_d,
               dilation_h, dilation_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_channels = filter_shape[4];

  if (input_shapes.size() >= 3) {
    const auto& bias_shape = input_shapes[2];
    if (bias_shape.size() != 1 || bias_shape[0] != out_channels) {
      LITERT_LOG(LITERT_ERROR, "Conv3D bias size mismatch: %d vs %d",
                 bias_shape.empty() ? 0 : bias_shape[0], out_channels);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  int32_t effective_filter_d = 0;
  int32_t effective_filter_h = 0;
  int32_t effective_filter_w = 0;
  if (!ComputeEffectiveFilterSize(filter_d, dilation_d, effective_filter_d) ||
      !ComputeEffectiveFilterSize(filter_h, dilation_h, effective_filter_h) ||
      !ComputeEffectiveFilterSize(filter_w, dilation_w, effective_filter_w)) {
    LITERT_LOG(LITERT_ERROR, "Conv3D effective filter size overflow");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  int32_t out_d = ComputeOutputSize(padding, input_shape[1], filter_d, stride_d,
                                    dilation_d);
  int32_t out_h = ComputeOutputSize(padding, input_shape[2], filter_h, stride_h,
                                    dilation_h);
  int32_t out_w = ComputeOutputSize(padding, input_shape[3], filter_w, stride_w,
                                    dilation_w);

  if (padding == tflite::Padding_VALID) {
    if (input_shape[1] < effective_filter_d ||
        input_shape[2] < effective_filter_h ||
        input_shape[3] < effective_filter_w) {
      LITERT_LOG(LITERT_ERROR,
                 "Conv3D input size (%dx%dx%d) too small for effective filter "
                 "size (%dx%dx%d) with VALID padding",
                 input_shape[1], input_shape[2], input_shape[3],
                 effective_filter_d, effective_filter_h, effective_filter_w);
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  if (out_d <= 0 || out_h <= 0 || out_w <= 0) {
    LITERT_LOG(LITERT_ERROR, "Conv3D invalid output size: %dx%dx%d", out_d,
               out_h, out_w);
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  Dims out_shape = {input_shape[0], out_d, out_h, out_w, out_channels};
  if (!CheckShapePositiveProductFitsInInt32(input_shape) ||
      !CheckShapePositiveProductFitsInInt32(filter_shape) ||
      !CheckShapePositiveProductFitsInInt32(out_shape)) {
    LITERT_LOG(LITERT_ERROR, "Conv3D tensor element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const bool need_im2col = stride_d != 1 || stride_h != 1 || stride_w != 1 ||
                           dilation_d != 1 || dilation_h != 1 ||
                           dilation_w != 1 || filter_d != 1 || filter_h != 1 ||
                           filter_w != 1;
  if (need_im2col && !CheckShapePositiveProductFitsInInt32(
                         {input_shape[0], out_d, out_h, out_w, filter_shape[3],
                          filter_d, filter_h, filter_w})) {
    LITERT_LOG(LITERT_ERROR, "Conv3D im2col element count overflows int32");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferConv3DTranspose(const LiteRtOpT& op,
                                         absl::Span<Dims> input_shapes,
                                         std::vector<Dims>& output_shapes) {
  constexpr size_t kOutputShapeArgIndex = 0;
  constexpr size_t kFilterArgIndex = 1;
  constexpr size_t kInputArgIndex = 2;
  constexpr size_t kConv3DTransposeMinArgs = 3;
  constexpr size_t kConv3DTransposeRank = 5;

  if (input_shapes.size() < kConv3DTransposeMinArgs ||
      output_shapes.size() != 1) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op.Inputs().size() > kOutputShapeArgIndex &&
      op.Inputs()[kOutputShapeArgIndex] != nullptr) {
    const auto& output_size_tensor = op.Input(kOutputShapeArgIndex);
    if (output_size_tensor.Weights().Buffer().Size() >= sizeof(int32_t)) {
      auto buf = output_size_tensor.Weights().Buffer();
      const int32_t* dims = reinterpret_cast<const int32_t*>(buf.Data());
      int rank = buf.Size() / sizeof(int32_t);
      Dims out_shape;
      for (int i = 0; i < rank; ++i) out_shape.push_back(dims[i]);
      output_shapes[0] = out_shape;
      return kLiteRtStatusOk;
    }
  }

  const auto& input_shape = input_shapes[kInputArgIndex];
  output_shapes[0] = Dims(input_shape.size(), -1);
  if (!input_shape.empty()) {
    output_shapes[0][0] = input_shape[0];  // Batch
  }

  // Filter: [D, H, W, In, Out] -> Out is index 4.
  const auto& filter_shape = input_shapes[kFilterArgIndex];
  if (filter_shape.size() == kConv3DTransposeRank &&
      output_shapes[0].size() == kConv3DTransposeRank) {
    output_shapes[0][4] = filter_shape[4];
  }

  return kLiteRtStatusOk;
}

inline void ReferenceConv2D(const float* input_data, const float* filter_data,
                            const float* bias_data, float* output_data,
                            int batch, int in_h, int in_w, int in_c, int out_h,
                            int out_w, int out_c, int filter_h, int filter_w,
                            int stride_h, int stride_w, int dilation_h,
                            int dilation_w, int pad_t, int pad_l,
                            tflite::ActivationFunctionType faf) {
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int oc = 0; oc < out_c; ++oc) {
          float sum = 0;
          for (int kh = 0; kh < filter_h; ++kh) {
            for (int kw = 0; kw < filter_w; ++kw) {
              for (int ic = 0; ic < in_c; ++ic) {
                int ih = oh * stride_h + kh * dilation_h - pad_t;
                int iw = ow * stride_w + kw * dilation_w - pad_l;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                  int input_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                  int filter_idx =
                      ((oc * filter_h + kh) * filter_w + kw) * in_c + ic;
                  sum += input_data[input_idx] * filter_data[filter_idx];
                }
              }
            }
          }
          float output_val = sum + (bias_data ? bias_data[oc] : 0.0f);
          if (faf == tflite::ActivationFunctionType_RELU) {
            if (output_val < 0.0f) output_val = 0.0f;
          } else if (faf == tflite::ActivationFunctionType_RELU6) {
            if (output_val < 0.0f) output_val = 0.0f;
            if (output_val > 6.0f) output_val = 6.0f;
          }
          output_data[((b * out_h + oh) * out_w + ow) * out_c + oc] =
              output_val;
        }
      }
    }
  }
}

inline void ReferenceDepthwiseConv2D(
    const float* input_data, const float* filter_data, const float* bias_data,
    float* output_data, int batch, int in_h, int in_w, int in_c, int out_h,
    int out_w, int out_c, int filter_h, int filter_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, int pad_t, int pad_l,
    int depth_multiplier, tflite::ActivationFunctionType faf) {
  for (int b = 0; b < batch; ++b) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int oc = 0; oc < out_c; ++oc) {
          float sum = 0;
          for (int kh = 0; kh < filter_h; ++kh) {
            for (int kw = 0; kw < filter_w; ++kw) {
              int ih = oh * stride_h + kh * dilation_h - pad_t;
              int iw = ow * stride_w + kw * dilation_w - pad_l;
              if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int input_idx = ((b * in_h + ih) * in_w + iw) * in_c +
                                oc / depth_multiplier;
                int filter_index = (kh * filter_w + kw) * out_c + oc;
                sum += input_data[input_idx] * filter_data[filter_index];
              }
            }
          }
          float output_val = sum + (bias_data ? bias_data[oc] : 0.0f);
          if (faf == tflite::ActivationFunctionType_RELU) {
            if (output_val < 0.0f) output_val = 0.0f;
          } else if (faf == tflite::ActivationFunctionType_RELU6) {
            if (output_val < 0.0f) output_val = 0.0f;
            if (output_val > 6.0f) output_val = 6.0f;
          }
          output_data[((b * out_h + oh) * out_w + ow) * out_c + oc] =
              output_val;
        }
      }
    }
  }
}

// Caller-supplied NHWC spatial parameters for 2D transposed convolution.
// Filter tensor layout is `[out_c, filter_h, filter_w, in_c]`.
struct TransposeConvParams {
  int32_t batch = 0;
  int32_t in_h = 0;
  int32_t in_w = 0;
  int32_t in_c = 0;
  int32_t out_h = 0;
  int32_t out_w = 0;
  int32_t out_c = 0;
  int32_t filter_h = 0;
  int32_t filter_w = 0;
  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t pad_t = 0;
  int32_t pad_l = 0;
};

// Validated spatial geometry and checked tensor element counts for 2D
// transposed convolution.
struct TransposeConvGeometry {
  TransposeConvParams params;
  size_t input_num_elements = 0;
  size_t filter_num_elements = 0;
  size_t output_num_elements = 0;
};

// Validates non-negative batch, strictly positive spatial/channel/filter/stride
// dimensions, and checked element/byte counts for input, filter, output, and
// optional accumulator buffers.
inline Expected<TransposeConvGeometry> PrepareTransposeConvGeometry(
    const TransposeConvParams& params, size_t input_elem_size,
    size_t filter_elem_size, size_t output_elem_size,
    size_t acc_elem_size = 0) {
  if (params.batch < 0 || params.in_h <= 0 || params.in_w <= 0 ||
      params.in_c <= 0 || params.out_h <= 0 || params.out_w <= 0 ||
      params.out_c <= 0 || params.filter_h <= 0 || params.filter_w <= 0 ||
      params.stride_h <= 0 || params.stride_w <= 0) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "TransposeConv requires non-negative batch and positive spatial, "
        "channel, filter, and stride dimensions");
  }
  const std::array<int32_t, 4> in_dims = {params.batch, params.in_h,
                                          params.in_w, params.in_c};
  const std::array<int32_t, 4> filter_dims = {params.out_c, params.filter_h,
                                              params.filter_w, params.in_c};
  const std::array<int32_t, 4> out_dims = {params.batch, params.out_h,
                                           params.out_w, params.out_c};
  Expected<ContiguousLayout> in_layout =
      PrepareContiguousLayout(in_dims, input_elem_size);
  if (!in_layout) {
    return in_layout.Error();
  }
  Expected<ContiguousLayout> filter_layout =
      PrepareContiguousLayout(filter_dims, filter_elem_size);
  if (!filter_layout) {
    return filter_layout.Error();
  }
  Expected<ContiguousLayout> out_layout =
      PrepareContiguousLayout(out_dims, output_elem_size);
  if (!out_layout) {
    return out_layout.Error();
  }
  if (acc_elem_size > 0) {
    Expected<size_t> acc_bytes =
        CheckedByteCount(out_layout->num_elements, acc_elem_size);
    if (!acc_bytes) {
      return acc_bytes.Error();
    }
  }
  return TransposeConvGeometry{
      .params = params,
      .input_num_elements = in_layout->num_elements,
      .filter_num_elements = filter_layout->num_elements,
      .output_num_elements = out_layout->num_elements,
  };
}

namespace detail {

template <typename T>
inline constexpr bool kIsSupportedQuantizedConvDataType =
    std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>;

template <typename T>
inline constexpr bool kIsSupportedQuantizedConvBiasType =
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

// Computes the row-major flat index for 4D coordinates within `dims`.
// Callers must guarantee each coordinate is in `[0, dims[i])` and the product
// of `dims` fits in `size_t`.
constexpr size_t FlatIndex4D(const std::array<size_t, 4>& coords,
                             const std::array<size_t, 4>& dims) {
  return ((coords[0] * dims[1] + coords[1]) * dims[2] + coords[2]) * dims[3] +
         coords[3];
}

inline bool IsSupportedTransposeConvActivation(
    tflite::ActivationFunctionType faf) {
  switch (faf) {
    case tflite::ActivationFunctionType_NONE:
    case tflite::ActivationFunctionType_RELU:
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
    case tflite::ActivationFunctionType_RELU6:
      return true;
    default:
      return false;
  }
}

// Validates the fused activation, spatial geometry (including optional `AccT`
// accumulator allocation size when `AccT != OutputT`), and input, filter,
// optional bias, and output buffer capacities.
template <typename AccT, typename InputT, typename FilterT, typename BiasT,
          typename OutputT>
inline Expected<TransposeConvGeometry> PrepareAndValidateTransposeConv(
    absl::Span<const InputT> input, absl::Span<const FilterT> filter,
    absl::Span<const BiasT> bias, absl::Span<OutputT> output,
    const TransposeConvParams& params, tflite::ActivationFunctionType faf) {
  if (!IsSupportedTransposeConvActivation(faf)) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Unsupported TransposeConv fused activation");
  }
  constexpr size_t kAccElemSize =
      std::is_same_v<AccT, OutputT> ? 0 : sizeof(AccT);
  Expected<TransposeConvGeometry> geom = PrepareTransposeConvGeometry(
      params, sizeof(InputT), sizeof(FilterT), sizeof(OutputT), kAccElemSize);
  if (!geom) {
    return geom.Error();
  }
  if (!ValidateBufferCapacity(input, geom->input_num_elements) ||
      !ValidateBufferCapacity(filter, geom->filter_num_elements) ||
      !ValidateBufferCapacity(output, geom->output_num_elements) ||
      (!bias.empty() && !ValidateBufferCapacity(
                            bias, static_cast<size_t>(geom->params.out_c)))) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "TransposeConv buffer is smaller than required");
  }
  return geom;
}

// Accumulates NHWC 2D transposed convolution into `acc` using pre-validated
// `geom` and pre-validated buffer capacities. `acc` must not alias `input` or
// `filter`.
template <typename InputT, typename FilterT, typename AccT>
inline void AccumulateTransposeConvUnchecked(absl::Span<const InputT> input,
                                             absl::Span<const FilterT> filter,
                                             absl::Span<AccT> acc,
                                             const TransposeConvGeometry& geom,
                                             AccT input_zero_point = AccT{0}) {
  static_assert(
      (std::is_floating_point_v<InputT> && std::is_floating_point_v<FilterT> &&
       std::is_floating_point_v<AccT>) ||
          (kIsSupportedQuantizedConvDataType<InputT> &&
           kIsSupportedQuantizedConvDataType<FilterT> &&
           std::is_same_v<AccT, int64_t>),
      "AccumulateTransposeConvUnchecked supports floating-point or <=16-bit "
      "integer inputs/filters accumulated into int64_t.");
  const TransposeConvParams& params = geom.params;
  const size_t batch = static_cast<size_t>(params.batch);
  const size_t in_h = static_cast<size_t>(params.in_h);
  const size_t in_w = static_cast<size_t>(params.in_w);
  const size_t in_c = static_cast<size_t>(params.in_c);
  const size_t out_h = static_cast<size_t>(params.out_h);
  const size_t out_w = static_cast<size_t>(params.out_w);
  const size_t out_c = static_cast<size_t>(params.out_c);
  const size_t filter_h = static_cast<size_t>(params.filter_h);
  const size_t filter_w = static_cast<size_t>(params.filter_w);
  const size_t filter_oc_stride = filter_h * filter_w * in_c;
  const std::array<size_t, 4> in_dims = {batch, in_h, in_w, in_c};
  const std::array<size_t, 4> filter_dims = {out_c, filter_h, filter_w, in_c};
  const std::array<size_t, 4> out_dims = {batch, out_h, out_w, out_c};
  std::fill_n(acc.data(), geom.output_num_elements, AccT{0});
  for (size_t b = 0; b < batch; ++b) {
    for (size_t ih = 0; ih < in_h; ++ih) {
      for (size_t iw = 0; iw < in_w; ++iw) {
        for (size_t ic = 0; ic < in_c; ++ic) {
          const AccT in_val =
              static_cast<AccT>(input[FlatIndex4D({b, ih, iw, ic}, in_dims)]) -
              input_zero_point;
          for (size_t kh = 0; kh < filter_h; ++kh) {
            const int64_t oh = static_cast<int64_t>(ih) * params.stride_h +
                               static_cast<int64_t>(kh) - params.pad_t;
            if (oh < 0 || oh >= params.out_h) {
              continue;
            }
            for (size_t kw = 0; kw < filter_w; ++kw) {
              const int64_t ow = static_cast<int64_t>(iw) * params.stride_w +
                                 static_cast<int64_t>(kw) - params.pad_l;
              if (ow < 0 || ow >= params.out_w) {
                continue;
              }
              const size_t filter_spatial_offset =
                  FlatIndex4D({0, kh, kw, ic}, filter_dims);
              const size_t acc_base = FlatIndex4D(
                  {b, static_cast<size_t>(oh), static_cast<size_t>(ow), 0},
                  out_dims);
              for (size_t oc = 0; oc < out_c; ++oc) {
                const AccT weight = static_cast<AccT>(
                    filter[oc * filter_oc_stride + filter_spatial_offset]);
                acc[acc_base + oc] += in_val * weight;
              }
            }
          }
        }
      }
    }
  }
}

// Applies TFLite's fused activation bounds (`CalculateActivationRange` +
// `ActivationFunctionWithMinMax`), including finite `[lowest(), max()]`
// clamping for `NONE` and `[0, max()]` for `RELU` so float overflows clamp to
// `+/-FLT_MAX`, matching TFLite's reference and optimized builtins.
template <typename T>
inline void ApplyTransposeConvActivation(absl::Span<T> data,
                                         tflite::ActivationFunctionType faf) {
  static_assert(std::is_floating_point_v<T>,
                "ApplyTransposeConvActivation operates in unquantized "
                "floating-point value space.");
  T min_val = std::numeric_limits<T>::lowest();
  T max_val = std::numeric_limits<T>::max();
  switch (faf) {
    case tflite::ActivationFunctionType_RELU:
      min_val = static_cast<T>(0);
      break;
    case tflite::ActivationFunctionType_RELU6:
      min_val = static_cast<T>(0);
      max_val = static_cast<T>(6);
      break;
    case tflite::ActivationFunctionType_RELU_N1_TO_1:
      min_val = static_cast<T>(-1);
      max_val = static_cast<T>(1);
      break;
    case tflite::ActivationFunctionType_NONE:
    default:
      break;
  }
  for (T& value : data) {
    value = std::min(std::max(value, min_val), max_val);
  }
}

}  // namespace detail

// Reference float NHWC 2D transposed convolution with optional per-channel bias
// and fused activation (`NONE`, `RELU`, `RELU6`, `RELU_N1_TO_1`) using TFLite's
// finite activation clamp bounds (`[-FLT_MAX, FLT_MAX]` for `NONE`,
// `[0, FLT_MAX]` for `RELU`). Supports `batch == 0` with empty input/output
// buffers after validating `faf`, filter, and optional bias. Unsupported `faf`
// values return `kLiteRtStatusErrorInvalidArgument` without modifying `output`.
// `output` must not alias `input` or `filter`.
inline LiteRtStatus ReferenceTransposeConv(absl::Span<const float> input,
                                           absl::Span<const float> filter,
                                           absl::Span<const float> bias,
                                           absl::Span<float> output,
                                           const TransposeConvParams& params,
                                           tflite::ActivationFunctionType faf) {
  Expected<TransposeConvGeometry> geom =
      detail::PrepareAndValidateTransposeConv<float>(input, filter, bias,
                                                     output, params, faf);
  if (!geom) {
    return geom.Error().Status();
  }
  const size_t total = geom->output_num_elements;
  if (total == 0) {
    return kLiteRtStatusOk;
  }
  detail::AccumulateTransposeConvUnchecked<float, float, float>(input, filter,
                                                                output, *geom);
  if (!bias.empty()) {
    const size_t out_c = static_cast<size_t>(params.out_c);
    for (size_t i = 0; i < total; ++i) {
      output[i] += bias[i % out_c];
    }
  }
  detail::ApplyTransposeConvActivation(output.first(total), faf);
  return kLiteRtStatusOk;
}

// Reference quantized NHWC 2D transposed convolution with symmetric per-tensor
// weights (`weight_zero_point == 0`, scalar `weight_scale`), `int64_t`
// accumulation for <=16-bit integer `T` and `FilterT`, optional `int32_t` /
// `uint32_t` per-channel bias, `double` effective-scale rescaling, fused
// activation (`NONE`, `RELU`, `RELU6`, `RELU_N1_TO_1`), and saturating
// requantization to `T`. Supports `batch == 0` with empty input/output buffers
// after validating `faf`, quantization parameters, filter, and optional bias.
// Unsupported `faf` or invalid quantization parameters return
// `kLiteRtStatusErrorInvalidArgument` without modifying `output`. `output` must
// not alias `input` or `filter`.
template <typename T, typename FilterT, typename BiasT>
inline LiteRtStatus ReferenceTransposeConvQuantized(
    absl::Span<const T> input, absl::Span<const FilterT> filter,
    absl::Span<const BiasT> bias, absl::Span<T> output,
    const TransposeConvParams& params, tflite::ActivationFunctionType faf,
    const AffineQuantSpec& input_quant, float weight_scale,
    const AffineQuantSpec& output_quant) {
  static_assert(detail::kIsSupportedQuantizedConvDataType<T>,
                "ReferenceTransposeConvQuantized supports <=16-bit integer "
                "activation types.");
  static_assert(detail::kIsSupportedQuantizedConvDataType<FilterT>,
                "ReferenceTransposeConvQuantized supports <=16-bit integer "
                "filter types.");
  static_assert(detail::kIsSupportedQuantizedConvBiasType<BiasT>,
                "ReferenceTransposeConvQuantized supports int32_t or uint32_t "
                "bias types.");
  if (Expected<void> valid = ValidateQuantization<T>(input_quant); !valid) {
    return valid.Error().Status();
  }
  if (Expected<void> valid = ValidateQuantization<T>(output_quant); !valid) {
    return valid.Error().Status();
  }
  if (!std::isfinite(weight_scale) || weight_scale <= 0.0f) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  Expected<TransposeConvGeometry> geom =
      detail::PrepareAndValidateTransposeConv<int64_t>(input, filter, bias,
                                                       output, params, faf);
  if (!geom) {
    return geom.Error().Status();
  }
  const size_t total = geom->output_num_elements;
  if (total == 0) {
    return kLiteRtStatusOk;
  }
  std::vector<int64_t> acc(total, 0);
  detail::AccumulateTransposeConvUnchecked<T, FilterT, int64_t>(
      input, filter, absl::MakeSpan(acc), *geom, input_quant.zero_point);
  const double effective_scale = static_cast<double>(input_quant.scale) *
                                 static_cast<double>(weight_scale);
  const size_t out_c = static_cast<size_t>(params.out_c);
  for (size_t i = 0; i < total; ++i) {
    const int64_t with_bias =
        acc[i] +
        (!bias.empty() ? static_cast<int64_t>(bias[i % out_c]) : int64_t{0});
    double real_val = static_cast<double>(with_bias) * effective_scale;
    detail::ApplyTransposeConvActivation(absl::MakeSpan(&real_val, 1), faf);
    Expected<T> quantized = QuantizeValueFromDouble<T>(real_val, output_quant);
    if (!quantized) {
      return quantized.Error().Status();
    }
    output[i] = *quantized;
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_CONVOLUTION_H_
