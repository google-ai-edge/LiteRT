/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// Shared helpers for extending kernels to accept asymmetric uint16
// activations without touching the kernel core.

#ifndef TENSORFLOW_LITE_KERNELS_UINT16_ASYM_WRAPPER_H_
#define TENSORFLOW_LITE_KERNELS_UINT16_ASYM_WRAPPER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "tflite/core/c/common.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/tensor_ctypes.h"
#include "tflite/kernels/internal/types.h"
#include "tflite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace uint16_asym {

// Extract (scale, zero_point) from a TfLiteAffineQuantization, defaulting
// to (1.0f, 0) when any field is missing so callers cannot deref null.
inline void GetAffineParams(const TfLiteTensor* t, float* scale, int32_t* zp) {
  const auto* p = reinterpret_cast<const TfLiteAffineQuantization*>(
      t->quantization.params);
  *scale = (p && p->scale && p->scale->size > 0) ? p->scale->data[0] : 1.0f;
  *zp = (p && p->zero_point && p->zero_point->size > 0)
            ? p->zero_point->data[0]
            : 0;
}

// Dequantize a uint16 raw tensor to float using affine (raw - zp) * scale.
inline void DequantizeUInt16(const TfLiteTensor* input,
                             std::vector<float>* out) {
  const int n = GetTensorShape(input).FlatSize();
  out->resize(n);
  float scale;
  int32_t zp;
  GetAffineParams(input, &scale, &zp);
  const uint16_t* raw = GetTensorData<uint16_t>(input);
  for (int i = 0; i < n; ++i) {
    (*out)[i] = (static_cast<int32_t>(raw[i]) - zp) * scale;
  }
}

// Requantize a float tensor to uint16 using affine round(y / scale + zp).
inline void RequantizeToUInt16(const std::vector<float>& in,
                               TfLiteTensor* output) {
  float scale;
  int32_t zp;
  GetAffineParams(output, &scale, &zp);
  const float inv_scale = 1.0f / scale;
  uint16_t* raw = GetTensorData<uint16_t>(output);
  const int n = GetTensorShape(output).FlatSize();
  for (int i = 0; i < n; ++i) {
    const int32_t q = static_cast<int32_t>(
        std::round(in[i] * inv_scale + static_cast<float>(zp)));
    raw[i] = static_cast<uint16_t>(std::min(65535, std::max(0, q)));
  }
}

// Elementwise apply: dequantize input, apply float unary op, requantize
// into output.  Both tensors must have TfLiteAffineQuantization params.
inline TfLiteStatus EvalUInt16Elementwise(
    const TfLiteTensor* input, TfLiteTensor* output,
    const std::function<float(float)>& op) {
  std::vector<float> tmp;
  DequantizeUInt16(input, &tmp);
  for (float& v : tmp) v = op(v);
  RequantizeToUInt16(tmp, output);
  return kTfLiteOk;
}

// Elementwise binary apply (two uint16 inputs → uint16 output). Supports
// NumPy-style broadcasting via the standard NdArrayDesc<4> preprocessor.
inline TfLiteStatus EvalUInt16Binary(
    const TfLiteTensor* input1, const TfLiteTensor* input2,
    TfLiteTensor* output, const std::function<float(float, float)>& op) {
  std::vector<float> x, y;
  DequantizeUInt16(input1, &x);
  DequantizeUInt16(input2, &y);
  const RuntimeShape s1 = GetTensorShape(input1);
  const RuntimeShape s2 = GetTensorShape(input2);
  const RuntimeShape sout = GetTensorShape(output);
  const int n_out = sout.FlatSize();
  std::vector<float> z(n_out);
  if (s1.DimensionsCount() == sout.DimensionsCount() &&
      s2.DimensionsCount() == sout.DimensionsCount() &&
      s1.FlatSize() == n_out && s2.FlatSize() == n_out) {
    for (int i = 0; i < n_out; ++i) z[i] = op(x[i], y[i]);
  } else {
    NdArrayDesc<4> d1, d2;
    NdArrayDescsForElementwiseBroadcast(s1, s2, &d1, &d2);
    const RuntimeShape extended_output =
        RuntimeShape::ExtendedShape(4, sout);
    for (int b = 0; b < extended_output.Dims(0); ++b) {
      for (int h = 0; h < extended_output.Dims(1); ++h) {
        for (int w = 0; w < extended_output.Dims(2); ++w) {
          for (int c = 0; c < extended_output.Dims(3); ++c) {
            const int i1 = SubscriptToIndex(d1, b, h, w, c);
            const int i2 = SubscriptToIndex(d2, b, h, w, c);
            z[Offset(extended_output, b, h, w, c)] = op(x[i1], y[i2]);
          }
        }
      }
    }
  }
  RequantizeToUInt16(z, output);
  return kTfLiteOk;
}

// Elementwise compare (two uint16 inputs → bool output). Same broadcast
// semantics as EvalUInt16Binary.
inline TfLiteStatus EvalUInt16Compare(
    const TfLiteTensor* input1, const TfLiteTensor* input2,
    TfLiteTensor* output, const std::function<bool(float, float)>& op) {
  std::vector<float> x, y;
  DequantizeUInt16(input1, &x);
  DequantizeUInt16(input2, &y);
  bool* dst = GetTensorData<bool>(output);
  const RuntimeShape s1 = GetTensorShape(input1);
  const RuntimeShape s2 = GetTensorShape(input2);
  const RuntimeShape sout = GetTensorShape(output);
  const int n_out = sout.FlatSize();
  if (s1.DimensionsCount() == sout.DimensionsCount() &&
      s2.DimensionsCount() == sout.DimensionsCount() &&
      s1.FlatSize() == n_out && s2.FlatSize() == n_out) {
    for (int i = 0; i < n_out; ++i) dst[i] = op(x[i], y[i]);
  } else {
    NdArrayDesc<4> d1, d2;
    NdArrayDescsForElementwiseBroadcast(s1, s2, &d1, &d2);
    const RuntimeShape extended_output =
        RuntimeShape::ExtendedShape(4, sout);
    for (int b = 0; b < extended_output.Dims(0); ++b) {
      for (int h = 0; h < extended_output.Dims(1); ++h) {
        for (int w = 0; w < extended_output.Dims(2); ++w) {
          for (int c = 0; c < extended_output.Dims(3); ++c) {
            const int i1 = SubscriptToIndex(d1, b, h, w, c);
            const int i2 = SubscriptToIndex(d2, b, h, w, c);
            dst[Offset(extended_output, b, h, w, c)] = op(x[i1], y[i2]);
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

// Pre-shift a uint16 input tensor into an int16 signed view (raw - 32768)
// so it can be routed through an int16 integer kernel. Returns the pointer
// to feed the kernel, plus the adjusted input_offset such that
// (signed_view[i] + adjusted_offset) == real_zero_recentered value.
// For non-uint16 inputs the tensor's own int16 buffer is returned and
// adjusted_offset = -zero_point.
inline const int16_t* PreshiftUInt16ToInt16(const TfLiteTensor* input,
                                            std::vector<int16_t>* scratch,
                                            int32_t* adjusted_input_offset) {
  if (input->type == kTfLiteUInt16) {
    const auto* raw = reinterpret_cast<const uint16_t*>(input->data.data);
    const int n = GetTensorShape(input).FlatSize();
    scratch->resize(n);
    for (int i = 0; i < n; ++i) {
      (*scratch)[i] =
          static_cast<int16_t>(static_cast<int32_t>(raw[i]) - 32768);
    }
    *adjusted_input_offset = -(input->params.zero_point - 32768);
    return scratch->data();
  }
  *adjusted_input_offset = -input->params.zero_point;
  return GetTensorData<int16_t>(input);
}

}  // namespace uint16_asym
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_UINT16_ASYM_WRAPPER_H_
