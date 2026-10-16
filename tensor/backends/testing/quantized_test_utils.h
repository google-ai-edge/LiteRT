/* Copyright 2026 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_QUANTIZED_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_QUANTIZED_TEST_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

// Helpers to build quantized weights and float references for backend tests.
namespace litert::tensor::quantized_test_utils {

// Zero points of a quantized test matrix.
enum class ZeroPoints { kNone, kSingle, kPerScale };

// A quantized rank 2 constant [rows, cols], together with its dequantized
// values for the float references.
struct QuantizedMatrix {
  Type type;
  int rows;
  int cols;
  // One integer per element.
  std::vector<int8_t> values;
  std::shared_ptr<Quantization> quantization;
  // Row-major, [rows, cols].
  std::vector<float> dequantized;

  float At(int r, int c) const { return dequantized[r * cols + c]; }

  template <class Tag>
  Tensor<Tag> ToTensor() const {
    return Tensor<Tag>({.name = "weights",
                        .type = type,
                        .shape = {rows, cols},
                        .buffer = values,
                        .quantization = quantization});
  }
};

// Builds a quantized [rows, cols] matrix whose channels are
// `quantized_dimension`. `block_size` 0 means per-channel quantization, and
// `per_tensor` uses a single scale instead.
inline QuantizedMatrix MakeQuantizedMatrix(
    Type type, int rows, int cols, int quantized_dimension, int block_size,
    bool per_tensor = false, ZeroPoints zero_points = ZeroPoints::kNone) {
  QuantizedMatrix m{.type = type, .rows = rows, .cols = cols};
  const int bits = static_cast<int>(BitSize(type));
  // The integers are in the symmetric range [-largest, largest] of `type`, as
  // the TFLite quantization spec requires for weights: its int8 kernels can
  // overflow on -128.
  const int largest = (1 << (bits - 1)) - 1;
  const int num_values = 2 * largest + 1;
  // They walk through that range with strides that are coprime with its size
  // (3, 15 or 255), so that they cover it. The row stride makes up to
  // `num_values` consecutive rows differ, so that a backend that mixes up rows
  // or channels fails.
  constexpr int kColumnStride = 7;
  constexpr int kRowStride = 11;
  constexpr int kOffset = 3;
  m.values.resize(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      m.values[r * cols + c] = static_cast<int8_t>(
          -largest +
          (r * kRowStride + c * kColumnStride + kOffset) % num_values);
    }
  }

  const int channels = quantized_dimension == 0 ? rows : cols;
  const int blocked = quantized_dimension == 0 ? cols : rows;
  const int num_blocks = block_size == 0 ? 1 : blocked / block_size;
  const int num_scales = per_tensor ? 1 : channels * num_blocks;
  // The scales take 5 values between 0.5 / largest and 1 / largest, so that
  // the dequantized values are within about [-1, 1] and neighboring channels
  // and blocks have different scales.
  constexpr int kNumDistinctScales = 5;
  constexpr float kMinScale = 0.5f;
  constexpr float kScaleStep = 0.125f;
  std::vector<float> scales(num_scales);
  for (int i = 0; i < num_scales; ++i) {
    scales[i] = (kMinScale + kScaleStep * (i % kNumDistinctScales)) / largest;
  }
  std::vector<int64_t> zps;
  switch (zero_points) {
    case ZeroPoints::kNone:
      break;
    case ZeroPoints::kSingle:
      zps = {1};
      break;
    case ZeroPoints::kPerScale:
      // -1, 0, 1, -1, ...
      for (int i = 0; i < num_scales; ++i) {
        zps.push_back(i % 3 - 1);
      }
      break;
  }

  m.dequantized.resize(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int channel = quantized_dimension == 0 ? r : c;
      const int position = quantized_dimension == 0 ? c : r;
      int index = 0;
      if (!per_tensor) {
        index = block_size == 0 ? channel
                                : channel * num_blocks + position / block_size;
      }
      const int64_t zp = zps.empty()       ? 0
                         : zps.size() == 1 ? zps[0]
                                           : zps[index];
      m.dequantized[r * cols + c] =
          (m.values[r * cols + c] - zp) * scales[index];
    }
  }

  if (block_size == 0) {
    m.quantization = std::make_shared<PerChannelAffineQuantization>(
        scales, zps, quantized_dimension);
  } else {
    m.quantization = std::make_shared<BlockwiseQuantization>(
        scales, zps, block_size, quantized_dimension);
  }
  return m;
}

// Generates `size` deterministic values in [-2, 2].
inline std::vector<float> MakeInput(int size) {
  // Avoid a step that would make values repeat "too early".
  constexpr float kStep = 0.37f;
  constexpr float kAmplitude = 2.0f;
  std::vector<float> data(size);
  for (int i = 0; i < size; ++i) {
    data[i] = kAmplitude * std::sin(kStep * i);
  }
  return data;
}

// Returns `a[rows, K] . b`, where `b(k, n)` returns the element of the
// right-hand side, and the tolerance for a backend's result.
//
// Backends may quantize each row of `a` to int8 dynamically. That changes each
// element of the row by up to half a quantization step, i.e. by up to
// max_k |a(m, k)| / 254 (symmetric) or / 255 (asymmetric), and each result by
// up to that times sum_k |b(k, n)|. The tolerance is twice the largest bound.
template <class B>
std::pair<std::vector<float>, float> MatMul(absl::Span<const float> a, int rows,
                                            int k_size, int n_size, B b) {
  constexpr float kInt8Steps = 127.f;
  std::vector<float> result(rows * n_size);
  float tolerance = 1e-5f;
  for (int m = 0; m < rows; ++m) {
    float row_max_abs = 0.f;
    for (int k = 0; k < k_size; ++k) {
      row_max_abs = std::max(row_max_abs, std::abs(a[m * k_size + k]));
    }
    for (int n = 0; n < n_size; ++n) {
      float sum = 0.f;
      float sum_abs_b = 0.f;
      for (int k = 0; k < k_size; ++k) {
        sum += a[m * k_size + k] * b(k, n);
        sum_abs_b += std::abs(b(k, n));
      }
      result[m * n_size + n] = sum;
      tolerance = std::max(tolerance, row_max_abs / kInt8Steps * sum_abs_b);
    }
  }
  return {result, tolerance};
}

}  // namespace litert::tensor::quantized_test_utils

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_TESTING_QUANTIZED_TEST_UTILS_H_
