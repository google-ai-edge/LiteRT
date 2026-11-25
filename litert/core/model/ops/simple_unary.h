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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/ops/reference_numeric_utils.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"

namespace litert::internal {

// Infers the output shape for an identity op (output shape == input shape).
inline LiteRtStatus InferIdentity(absl::Span<Dims> input_shapes,
                                  std::vector<Dims>& output_shapes) {
  if (input_shapes.empty() || output_shapes.size() != 1) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid number of input/output shapes for unary op.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  output_shapes[0] = input_shapes[0];
  return kLiteRtStatusOk;
}

#define DEFINE_SIMPLE_UNARY_INFER(name)                               \
  inline LiteRtStatus Infer##name(const LiteRtOpT& op,                \
                                  absl::Span<Dims> input_shapes,      \
                                  std::vector<Dims>& output_shapes) { \
    return InferIdentity(input_shapes, output_shapes);                \
  }

DEFINE_SIMPLE_UNARY_INFER(Abs)
DEFINE_SIMPLE_UNARY_INFER(Cast)
DEFINE_SIMPLE_UNARY_INFER(Ceil)
DEFINE_SIMPLE_UNARY_INFER(Cos)
DEFINE_SIMPLE_UNARY_INFER(Cumsum)
DEFINE_SIMPLE_UNARY_INFER(Dequantize)
DEFINE_SIMPLE_UNARY_INFER(Elu)
DEFINE_SIMPLE_UNARY_INFER(Exp)
DEFINE_SIMPLE_UNARY_INFER(Floor)
DEFINE_SIMPLE_UNARY_INFER(Gelu)
DEFINE_SIMPLE_UNARY_INFER(HardSwish)
DEFINE_SIMPLE_UNARY_INFER(L2Normalization)
DEFINE_SIMPLE_UNARY_INFER(LeakyRelu)
DEFINE_SIMPLE_UNARY_INFER(Log)
DEFINE_SIMPLE_UNARY_INFER(LogicalNot)
DEFINE_SIMPLE_UNARY_INFER(Logistic)
DEFINE_SIMPLE_UNARY_INFER(LogSoftmax)
DEFINE_SIMPLE_UNARY_INFER(Neg)
DEFINE_SIMPLE_UNARY_INFER(Quantize)
DEFINE_SIMPLE_UNARY_INFER(Relu)
DEFINE_SIMPLE_UNARY_INFER(Relu0To1)
DEFINE_SIMPLE_UNARY_INFER(Relu6)
DEFINE_SIMPLE_UNARY_INFER(ReluN1To1)
DEFINE_SIMPLE_UNARY_INFER(ReverseV2)
DEFINE_SIMPLE_UNARY_INFER(Round)
DEFINE_SIMPLE_UNARY_INFER(Rsqrt)
DEFINE_SIMPLE_UNARY_INFER(Sign)
DEFINE_SIMPLE_UNARY_INFER(Sin)
DEFINE_SIMPLE_UNARY_INFER(Softmax)
DEFINE_SIMPLE_UNARY_INFER(Sqrt)
DEFINE_SIMPLE_UNARY_INFER(Square)
DEFINE_SIMPLE_UNARY_INFER(Tanh)

#undef DEFINE_SIMPLE_UNARY_INFER

inline void ReferenceSoftmax(const float* input_data, float* output_data,
                             int batch, int depth, float beta) {
  for (int b = 0; b < batch; ++b) {
    float max_val = input_data[b * depth];
    for (int d = 1; d < depth; ++d) {
      max_val = std::max(max_val, input_data[b * depth + d]);
    }
    float sum = 0;
    for (int d = 0; d < depth; ++d) {
      sum += std::exp(beta * (input_data[b * depth + d] - max_val));
    }
    for (int d = 0; d < depth; ++d) {
      output_data[b * depth + d] =
          std::exp(beta * (input_data[b * depth + d] - max_val)) / sum;
    }
  }
}

inline void ReferenceLogSoftmax(const float* input_data, float* output_data,
                                int batch, int depth) {
  for (int b = 0; b < batch; ++b) {
    float max_val = input_data[b * depth];
    for (int d = 1; d < depth; ++d) {
      max_val = std::max(max_val, input_data[b * depth + d]);
    }
    float sum = 0;
    for (int d = 0; d < depth; ++d) {
      sum += std::exp(input_data[b * depth + d] - max_val);
    }
    float log_sum = std::log(sum);
    for (int d = 0; d < depth; ++d) {
      output_data[b * depth + d] =
          (input_data[b * depth + d] - max_val) - log_sum;
    }
  }
}

template <typename UnaryOp>
inline void ReferenceUnaryGeneric(const float* input_data, size_t num_elements,
                                  float* output_data, UnaryOp op) {
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = op(input_data[i]);
  }
}

inline void ReferenceTanh(const float* input_data, size_t num_elements,
                          float* output_data) {
  ReferenceUnaryGeneric(input_data, num_elements, output_data,
                        [](float val) { return std::tanh(val); });
}

inline void ReferenceLogistic(const float* input_data, size_t num_elements,
                              float* output_data) {
  ReferenceUnaryGeneric(input_data, num_elements, output_data, [](float val) {
    return 1.0f / (1.0f + std::exp(-val));
  });
}

inline void ReferenceRsqrt(const float* input_data, size_t num_elements,
                           float* output_data) {
  ReferenceUnaryGeneric(input_data, num_elements, output_data,
                        [](float val) { return 1.0f / std::sqrt(val); });
}

inline void ReferenceSin(const float* input_data, size_t num_elements,
                         float* output_data) {
  ReferenceUnaryGeneric(input_data, num_elements, output_data,
                        [](float val) { return std::sin(val); });
}

inline void ReferenceCos(const float* input_data, size_t num_elements,
                         float* output_data) {
  ReferenceUnaryGeneric(input_data, num_elements, output_data,
                        [](float val) { return std::cos(val); });
}

template <typename OutT, typename InT>
inline void ReferenceCast(const InT* input_data, size_t num_elements,
                          OutT* output_data) {
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = static_cast<OutT>(input_data[i]);
  }
}

template <typename InT>
inline void ReferenceDequantize(const InT* input_data, size_t num_elements,
                                float* output_data, float scale,
                                int64_t zero_point) {
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = DequantizeValue(input_data[i], scale, zero_point);
  }
}

template <typename InT>
inline LiteRtStatus ReferenceDequantizePerChannel(const InT* input_data,
                                                  const int32_t* dims, int rank,
                                                  int32_t quantized_dimension,
                                                  const float* scales,
                                                  const int64_t* zero_points,
                                                  float* output_data) {
  if (rank <= 0 || dims == nullptr || scales == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto layout = PrepareAxisLayout(absl::MakeConstSpan(dims, rank),
                                  quantized_dimension, sizeof(InT));
  if (!layout || !CheckedByteCount(layout->num_elements, sizeof(float))) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (layout->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (size_t o = 0; o < layout->outer_size; ++o) {
    for (size_t c = 0; c < layout->axis_size; ++c) {
      const float scale = scales[c];
      const int64_t zp = zero_points != nullptr ? zero_points[c] : 0;
      const size_t base = (o * layout->axis_size + c) * layout->inner_size;
      for (size_t i = 0; i < layout->inner_size; ++i) {
        output_data[base + i] =
            DequantizeValue(input_data[base + i], scale, zp);
      }
    }
  }
  return kLiteRtStatusOk;
}

inline void ReferenceGelu(const float* input_data, size_t num_elements,
                          bool approximate, float* output_data) {
  if (approximate) {
    // Tanh-based GELU approximation:
    // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
    constexpr float kGeluCubicCoeff = 0.044715f;
    const float kSqrtTwoOverPi = std::sqrt(2.0f / static_cast<float>(M_PI));
    ReferenceUnaryGeneric(
        input_data, num_elements, output_data, [kSqrtTwoOverPi](float val) {
          return 0.5f * val *
                 (1.0f + std::tanh(kSqrtTwoOverPi *
                                   (val + kGeluCubicCoeff * val * val * val)));
        });
  } else {
    ReferenceUnaryGeneric(input_data, num_elements, output_data, [](float val) {
      return 0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f)));
    });
  }
}

namespace detail {

template <typename InT, typename OutT>
inline LiteRtStatus ValidateUnaryPointers(const InT* input_data,
                                          const OutT* output_data,
                                          size_t num_elements) {
  if (!CheckedByteCount(num_elements, sizeof(InT)) ||
      !CheckedByteCount(num_elements, sizeof(OutT))) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (num_elements == 0) {
    return kLiteRtStatusOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

inline Expected<std::vector<bool>> ParseReverseV2Axes(
    size_t rank, absl::Span<const int32_t> axes) {
  std::vector<bool> reverse_dim(rank, false);
  for (int32_t raw_axis : axes) {
    const int64_t axis = raw_axis < 0 ? static_cast<int64_t>(raw_axis) +
                                            static_cast<int64_t>(rank)
                                      : static_cast<int64_t>(raw_axis);
    if (axis < 0 || axis >= static_cast<int64_t>(rank) ||
        reverse_dim[static_cast<size_t>(axis)]) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "ReverseV2 axis is out of range or duplicated");
    }
    reverse_dim[static_cast<size_t>(axis)] = true;
  }
  return reverse_dim;
}

template <typename T>
inline void ExecuteReverseV2WithLayout(const ContiguousLayout& layout,
                                       absl::Span<const int32_t> dims,
                                       const std::vector<bool>& reverse_dim,
                                       const T* input_data, T* output_data) {
  if (layout.IsEmpty()) {
    return;
  }
  if (dims.empty()) {
    output_data[0] = input_data[0];
    return;
  }
  const size_t rank = dims.size();
  for (size_t out_idx = 0; out_idx < layout.num_elements; ++out_idx) {
    size_t rem = out_idx;
    size_t in_idx = 0;
    for (size_t d = 0; d < rank; ++d) {
      const size_t coord = rem / layout.strides[d];
      rem %= layout.strides[d];
      const size_t src_coord =
          reverse_dim[d] ? (static_cast<size_t>(dims[d]) - 1 - coord) : coord;
      in_idx += src_coord * layout.strides[d];
    }
    output_data[out_idx] = input_data[in_idx];
  }
}

}  // namespace detail

// Evaluates Leaky ReLU (`x >= 0 ? x : alpha * x`) elementwise over
// `num_elements` floats. Valid empty tensors (`num_elements == 0`) return
// `kLiteRtStatusOk` without dereferencing data pointers; `input_data` and
// `output_data` may alias.
inline LiteRtStatus ReferenceLeakyRelu(const float* input_data,
                                       float* output_data, size_t num_elements,
                                       float alpha) {
  if (LiteRtStatus status =
          detail::ValidateUnaryPointers(input_data, output_data, num_elements);
      status != kLiteRtStatusOk) {
    return status;
  }
  ReferenceUnaryGeneric(
      input_data, num_elements, output_data,
      [alpha](float val) { return val >= 0.0f ? val : alpha * val; });
  return kLiteRtStatusOk;
}

// Evaluates quantized Leaky ReLU (`x >= 0 ? x : alpha * x`) elementwise with
// affine dequantization (`input_quant`), `float` arithmetic, and saturating
// requantization (`output_quant`). Valid empty tensors (`num_elements == 0`)
// return `kLiteRtStatusOk` without dereferencing data pointers; `input_data`
// and `output_data` may alias.
template <typename T>
inline LiteRtStatus ReferenceLeakyReluQuantized(
    const T* input_data, T* output_data, size_t num_elements, float alpha,
    const AffineQuantSpec& input_quant, const AffineQuantSpec& output_quant) {
  if constexpr (!std::is_floating_point_v<T>) {
    if (auto valid = ValidateQuantization<T>(output_quant); !valid) {
      return valid.Error().Status();
    }
  }
  if (LiteRtStatus status =
          detail::ValidateUnaryPointers(input_data, output_data, num_elements);
      status != kLiteRtStatusOk) {
    return status;
  }
  for (size_t i = 0; i < num_elements; ++i) {
    const float x = DequantizeValue(input_data[i], input_quant);
    const float y = x >= 0.0f ? x : alpha * x;
    auto q = QuantizeValue<T>(y, output_quant);
    if (!q) {
      return q.Error().Status();
    }
    output_data[i] = *q;
  }
  return kLiteRtStatusOk;
}

// Evaluates elementwise boolean negation (`!x`) over `num_elements` elements.
// Valid empty tensors (`num_elements == 0`) return `kLiteRtStatusOk` without
// dereferencing data pointers; `input_data` and `output_data` may alias.
inline LiteRtStatus ReferenceLogicalNot(const bool* input_data,
                                        bool* output_data,
                                        size_t num_elements) {
  if (LiteRtStatus status =
          detail::ValidateUnaryPointers(input_data, output_data, num_elements);
      status != kLiteRtStatusOk) {
    return status;
  }
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = !input_data[i];
  }
  return kLiteRtStatusOk;
}

// Requantizes `num_elements` values from `(InT, input_scale, input_zero_point)`
// to `(OutT, output_scale, output_zero_point)` via `double` dequantization and
// saturating requantization. Valid empty tensors (`num_elements == 0`) return
// `kLiteRtStatusOk` without dereferencing data pointers; `input_data` and
// `output_data` must not alias unless `InT` and `OutT` are identical.
template <typename InT, typename OutT>
inline LiteRtStatus ReferenceQuantize(const InT* input_data,
                                      size_t num_elements, float input_scale,
                                      int64_t input_zero_point,
                                      float output_scale,
                                      int64_t output_zero_point,
                                      OutT* output_data) {
  if constexpr (!std::is_floating_point_v<OutT>) {
    if (auto valid =
            ValidateQuantization<OutT>(output_scale, output_zero_point);
        !valid) {
      return valid.Error().Status();
    }
  }
  if (LiteRtStatus status =
          detail::ValidateUnaryPointers(input_data, output_data, num_elements);
      status != kLiteRtStatusOk) {
    return status;
  }
  for (size_t i = 0; i < num_elements; ++i) {
    double real_val = 0.0;
    if constexpr (std::is_floating_point_v<InT>) {
      real_val = static_cast<double>(input_data[i]);
    } else {
      real_val = (static_cast<double>(input_data[i]) -
                  static_cast<double>(input_zero_point)) *
                 static_cast<double>(input_scale);
    }
    auto q = QuantizeValueFromDouble<OutT>(real_val, output_scale,
                                           output_zero_point);
    if (!q) {
      return q.Error().Status();
    }
    output_data[i] = *q;
  }
  return kLiteRtStatusOk;
}

// Reverses elements of `input` along `axes` into `output`.
//
// Domain and preconditions:
//   - `output.dims` must equal `input.dims` (concrete, nonnegative dimensions).
//   - Negative axes in `[-rank, -1]` are normalized by adding `rank`;
//     out-of-range axes return `kLiteRtStatusErrorInvalidArgument`.
//   - Duplicate normalized axes (e.g., `{1, -1}` on a 2D tensor) return
//     `kLiteRtStatusErrorInvalidArgument` (matching TF/TFLite).
//   - Scalar inputs (`rank == 0`) require `axes` to be empty.
//   - Valid empty tensors return `kLiteRtStatusOk` after shape and axis
//     validation without dereferencing data pointers.
//   - `input.data` must not alias `output.data`.
template <typename T>
inline LiteRtStatus ReferenceReverseV2(ConstTensorView<T> input,
                                       absl::Span<const int32_t> axes,
                                       TensorView<T> output) {
  if (output.dims.size() != input.dims.size() ||
      !std::equal(input.dims.begin(), input.dims.end(), output.dims.begin())) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto layout = PrepareContiguousLayout(input.dims, sizeof(T));
  if (!layout) {
    return layout.Error().Status();
  }
  auto reverse_dim = detail::ParseReverseV2Axes(input.dims.size(), axes);
  if (!reverse_dim) {
    return reverse_dim.Error().Status();
  }
  if (!ValidateBufferCapacity(input.data, layout->num_elements) ||
      !ValidateBufferCapacity(output.data, layout->num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteReverseV2WithLayout(*layout, input.dims, *reverse_dim,
                                     input.data.data(), output.data.data());
  return kLiteRtStatusOk;
}

// Raw-pointer overload of `ReferenceReverseV2` assuming `output_data` has the
// same shape `(input_dims, rank)` as `input_data`.
template <typename T>
inline LiteRtStatus ReferenceReverseV2(const T* input_data,
                                       const int32_t* input_dims, int rank,
                                       const int32_t* axes, int num_axes,
                                       T* output_data) {
  auto dims_span = MakeDimsSpan(input_dims, rank);
  auto axes_span = MakeDimsSpan(axes, num_axes);
  if (!dims_span || !axes_span) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto layout = PrepareContiguousLayout(*dims_span, sizeof(T));
  if (!layout) {
    return layout.Error().Status();
  }
  auto reverse_dim = detail::ParseReverseV2Axes(dims_span->size(), *axes_span);
  if (!reverse_dim) {
    return reverse_dim.Error().Status();
  }
  if (layout->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteReverseV2WithLayout(*layout, *dims_span, *reverse_dim,
                                     input_data, output_data);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_
