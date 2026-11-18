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
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
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
inline void ReferenceDequantizePerChannel(const InT* input_data,
                                          const int32_t* dims, int rank,
                                          int32_t quantized_dimension,
                                          const float* scales,
                                          const int64_t* zero_points,
                                          float* output_data) {
  if (rank <= 0 || dims == nullptr || scales == nullptr) {
    return;
  }
  auto layout = PrepareAxisLayout(absl::MakeConstSpan(dims, rank),
                                  quantized_dimension, sizeof(InT));
  if (!layout || !CheckedByteCount(layout->num_elements, sizeof(float))) {
    return;
  }
  if (layout->IsEmpty()) {
    return;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return;
  }
  for (size_t o = 0; o < layout->outer_size; ++o) {
    for (size_t c = 0; c < layout->axis_size; ++c) {
      const float scale = scales[c];
      const float zp =
          zero_points != nullptr ? static_cast<float>(zero_points[c]) : 0.0f;
      const size_t base = (o * layout->axis_size + c) * layout->inner_size;
      for (size_t i = 0; i < layout->inner_size; ++i) {
        output_data[base + i] =
            (static_cast<float>(input_data[base + i]) - zp) * scale;
      }
    }
  }
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

inline void ReferenceLeakyRelu(const float* input_data, float* output_data,
                               size_t num_elements, float alpha) {
  ReferenceUnaryGeneric(
      input_data, num_elements, output_data,
      [alpha](float val) { return val >= 0.0f ? val : alpha * val; });
}

template <typename T>
inline void ReferenceLeakyReluQuantized(const T* input_data, T* output_data,
                                        size_t num_elements, float alpha,
                                        const AffineQuantSpec& input_quant,
                                        const AffineQuantSpec& output_quant) {
  for (size_t i = 0; i < num_elements; ++i) {
    const float x = DequantizeValue(input_data[i], input_quant);
    const float y = x >= 0.0f ? x : alpha * x;
    output_data[i] = QuantizeValue<T>(y, output_quant);
  }
}

inline void ReferenceLogicalNot(const bool* input_data, bool* output_data,
                                size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = !input_data[i];
  }
}

template <typename InT, typename OutT>
inline void ReferenceQuantize(const InT* input_data, size_t num_elements,
                              float input_scale, int64_t input_zero_point,
                              float output_scale, int64_t output_zero_point,
                              OutT* output_data) {
  for (size_t i = 0; i < num_elements; ++i) {
    const float real_val =
        DequantizeValue(input_data[i], input_scale, input_zero_point);
    output_data[i] =
        QuantizeValue<OutT>(real_val, output_scale, output_zero_point);
  }
}

template <typename T>
inline LiteRtStatus ReferenceReverseV2(const T* input_data,
                                       const int32_t* input_dims, int rank,
                                       const int32_t* axes, int num_axes,
                                       T* output_data) {
  if (rank < 0 || num_axes < 0 || (num_axes > 0 && axes == nullptr)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (rank == 0) {
    if (num_axes > 0 || input_data == nullptr || output_data == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    output_data[0] = input_data[0];
    return kLiteRtStatusOk;
  }
  if (input_dims == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto layout =
      PrepareContiguousLayout(absl::MakeConstSpan(input_dims, rank), sizeof(T));
  if (!layout) {
    return layout.Error().Status();
  }
  std::vector<bool> reverse_dim(rank, false);
  for (int i = 0; i < num_axes; ++i) {
    int32_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
    if (axis < 0 || axis >= rank) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    reverse_dim[axis] = true;
  }
  if (layout->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (size_t out_idx = 0; out_idx < layout->num_elements; ++out_idx) {
    size_t rem = out_idx;
    size_t in_idx = 0;
    for (int d = 0; d < rank; ++d) {
      const size_t coord = rem / layout->strides[d];
      rem %= layout->strides[d];
      const size_t src_coord =
          reverse_dim[d] ? (static_cast<size_t>(input_dims[d]) - 1 - coord)
                         : coord;
      in_idx += src_coord * layout->strides[d];
    }
    output_data[out_idx] = input_data[in_idx];
  }
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_UNARY_H_
