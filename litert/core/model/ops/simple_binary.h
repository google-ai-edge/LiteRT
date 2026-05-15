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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "tflite/converter/schema/schema_generated.h"
#include "tflite/kernels/internal/runtime_shape.h"

namespace litert::internal {

inline LiteRtStatus InferElementwiseBinary(absl::Span<Dims> input_shapes,
                                           std::vector<Dims>& output_shapes) {
  if (input_shapes.size() != 2) {
    LITERT_LOG(LITERT_ERROR, "Invalid number of input shapes for binary op.");
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& s1 = input_shapes[0];
  const auto& s2 = input_shapes[1];

  size_t rank1 = s1.size();
  size_t rank2 = s2.size();
  size_t max_rank = std::max(rank1, rank2);
  Dims out_shape(max_rank);

  for (int i = 0; i < max_rank; ++i) {
    // Reverse index
    int idx1 = rank1 - 1 - i;
    int idx2 = rank2 - 1 - i;
    int32_t d1 = (idx1 >= 0) ? s1[idx1] : 1;
    int32_t d2 = (idx2 >= 0) ? s2[idx2] : 1;

    if (d1 == -1 || d2 == -1) {
      out_shape[max_rank - 1 - i] = -1;
    } else if (d1 == d2) {
      out_shape[max_rank - 1 - i] = d1;
    } else if (d1 == 1) {
      out_shape[max_rank - 1 - i] = d2;
    } else if (d2 == 1) {
      out_shape[max_rank - 1 - i] = d1;
    } else {
      LITERT_LOG(LITERT_ERROR, "Incompatible dims for binary op.");
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

#define DEFINE_SIMPLE_BINARY_INFER(name)                              \
  inline LiteRtStatus Infer##name(const LiteRtOpT& op,                \
                                  absl::Span<Dims> input_shapes,      \
                                  std::vector<Dims>& output_shapes) { \
    return InferElementwiseBinary(input_shapes, output_shapes);       \
  }

DEFINE_SIMPLE_BINARY_INFER(Add)
DEFINE_SIMPLE_BINARY_INFER(Div)
DEFINE_SIMPLE_BINARY_INFER(Equal)
DEFINE_SIMPLE_BINARY_INFER(FloorDiv)
DEFINE_SIMPLE_BINARY_INFER(Greater)
DEFINE_SIMPLE_BINARY_INFER(GreaterEqual)
DEFINE_SIMPLE_BINARY_INFER(Less)
DEFINE_SIMPLE_BINARY_INFER(LessEqual)
DEFINE_SIMPLE_BINARY_INFER(LogicalAnd)
DEFINE_SIMPLE_BINARY_INFER(LogicalOr)
DEFINE_SIMPLE_BINARY_INFER(Maximum)
DEFINE_SIMPLE_BINARY_INFER(Minimum)
DEFINE_SIMPLE_BINARY_INFER(Mul)
DEFINE_SIMPLE_BINARY_INFER(NotEqual)
DEFINE_SIMPLE_BINARY_INFER(Pow)
DEFINE_SIMPLE_BINARY_INFER(Prelu)
DEFINE_SIMPLE_BINARY_INFER(SquaredDifference)
DEFINE_SIMPLE_BINARY_INFER(Sub)

#undef DEFINE_SIMPLE_BINARY_INFER

inline void ComputeBroadcastStrides(const int32_t* dims, int input_rank,
                                    int output_rank, size_t* strides) {
  size_t current_stride = 1;
  for (int i = output_rank - 1; i >= 0; --i) {
    int idx = i - (output_rank - input_rank);
    int32_t dim = (idx >= 0) ? dims[idx] : 1;
    strides[i] = (dim == 1) ? 0 : current_stride;
    current_stride *= dim;
  }
}

template <typename T, typename BinaryOp>
void RunBinaryOp(const T* a, const T* b, T* output, const size_t* a_stride,
                 const size_t* b_stride, const size_t* output_stride,
                 const size_t* output_shape, int rank, BinaryOp op) {
  if (rank <= 0) {
    *output = op(*a, *b);
  } else if (rank == 1) {
    for (size_t i = 0; i < output_shape[0]; ++i) {
      output[i * output_stride[0]] = op(a[i * a_stride[0]], b[i * b_stride[0]]);
    }
  } else {
    for (size_t i = 0; i < output_shape[0]; ++i) {
      RunBinaryOp(a + i * a_stride[0], b + i * b_stride[0],
                  output + i * output_stride[0], a_stride + 1, b_stride + 1,
                  output_stride + 1, output_shape + 1, rank - 1, op);
    }
  }
}

template <typename T, typename BinaryOp>
inline void ReferenceBinaryGeneric(const T* input1_data,
                                   const int32_t* input1_dims, int input1_rank,
                                   const T* input2_data,
                                   const int32_t* input2_dims, int input2_rank,
                                   T* output_data, const int32_t* output_dims,
                                   int rank, BinaryOp op) {
  constexpr int kMaxRank = tflite::RuntimeShape::kMaxSmallSize;
  size_t a_stride[kMaxRank];
  size_t b_stride[kMaxRank];
  size_t o_stride[kMaxRank];
  size_t o_shape[kMaxRank];

  ComputeBroadcastStrides(input1_dims, input1_rank, rank, a_stride);
  ComputeBroadcastStrides(input2_dims, input2_rank, rank, b_stride);

  size_t current_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    o_stride[i] = current_stride;
    o_shape[i] = output_dims[i];
    current_stride *= output_dims[i];
  }

  RunBinaryOp(input1_data, input2_data, output_data, a_stride, b_stride,
              o_stride, o_shape, rank, op);
}

inline void ApplyActivation(float* output_data, size_t num_elements,
                            tflite::ActivationFunctionType faf) {
  if (faf == tflite::ActivationFunctionType_RELU) {
    for (size_t i = 0; i < num_elements; ++i) {
      output_data[i] = std::max(0.0f, output_data[i]);
    }
  } else if (faf == tflite::ActivationFunctionType_RELU6) {
    for (size_t i = 0; i < num_elements; ++i) {
      output_data[i] = std::min(6.0f, std::max(0.0f, output_data[i]));
    }
  }
}


}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_SIMPLE_BINARY_H_
