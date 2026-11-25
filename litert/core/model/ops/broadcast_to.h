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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/tensor_layout_util.h"

namespace litert::internal {

inline LiteRtStatus InferBroadcastTo(const LiteRtOpT& op,
                                     absl::Span<Dims> input_shapes,
                                     std::vector<Dims>& output_shapes) {
  constexpr size_t kBroadcastToMinArgs = 2;
  constexpr size_t kShapeTensorIndex = 1;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kBroadcastToMinArgs ||
      op.Inputs().size() < kBroadcastToMinArgs ||
      op.Inputs()[kShapeTensorIndex] == nullptr) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  if (output_shapes.size() != 1) {
    return kLiteRtStatusErrorShapeInferenceFailed;
  }
  const auto& shape_tensor = op.Input(kShapeTensorIndex);

  if (shape_tensor.Weights().Buffer().Size() >= kShapeTensorSize) {
    auto buf = shape_tensor.Weights().Buffer();
    // Ensure the shape tensor is a ranked tensor.
    if (shape_tensor.Type().first != kLiteRtRankedTensorType) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }

    Dims out_shape;
    auto element_type =
        shape_tensor.Type().second.ranked_tensor_type.element_type;

    // Parse the shape data from the buffer, supporting Int64 and Int32.
    if (element_type == kLiteRtElementTypeInt64 &&
        buf.Size() % sizeof(int64_t) == 0) {
      int rank = buf.Size() / sizeof(int64_t);
      for (int i = 0; i < rank; ++i) {
        int64_t val;
        std::memcpy(&val, buf.Data() + i * sizeof(int64_t), sizeof(int64_t));
        if (val < -1 || val > std::numeric_limits<int32_t>::max()) {
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        out_shape.push_back(static_cast<int32_t>(val));
      }
    } else if (element_type == kLiteRtElementTypeInt32 &&
               buf.Size() % sizeof(int32_t) == 0) {
      int rank = buf.Size() / sizeof(int32_t);
      for (int i = 0; i < rank; ++i) {
        int32_t val;
        std::memcpy(&val, buf.Data() + i * sizeof(int32_t), sizeof(int32_t));
        if (val < -1) {
          return kLiteRtStatusErrorShapeInferenceFailed;
        }
        out_shape.push_back(val);
      }
    } else {
      LITERT_LOG(LITERT_ERROR, "Invalid shape tensor type for BroadcastTo.");
      return kLiteRtStatusErrorShapeInferenceFailed;
    }

    const auto& input_shape = input_shapes[0];

    // Check if the input shape is broadcastable to the target shape.
    if (input_shape.size() > out_shape.size()) {
      return kLiteRtStatusErrorShapeInferenceFailed;
    }
    for (int i = 0; i < input_shape.size(); ++i) {
      auto in_dim = input_shape[input_shape.size() - 1 - i];
      auto out_dim = out_shape[out_shape.size() - 1 - i];
      if (in_dim != 1 && in_dim != out_dim) {
        return kLiteRtStatusErrorShapeInferenceFailed;
      }
    }

    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  return kLiteRtStatusErrorShapeInferenceFailed;
}

namespace detail {

template <typename T>
inline void ExecuteBroadcastToWithPlan(const BroadcastPlan<1>& plan,
                                       const T* input_data, T* output_data) {
  for (const auto& span : plan.Spans()) {
    const T* in_ptr = input_data + span.input_offsets[0];
    T* out_ptr = output_data + span.output_offset;
    if (span.input_steps[0] == 1) {
      std::copy_n(in_ptr, span.count, out_ptr);
    } else {
      std::fill_n(out_ptr, span.count, *in_ptr);
    }
  }
}

}  // namespace detail

// Broadcasts `input` into the caller-requested `output.dims` shape under
// trailing-dimension NumPy broadcasting (`BroadcastOutputMode::kTargetShape`).
// Validates concrete shapes, checked element/byte counts, and buffer
// capacities before reading any input; valid empty outputs return
// `kLiteRtStatusOk` without dereferencing data pointers. `output.data` must
// not alias `input.data`.
template <typename T>
inline LiteRtStatus ReferenceBroadcastTo(ConstTensorView<T> input,
                                         TensorView<T> output) {
  auto plan =
      PrepareBroadcastPlan<1>({input.dims}, {sizeof(T)}, output.dims, sizeof(T),
                              BroadcastOutputMode::kTargetShape);
  if (!plan) {
    return plan.Error().Status();
  }
  if (!ValidateBufferCapacity(input.data, plan->input_num_elements[0]) ||
      !ValidateBufferCapacity(output.data, plan->output_num_elements)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteBroadcastToWithPlan(*plan, input.data.data(),
                                     output.data.data());
  return kLiteRtStatusOk;
}

template <typename T>
inline LiteRtStatus ReferenceBroadcastTo(const T* input_data,
                                         const int32_t* input_dims,
                                         int input_rank,
                                         const int32_t* output_dims,
                                         int output_rank, T* output_data) {
  auto in_dims = MakeDimsSpan(input_dims, input_rank);
  auto out_dims = MakeDimsSpan(output_dims, output_rank);
  if (!in_dims || !out_dims) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto plan =
      PrepareBroadcastPlan<1>({*in_dims}, {sizeof(T)}, *out_dims, sizeof(T),
                              BroadcastOutputMode::kTargetShape);
  if (!plan) {
    return plan.Error().Status();
  }
  if (plan->IsEmpty()) {
    return kLiteRtStatusOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  detail::ExecuteBroadcastToWithPlan(*plan, input_data, output_data);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_BROADCAST_TO_H_
