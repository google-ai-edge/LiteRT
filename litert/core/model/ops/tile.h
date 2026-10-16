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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_TILE_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_TILE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "SafeInt.hpp"  // from @SafeInt
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferTile(const ShapeInferenceContext& ctx,
                              InferenceResult& result) {
  constexpr size_t kTileMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kMultiplesArgIndex = 1;

  if (ctx.GetNumInputs() < kTileMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }

  const Dims input_shape = ctx.GetInputShape(kInputArgIndex);
  const Dims multiples_shape = ctx.GetInputShape(kMultiplesArgIndex);

  if (!multiples_shape.empty()) {
    if (multiples_shape.size() != 1 ||
        (multiples_shape[0] != -1 &&
         multiples_shape[0] != static_cast<int32_t>(input_shape.size()))) {
      return kLiteRtStatusErrorInvalidArgument;
    }
  }

  const LiteRtElementType mult_type =
      ctx.GetInputElementType(kMultiplesArgIndex);
  if (mult_type != kLiteRtElementTypeInt32 &&
      mult_type != kLiteRtElementTypeInt64 &&
      mult_type != kLiteRtElementTypeNone) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  absl::Span<const uint8_t> buf = ctx.GetInputData(kMultiplesArgIndex);
  if (!buf.empty()) {
    const size_t rank = input_shape.size();
    const bool is_int64 =
        mult_type == kLiteRtElementTypeInt64 ||
        (mult_type == kLiteRtElementTypeNone && rank > 0 &&
         buf.size() == rank * sizeof(int64_t));
    std::vector<int64_t> mults(rank);
    if (is_int64) {
      if (buf.size() != rank * sizeof(int64_t)) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      auto data = absl::MakeConstSpan(
          reinterpret_cast<const int64_t*>(buf.data()), rank);
      mults.assign(data.begin(), data.end());
    } else {
      if (buf.size() != rank * sizeof(int32_t)) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      auto data = absl::MakeConstSpan(
          reinterpret_cast<const int32_t*>(buf.data()), rank);
      for (size_t i = 0; i < rank; ++i) {
        mults[i] = data[i];
      }
    }

    Dims out_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      const int64_t m = mults[i];
      const int64_t in_dim = input_shape[i];
      if (m < 0 || m > std::numeric_limits<int32_t>::max() || in_dim < -1) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (m == 0 || in_dim == 0) {
        out_shape[i] = 0;
      } else if (in_dim == -1) {
        out_shape[i] = -1;
      } else if (!SafeMultiply(input_shape[i], m, out_shape[i])) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
    result.output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  result.output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferTile(const LiteRtOpT& op,
                              absl::Span<const Dims> input_shapes,
                              std::vector<Dims>& output_shapes) {
  constexpr size_t kTileMinArgs = 2;
  constexpr size_t kMultiplesArgIndex = 1;
  if (input_shapes.size() < kTileMinArgs || output_shapes.size() != 1 ||
      op.Inputs().size() < kTileMinArgs ||
      op.Inputs()[kMultiplesArgIndex] == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  StandaloneShapeInferenceContext ctx(op, input_shapes);
  InferenceResult result;
  result.output_shapes = std::move(output_shapes);
  const LiteRtStatus status = InferTile(ctx, result);
  output_shapes = std::move(result.output_shapes);
  return status;
}

template <typename T>
inline void ReferenceTile(const T* input_data, const int32_t* input_dims,
                          const int32_t* multiples, int rank, T* output_data) {
  if (rank <= 0) return;
  constexpr int kMaxRank = 8;
  ABSL_CHECK_LE(rank, kMaxRank);

  int64_t in_strides[kMaxRank];
  int64_t out_strides[kMaxRank];
  int64_t out_dims[kMaxRank];

  for (int i = 0; i < rank; ++i) {
    if (input_dims[i] <= 0 || multiples[i] <= 0) return;
    out_dims[i] = static_cast<int64_t>(input_dims[i]) * multiples[i];
  }

  in_strides[rank - 1] = 1;
  out_strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_dims[i + 1];
    out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
  }

  int64_t total_out_elements = out_strides[0] * out_dims[0];

  for (int64_t out_idx = 0; out_idx < total_out_elements; ++out_idx) {
    int64_t temp = out_idx;
    int64_t in_idx = 0;
    for (int i = 0; i < rank; ++i) {
      int64_t coord = temp / out_strides[i];
      temp %= out_strides[i];
      int64_t in_coord = coord % input_dims[i];
      in_idx += in_coord * in_strides[i];
    }
    output_data[out_idx] = input_data[in_idx];
  }
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TILE_H_
