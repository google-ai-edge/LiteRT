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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferPad(const LiteRtOpT& op,
                             absl::Span<const Dims> input_shapes,
                             std::vector<Dims>& output_shapes) {
  constexpr size_t kPadMinArgs = 2;
  constexpr size_t kInputArgIndex = 0;
  constexpr size_t kPaddingsArgIndex = 1;
  constexpr size_t kPairsPerDim = 2;
  constexpr size_t kShapeTensorSize = sizeof(int32_t);

  if (input_shapes.size() < kPadMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& paddings_tensor = op.Input(kPaddingsArgIndex);

  if (paddings_tensor.Weights().Buffer().Size() < kShapeTensorSize) {
    // Dynamic padding.
    output_shapes[0] = Dims(input_shape.size(), -1);
    return kLiteRtStatusErrorUnsupported;
  }

  auto buf = paddings_tensor.Weights().Buffer();
  // Padding tensor is [Rank, 2] int32 or int64.
  const size_t num_elements = input_shape.size() * kPairsPerDim;
  std::vector<int64_t> paddings(num_elements);

  if (buf.Size() == num_elements * sizeof(int32_t)) {
    const uint8_t* data = buf.Data();
    for (size_t i = 0; i < num_elements; ++i) {
      int32_t val;
      std::memcpy(&val, data + i * sizeof(int32_t), sizeof(int32_t));
      paddings[i] = val;
    }
  } else if (buf.Size() == num_elements * sizeof(int64_t)) {
    std::memcpy(paddings.data(), buf.Data(), num_elements * sizeof(int64_t));
  } else {
    return kLiteRtStatusErrorInvalidArgument;
  }

  Dims out_shape = input_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    int64_t before = paddings[i * kPairsPerDim];
    int64_t after = paddings[i * kPairsPerDim + 1];
    if (out_shape[i] != -1) {
      out_shape[i] += before + after;
    }
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

inline LiteRtStatus InferPadv2(const LiteRtOpT& op,
                               absl::Span<const Dims> input_shapes,
                               std::vector<Dims>& output_shapes) {
  return InferPad(op, input_shapes, output_shapes);
}

inline LiteRtStatus InferMirrorPad(const LiteRtOpT& op,
                                   absl::Span<const Dims> input_shapes,
                                   std::vector<Dims>& output_shapes) {
  return InferPad(op, input_shapes, output_shapes);
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_PAD_H_
