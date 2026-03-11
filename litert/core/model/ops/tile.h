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
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferTile(const LiteRtOpT& op,
                              absl::Span<const Dims> input_shapes,
                              std::vector<Dims>& output_shapes) {
  constexpr int kTileMinArgs = 2;
  constexpr int kInputArgIndex = 0;
  constexpr int kMultiplesArgIndex = 1;

  if (input_shapes.size() < kTileMinArgs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& input_shape = input_shapes[kInputArgIndex];
  const auto& multiples_tensor = op.Input(kMultiplesArgIndex);

  if (multiples_tensor.Weights().Buffer().Size() > 0) {
    auto buf = multiples_tensor.Weights().Buffer();
    const int32_t* mults = reinterpret_cast<const int32_t*>(buf.Data());
    int mult_rank = buf.Size() / sizeof(int32_t);
    if (mult_rank != input_shape.size())
      return kLiteRtStatusErrorInvalidArgument;

    Dims out_shape = input_shape;
    for (int i = 0; i < mult_rank; ++i) {
      if (out_shape[i] != -1) out_shape[i] *= mults[i];
    }
    output_shapes[0] = std::move(out_shape);
    return kLiteRtStatusOk;
  }

  output_shapes[0] = Dims(input_shape.size(), -1);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_TILE_H_
