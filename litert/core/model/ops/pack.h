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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_OPS_PACK_H_
#define ODML_LITERT_LITERT_CORE_MODEL_OPS_PACK_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferPack(const LiteRtOpT& op,
                              absl::Span<const Dims> input_shapes,
                              std::vector<Dims>& output_shapes) {
  if (input_shapes.empty()) return kLiteRtStatusErrorInvalidArgument;

  const auto& opts = GetTflOptions(op);
  const auto* pack_opts = opts.AsPackOptions();
  constexpr int32_t kDefaultAxis = 0;
  int32_t axis = pack_opts ? pack_opts->axis : kDefaultAxis;

  constexpr size_t kFirstInputIndex = 0;
  const auto& s0 = input_shapes[kFirstInputIndex];
  int rank = s0.size();
  if (axis < 0) axis += rank + 1;

  Dims out_shape = s0;
  out_shape.insert(out_shape.begin() + axis, input_shapes.size());

  // Verify all inputs match s0
  for (size_t i = 1; i < input_shapes.size(); ++i) {
    if (input_shapes[i].size() != rank)
      return kLiteRtStatusErrorInvalidArgument;
  }

  output_shapes[0] = std::move(out_shape);
  return kLiteRtStatusOk;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_PACK_H_
