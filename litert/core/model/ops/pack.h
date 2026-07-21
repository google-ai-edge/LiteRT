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
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference_types.h"

namespace litert::internal {

inline LiteRtStatus InferPack(const ShapeInferenceContext& ctx,
                              InferenceResult& result) {
  const size_t num_inputs = ctx.GetNumInputs();
  if (num_inputs == 0) return kLiteRtStatusErrorInvalidArgument;

  const auto& opts = ctx.GetOptions();
  const auto* pack_opts = opts.AsPackOptions();
  constexpr int32_t kDefaultAxis = 0;
  int32_t axis = pack_opts ? pack_opts->axis : kDefaultAxis;

  constexpr size_t kFirstInputIndex = 0;
  Dims element_shape = ctx.GetInputShape(kFirstInputIndex);
  int rank = element_shape.size();
  if (axis < 0) axis += rank + 1;
  if (axis < 0 || axis > rank) return kLiteRtStatusErrorInvalidArgument;

  // Unify across all inputs to catch mismatches and resolve dynamic -1
  // dimensions.
  for (size_t i = 1; i < num_inputs; ++i) {
    const auto& si = ctx.GetInputShape(i);
    if (si.size() != static_cast<size_t>(rank)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    for (size_t j = 0; j < static_cast<size_t>(rank); ++j) {
      if (element_shape[j] == -1 && si[j] != -1) {
        element_shape[j] = si[j];
      } else if (element_shape[j] != -1 && si[j] != -1 &&
                 element_shape[j] != si[j]) {
        return kLiteRtStatusErrorInvalidArgument;
      }
    }
  }

  Dims out_shape = element_shape;
  out_shape.insert(out_shape.begin() + axis, num_inputs);

  if (result.output_shapes.empty()) {
    result.output_shapes.resize(1);
  }
  result.output_shapes[0] = std::move(out_shape);

  // Check if all input data buffers are statically available and uniform.
  bool all_static = true;
  size_t elem_bytes = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    auto data = ctx.GetInputData(i);
    if (data.empty() || (i == 0 ? (elem_bytes = data.size()) == 0
                                : data.size() != elem_bytes)) {
      all_static = false;
      break;
    }
  }

  if (all_static) {
    // Determine outer and inner chunk sizes (in bytes) around `axis`.
    size_t outer_size = 1;
    for (int j = 0; j < axis; ++j) {
      if (element_shape[j] == -1) {
        all_static = false;
        break;
      }
      outer_size *= element_shape[j];
    }
    if (all_static && elem_bytes % outer_size == 0) {
      size_t inner_bytes = elem_bytes / outer_size;
      std::vector<uint8_t> packed_data(num_inputs * elem_bytes);
      for (size_t o = 0; o < outer_size; ++o) {
        for (size_t i = 0; i < num_inputs; ++i) {
          auto data = ctx.GetInputData(i);
          const uint8_t* src = data.data() + o * inner_bytes;
          uint8_t* dst =
              packed_data.data() + (o * num_inputs + i) * inner_bytes;
          std::memcpy(dst, src, inner_bytes);
        }
      }
      result.propagated_data[0] = std::move(packed_data);
    }
  }

  return kLiteRtStatusOk;
}

inline LiteRtStatus InferPack(const LiteRtOpT& op,
                              absl::Span<const Dims> input_shapes,
                              std::vector<Dims>& output_shapes) {
  StandaloneShapeInferenceContext ctx(op, input_shapes);
  InferenceResult result;
  result.output_shapes = std::move(output_shapes);
  auto status = InferPack(ctx, result);
  output_shapes = std::move(result.output_shapes);
  return status;
}

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_OPS_PACK_H_
