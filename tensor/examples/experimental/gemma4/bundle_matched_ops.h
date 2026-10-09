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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_MATCHED_OPS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_MATCHED_OPS_H_

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/datatypes.h"
#include "tensor/examples/experimental/gemma4/mobile_fully_connected.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::experimental::gemma4::cpu {

// Controls whether per-layer projection reshapes use static authoring shapes
// (`kStatic`) or infer the sequence-row dimension on axis 1 (`kDynamic`).
enum class PleRowMode {
  kStatic,
  kDynamic,
};

// Projects `scaled_embedding` (already multiplied by `sqrt(embed_dim)`)
// through the bundle's single global INT8 PLE matrix `[num_layers *
// per_layer_dim, embed_dim]` and splits the output into one tensor per layer
// after static output requantization. Returns error tensors on invalid inputs
// or missing activation scales.
template <class... Mixins>
std::vector<Tensor<Mixins...>> BundlePerLayerProjectionParts(
    const Tensor<Mixins...>& scaled_embedding,
    const Tensor<Mixins...>& full_weight,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    int num_layers, int per_layer_dim,
    PleRowMode row_mode = PleRowMode::kStatic) {
  const auto fail_status = [&](absl::Status status) {
    return std::vector<Tensor<Mixins...>>(std::max(1, num_layers),
                                          Tensor<Mixins...>(std::move(status)));
  };
  const auto fail = [&](absl::string_view message) {
    return fail_status(absl::InvalidArgumentError(message));
  };
  if (!scaled_embedding.GetStatus().ok()) {
    return fail_status(scaled_embedding.GetStatus());
  }
  if (!full_weight.GetStatus().ok()) {
    return fail_status(full_weight.GetStatus());
  }
  if (num_layers <= 0 || per_layer_dim <= 0 ||
      num_layers > std::numeric_limits<int>::max() / per_layer_dim ||
      scaled_embedding.GetShape().empty() ||
      (row_mode == PleRowMode::kDynamic &&
       scaled_embedding.GetShape().size() < 3) ||
      full_weight.GetShape().size() != 2 ||
      full_weight.GetType() != Type::kI8 ||
      full_weight.GetShape()[0] != num_layers * per_layer_dim ||
      full_weight.GetShape()[1] != scaled_embedding.GetShape().back() ||
      full_weight.GetName() != "model.per_layer_model_projection.weight") {
    return fail(
        "Bundle global PLE requires the named full INT8 matrix with "
        "shape [num_layers * per_layer_dim, embed_dim]");
  }
  if (!weights.contains("model.per_layer_model_projection.input_scale") ||
      !weights.contains("model.per_layer_model_projection.output_scale")) {
    return fail("Bundle global PLE requires both original activation scales");
  }
  Tensor<Mixins...> projected =
      MobileFullyConnected(scaled_embedding, full_weight, &weights);
  if (!projected.GetStatus().ok()) {
    return fail_status(projected.GetStatus());
  }
  Shape expanded_shape = scaled_embedding.GetShape();
  expanded_shape.back() = num_layers;
  expanded_shape.push_back(per_layer_dim);
  if (row_mode == PleRowMode::kDynamic) {
    expanded_shape[1] = kInferredDim;
  }
  Tensor<Mixins...> expanded = Reshape(projected, expanded_shape);
  Shape part_shape = scaled_embedding.GetShape();
  part_shape.back() = per_layer_dim;
  if (row_mode == PleRowMode::kDynamic) {
    part_shape[1] = kInferredDim;
  }
  Shape offsets(expanded_shape.size(), 0);
  Shape sizes = expanded_shape;
  sizes[sizes.size() - 2] = 1;
  std::vector<Tensor<Mixins...>> parts;
  parts.reserve(num_layers);
  for (int layer = 0; layer < num_layers; ++layer) {
    offsets[offsets.size() - 2] = layer;
    parts.push_back(Reshape(Slice(expanded, offsets, sizes), part_shape));
  }
  return parts;
}

}  // namespace litert::tensor::examples::experimental::gemma4::cpu

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_EXPERIMENTAL_GEMMA4_BUNDLE_MATCHED_OPS_H_
