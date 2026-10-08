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

#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_BUNDLE_MATCHED_OPS_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_LITERT_BUNDLE_MATCHED_OPS_H_
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/examples/gemma4/litert/mobile_fully_connected.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"
namespace litert::tensor::examples::gemma4::cpu {
// Input is already multiplied by sqrt(embed_dim), as required by the bundle.
// The one global INT8 FC produces [..., num_layers * per_layer_dim]. Split only
// after its static output requantization, preserving that full-matrix boundary.
template <class... Mixins>
std::vector<Tensor<Mixins...>> BundlePerLayerProjectionParts(
    Tensor<Mixins...> scaled_embedding, Tensor<Mixins...> full_weight,
    const absl::flat_hash_map<std::string, Tensor<Mixins...>>& weights,
    int num_layers, int per_layer_dim) {
  auto fail = [&](const std::string& message) {
    return std::vector<Tensor<Mixins...>>(
        std::max(1, num_layers), Tensor<Mixins...>(graph::ErrorTensor(
                                     absl::InvalidArgumentError(message))));
  };
  if (num_layers <= 0 || per_layer_dim <= 0 ||
      num_layers > std::numeric_limits<int>::max() / per_layer_dim ||
      scaled_embedding.GetShape().empty() ||
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
  Shape expanded_shape = scaled_embedding.GetShape();
  expanded_shape.back() = num_layers;
  expanded_shape.push_back(per_layer_dim);
  Tensor<Mixins...> expanded = Reshape(projected, expanded_shape);
  Shape part_shape = scaled_embedding.GetShape();
  part_shape.back() = per_layer_dim;
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

}  // namespace litert::tensor::examples::gemma4::cpu
#endif
