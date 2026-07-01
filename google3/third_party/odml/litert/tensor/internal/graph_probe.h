/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_PROBE_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_PROBE_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert {
namespace tensor {

// A utility to traverse a litert::tensor graph and identify all intermediate
// tensors with stable identifiers.
class GraphProbe {
 public:
  // A stable identifier for a tensor, composed of the producer's operation ID
  // (its index in a topologically sorted list) and the tensor's output index
  // from that operation.
  using StableTensorId = std::pair<int, int>;

  // Hash function for StableTensorId to be used in absl::flat_hash_map.
  struct StableTensorIdHash {
    std::size_t operator()(const StableTensorId& id) const {
      return std::hash<int>()(id.first) ^ std::hash<int>()(id.second);
    }
  };

  // Initializes the probe with the final output tensors of a graph.
  explicit GraphProbe(
      const absl::flat_hash_map<std::string, TensorHandle*>& outputs);

  // Returns a map from the stable tensor ID to a unique generated name.
  const absl::flat_hash_map<StableTensorId, std::string, StableTensorIdHash>&
  GetProbedTensors() const {
    return probed_tensors_;
  }

 private:
  absl::flat_hash_map<StableTensorId, std::string, StableTensorIdHash>
      probed_tensors_;
  int op_counter_ = 0;
};

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_PROBE_H_
