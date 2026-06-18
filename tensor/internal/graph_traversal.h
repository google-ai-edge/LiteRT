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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_TRAVERSAL_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_TRAVERSAL_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert {
namespace tensor {

// Performs a reverse topological sort on the graph reachable from the given
// output tensors to produce a deterministic execution plan.
absl::StatusOr<std::vector<const graph::Operation*>> GetExecutionPlan(
    absl::Span<const TensorHandle> outputs);

}  // namespace tensor
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_GRAPH_TRAVERSAL_H_
