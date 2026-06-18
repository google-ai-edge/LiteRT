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

#include "tensor/internal/graph_traversal.h"

#include <functional>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert {
namespace tensor {

absl::StatusOr<std::vector<const graph::Operation*>> GetExecutionPlan(
    absl::Span<const TensorHandle> outputs) {
  absl::flat_hash_set<const graph::Operation*> visited;
  absl::flat_hash_set<const graph::Operation*> recursion_stack;
  std::vector<const graph::Operation*> execution_plan;

  std::function<bool(const graph::Operation*)> dfs =
      [&](const graph::Operation* op) -> bool {
    if (recursion_stack.count(op)) {
      return true;
    }
    if (visited.count(op)) {
      return false;
    }

    recursion_stack.insert(op);
    visited.insert(op);

    for (const auto& input : op->inputs) {
      auto producer_or = graph::GetProducer(input);
      if (producer_or.ok()) {
        auto producer = producer_or.value();
        if (producer) {
          if (dfs(producer.get())) {
            return true;
          }
        }
      }
    }

    execution_plan.push_back(op);
    recursion_stack.erase(op);
    return false;
  };

  for (const auto& output : outputs) {
    auto producer_or = graph::GetProducer(output.GetRaw());
    if (producer_or.ok()) {
      auto producer = producer_or.value();
      if (producer) {
        if (dfs(producer.get())) {
          return absl::InternalError("Cycle detected in the graph.");
        }
      }
    }
  }

  return execution_plan;
}

}  // namespace tensor
}  // namespace litert
