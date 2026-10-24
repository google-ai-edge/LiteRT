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

#include "third_party/odml/litert/tensor/internal/graph_probe.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/internal/graph_traversal.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert {
namespace tensor {

GraphProbe::GraphProbe(
    const absl::flat_hash_map<std::string, TensorHandle*>& outputs) {
  std::vector<TensorHandle> output_handles;
  for (const auto& [name, tensor] : outputs) {
    output_handles.push_back(*tensor);
  }

  auto execution_plan_or = GetExecutionPlan(output_handles);
  if (!execution_plan_or.ok()) {
    ABSL_LOG(ERROR) << "Failed to get execution plan: "
                    << execution_plan_or.status();
    return;
  }
  const auto& execution_plan = *execution_plan_or;

  absl::flat_hash_map<const graph::Operation*, int> op_to_id;
  for (int i = 0; i < execution_plan.size(); ++i) {
    op_to_id[execution_plan[i]] = i;
  }

  absl::flat_hash_set<const graph::Tensor*> visited_tensors;
  std::vector<const graph::Tensor*> tensors_to_visit;
  for (const auto& [name, tensor] : outputs) {
    tensors_to_visit.push_back(&tensor->GetRaw());
    visited_tensors.insert(&tensor->GetRaw());
  }

  while (!tensors_to_visit.empty()) {
    const graph::Tensor* tensor = tensors_to_visit.back();
    tensors_to_visit.pop_back();

    auto producer_or = graph::GetProducer(*tensor);
    if (!producer_or.ok()) {
      continue;
    }
    const auto& producer = *producer_or;
    if (!producer) {
      continue;
    }

    auto it = op_to_id.find(producer.get());
    if (it == op_to_id.end()) {
      continue;
    }
    const int op_id = it->second;
    const StableTensorId stable_id = {op_id, tensor->index};

    if (probed_tensors_.find(stable_id) == probed_tensors_.end()) {
      probed_tensors_[stable_id] = absl::StrCat(
          "probe_",
          producer->GetName().empty() ? "Unknown" : producer->GetName(), "_",
          op_counter_++);
    }

    for (const auto& input : producer->inputs) {
      if (visited_tensors.find(&input) == visited_tensors.end()) {
        tensors_to_visit.push_back(&input);
        visited_tensors.insert(&input);
      }
    }
  }
}

}  // namespace tensor
}  // namespace litert
