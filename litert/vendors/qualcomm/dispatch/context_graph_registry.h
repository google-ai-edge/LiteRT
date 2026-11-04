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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_CONTEXT_GRAPH_REGISTRY_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_CONTEXT_GRAPH_REGISTRY_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::qnn {

// Owns the QNN contexts created from one context binary and records which
// context contains each graph.
template <typename ContextHandle>
class ContextGraphRegistry {
 public:
  ContextGraphRegistry(absl::Span<const std::string> available_graphs,
                       const absl::flat_hash_set<std::string>& expected_graphs)
      : available_graphs_(available_graphs.begin(), available_graphs.end()) {
    if (expected_graphs.empty()) {
      expected_graphs_ = available_graphs_;
    } else {
      for (const std::string& graph : available_graphs_) {
        if (expected_graphs.contains(graph)) {
          expected_graphs_.insert(graph);
        }
      }
    }
  }

  ContextHandle* FindContext(absl::string_view graph) {
    auto it = graph_to_context_.find(graph);
    return it == graph_to_context_.end() ? nullptr
                                         : contexts_[it->second].get();
  }

  // Returns only graphs that are not already present in a registered context.
  // An empty result means that a new context should enable all available
  // graphs.
  Expected<std::vector<std::string>> PlanAdditionalGraphsToEnable(
      absl::string_view required_graph) const {
    if (!available_graphs_.contains(required_graph)) {
      return Unexpected(kLiteRtStatusErrorInvalidArgument,
                        "Required graph is not in the context binary");
    }

    std::vector<std::string> graphs;
    for (const std::string& graph : available_graphs_) {
      if ((graph == required_graph || expected_graphs_.contains(graph)) &&
          !graph_to_context_.contains(graph)) {
        graphs.emplace_back(graph);
      }
    }
    if (contexts_.empty() && graphs.size() == available_graphs_.size()) {
      graphs.clear();
    }
    return graphs;
  }

  ContextHandle& RegisterContextWithAllAvailableGraphs(
      std::unique_ptr<ContextHandle> context) {
    const size_t context_index = contexts_.size();
    contexts_.emplace_back(std::move(context));
    for (const std::string& graph : available_graphs_) {
      graph_to_context_[graph] = context_index;
    }
    return *contexts_.back();
  }

  ContextHandle& RegisterContext(std::unique_ptr<ContextHandle> context,
                                 absl::Span<const std::string> enabled_graphs) {
    const size_t context_index = contexts_.size();
    contexts_.emplace_back(std::move(context));
    for (const std::string& graph : enabled_graphs) {
      graph_to_context_[graph] = context_index;
    }
    return *contexts_.back();
  }

 private:
  absl::flat_hash_set<std::string> available_graphs_;
  absl::flat_hash_set<std::string> expected_graphs_;
  std::vector<std::unique_ptr<ContextHandle>> contexts_;
  absl::flat_hash_map<std::string, size_t> graph_to_context_;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_CONTEXT_GRAPH_REGISTRY_H_
