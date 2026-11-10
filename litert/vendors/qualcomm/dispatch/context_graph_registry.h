// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

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

  // Returns only graphs that are not already present in a registered context.
  // An empty result means that the required graph is already enabled.
  Expected<std::vector<std::string>> PlanAdditionalGraphsToEnable(
      absl::string_view required_graph) const {
    if (!available_graphs_.contains(required_graph)) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          "Required graph is not available in the context binary");
    }

    std::vector<std::string> graphs;
    for (const std::string& graph : available_graphs_) {
      if ((graph == required_graph || expected_graphs_.contains(graph)) &&
          !enabled_graph_to_context_.contains(graph)) {
        graphs.emplace_back(graph);
      }
    }
    return graphs;
  }

  ContextHandle* FindContext(absl::string_view graph) {
    auto it = enabled_graph_to_context_.find(graph);
    return it == enabled_graph_to_context_.end() ? nullptr
                                                 : contexts_[it->second].get();
  }

  ContextHandle& RegisterContextWithAllAvailableGraphs(
      std::unique_ptr<ContextHandle> context) {
    const size_t context_index = contexts_.size();
    contexts_.emplace_back(std::move(context));
    for (const std::string& graph : available_graphs_) {
      enabled_graph_to_context_[graph] = context_index;
    }
    return *contexts_.back();
  }

  ContextHandle& RegisterContext(std::unique_ptr<ContextHandle> context,
                                 absl::Span<const std::string> enabled_graphs) {
    const size_t context_index = contexts_.size();
    contexts_.emplace_back(std::move(context));
    for (const std::string& graph : enabled_graphs) {
      enabled_graph_to_context_[graph] = context_index;
    }
    return *contexts_.back();
  }

 private:
  absl::flat_hash_set<std::string> available_graphs_;
  absl::flat_hash_set<std::string> expected_graphs_;
  std::vector<std::unique_ptr<ContextHandle>> contexts_;
  absl::flat_hash_map<std::string, size_t> enabled_graph_to_context_;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_DISPATCH_CONTEXT_GRAPH_REGISTRY_H_
