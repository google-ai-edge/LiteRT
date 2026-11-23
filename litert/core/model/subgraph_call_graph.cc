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

#include "litert/core/model/subgraph_call_graph.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/model/subgraph_refs.h"

namespace litert::internal {
Expected<SubgraphCallGraph> BuildSubgraphCallGraph(const LiteRtModelT& model) {
  const auto subgraphs = model.Subgraphs();
  const size_t num_subgraphs = subgraphs.size();
  SubgraphCallGraph graph;
  graph.callees.resize(num_subgraphs);
  for (size_t i = 0; i < num_subgraphs; ++i) {
    auto& callees = graph.callees[i];
    for (const LiteRtOpT* op : subgraphs[i]->Ops()) {
      for (const int32_t ref : SubgraphRefs(*op)) {
        if (ref < 0 || static_cast<size_t>(ref) >= num_subgraphs) {
          return Unexpected(
              kLiteRtStatusErrorUnsupported,
              absl::StrCat("An op in subgraph ", i,
                           " references invalid subgraph index ", ref,
                           " (model has ", num_subgraphs, " subgraphs)"));
        }
        if (!absl::c_linear_search(callees, ref)) {
          callees.push_back(ref);
        }
      }
    }
  }
  if (TopologicalOrder(graph).size() != num_subgraphs) {
    return Unexpected(kLiteRtStatusErrorUnsupported,
                      "Subgraph references form a cycle");
  }
  return graph;
}

absl::flat_hash_set<int32_t> SubgraphClosure(const SubgraphCallGraph& graph,
                                             absl::Span<const int32_t> roots) {
  absl::flat_hash_set<int32_t> closure;
  std::vector<int32_t> pending;
  for (const int32_t root : roots) {
    if (closure.insert(root).second) {
      pending.push_back(root);
    }
  }
  while (!pending.empty()) {
    const int32_t subgraph = pending.back();
    pending.pop_back();
    for (const int32_t callee : graph.callees[subgraph]) {
      if (closure.insert(callee).second) {
        pending.push_back(callee);
      }
    }
  }
  return closure;
}

std::vector<int32_t> TopologicalOrder(const SubgraphCallGraph& graph) {
  const size_t num_subgraphs = graph.callees.size();
  std::vector<int> num_callers(num_subgraphs, 0);
  for (const auto& callees : graph.callees) {
    for (const int32_t callee : callees) {
      ++num_callers[callee];
    }
  }
  // Kahn's algorithm. Roots are queued in ascending index order and callees in
  // first-seen order, so the result is deterministic.
  std::deque<int32_t> ready;
  for (size_t i = 0; i < num_subgraphs; ++i) {
    if (num_callers[i] == 0) {
      ready.push_back(static_cast<int32_t>(i));
    }
  }
  std::vector<int32_t> order;
  order.reserve(num_subgraphs);
  while (!ready.empty()) {
    const int32_t subgraph = ready.front();
    ready.pop_front();
    order.push_back(subgraph);
    for (const int32_t callee : graph.callees[subgraph]) {
      if (--num_callers[callee] == 0) {
        ready.push_back(callee);
      }
    }
  }
  return order;
}

}  // namespace litert::internal
