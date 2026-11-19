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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_CALL_GRAPH_H_
#define ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_CALL_GRAPH_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"

namespace litert::internal {

// Which subgraphs reference which, derived from the subgraph references held
// by ops (see subgraph_refs.h). Indices are positions in
// `LiteRtModelT::Subgraphs()` at the time the graph was built.
struct SubgraphCallGraph {
  // Distinct subgraphs referenced by ops in each subgraph, in first-seen
  // order.
  std::vector<absl::InlinedVector<int32_t, 2>> callees;
};

// Builds the call graph of `model`. Fails with kLiteRtStatusErrorUnsupported
// if any op references an out-of-range subgraph, holds -1 in a non-nullable
// slot, or if the references form a cycle.
Expected<SubgraphCallGraph> BuildSubgraphCallGraph(const LiteRtModelT& model);

// `roots` and every subgraph transitively referenced from them.
absl::flat_hash_set<int32_t> SubgraphClosure(const SubgraphCallGraph& graph,
                                             absl::Span<const int32_t> roots);

// All subgraph indices ordered so that every caller precedes its callees. The
// order is deterministic. `graph` must be acyclic, which
// `BuildSubgraphCallGraph` guarantees.
std::vector<int32_t> TopologicalOrder(const SubgraphCallGraph& graph);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_CALL_GRAPH_H_
