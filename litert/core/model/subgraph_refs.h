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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_

// Utilities for ops that reference other subgraphs by index.
//
// TFLite flatbuffers cannot nest regions inside an operator. Region-bearing
// ops (e.g. STABLEHLO_REDUCE_WINDOW, WHILE, IF, STABLEHLO_COMPOSITE) instead
// store their region computations as separate subgraphs and reference them by
// integer index in BuiltinOptions / BuiltinOptions2. This header is the single
// source of truth for which option fields hold such indices. Any code that
// moves, removes, clones or reorders subgraphs must keep these indices
// consistent, see `UpdateSubgraphRefs` and `SubgraphCallGraph`.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

// The semantic role of a referenced subgraph with respect to the referencing
// op.
enum class SubgraphRefKind : uint8_t {
  // Optional decomposition of a STABLEHLO_COMPOSITE. The composite has
  // well-defined semantics without it, and it may be inlined in place of the
  // composite. Index -1 means "no decomposition".
  kDecomposition = 0,

  // Scalar or elementwise computation attached to an op that applies it
  // internally, e.g. the reduction body of STABLEHLO_REDUCE(_WINDOW), the
  // comparator of STABLEHLO_SORT or the update computation of
  // STABLEHLO_SCATTER. The subgraph has no meaning on its own and is invoked
  // many times per execution of the parent op. It must always live in the
  // same compilation unit as its parent op.
  kComputation = 1,

  // Control flow body, e.g. the cond/body of WHILE, the branches of IF or
  // STABLEHLO_CASE and the init subgraph of CALL_ONCE. These are "large"
  // bodies that may be partitioned independently while the parent op itself
  // runs on CPU.
  kControlFlow = 2,
};

// Returns a human readable name of `kind` for logging.
absl::string_view SubgraphRefKindName(SubgraphRefKind kind);

// Returns true for the region kinds (`kComputation` and `kControlFlow`), i.e.
// references to subgraphs that are part of the semantics of the referencing
// op, as opposed to optional composite decompositions.
inline bool IsRegionRefKind(SubgraphRefKind kind) {
  return kind == SubgraphRefKind::kComputation ||
         kind == SubgraphRefKind::kControlFlow;
}

// A single subgraph reference held by an op.
struct SubgraphRef {
  int32_t index;
  SubgraphRefKind kind;
};

using SubgraphRefs = absl::InlinedVector<SubgraphRef, 2>;

// Returns true if the op's options contain at least one subgraph index field,
// regardless of the value held.
bool HasSubgraphRefs(const LiteRtOpT& op);

// Returns true if the op holds a region reference (`kComputation` or
// `kControlFlow`) with a non-negative index. Region-bearing ops cannot be
// separated from the subgraphs they reference.
bool HasRegionRefs(const LiteRtOpT& op);

// Returns all subgraph references held by the op, in a deterministic order.
// Negative indices (e.g. -1 for a composite without decomposition) are reported
// as is.
SubgraphRefs GetSubgraphRefs(const LiteRtOpT& op);

// Calls `fn` with a mutable reference to every subgraph index field held by the
// op. `fn` may update the index in place. This is a no-op for ops without
// subgraph references.
void UpdateSubgraphRefs(
    LiteRtOpT& op,
    absl::FunctionRef<void(int32_t& index, SubgraphRefKind kind)> fn);

// Verifies that every subgraph reference in the model is in range. Region
// references must be in [0, NumSubgraphs()), decomposition references must be
// in [-1, NumSubgraphs()).
Expected<void> ValidateSubgraphRefs(const LiteRtModelT& model);

// Directed graph of "op in subgraph A references subgraph B" edges for a
// snapshot of a model. The graph does not track later model mutations; rebuild
// it after subgraphs are added, removed or reordered.
class SubgraphCallGraph {
 public:
  struct Edge {
    LiteRtOp op;
    uint32_t caller;
    uint32_t callee;
    SubgraphRefKind kind;
  };

  // Builds the graph. Fails if any reference is out of range (see
  // `ValidateSubgraphRefs`). Composite decompositions with index -1 are
  // skipped.
  static Expected<SubgraphCallGraph> Build(const LiteRtModelT& model);

  size_t NumSubgraphs() const { return callers_.size(); }

  // Edges pointing at `subgraph`.
  absl::Span<const Edge> Callers(uint32_t subgraph) const {
    return callers_.at(subgraph);
  }

  // Edges originating from ops in `subgraph`.
  absl::Span<const Edge> Callees(uint32_t subgraph) const {
    return callees_.at(subgraph);
  }

  // True if any op references `subgraph`.
  bool IsReferenced(uint32_t subgraph) const {
    return !callers_.at(subgraph).empty();
  }

  // True if any op references `subgraph` with the given kind.
  bool IsReferencedAs(uint32_t subgraph, SubgraphRefKind kind) const;

  // All subgraph indices ordered such that every caller precedes its callees.
  // Fails if the reference graph has a cycle (recursion is not supported by
  // TFLite).
  Expected<std::vector<uint32_t>> TopologicalOrder() const;

 private:
  std::vector<std::vector<Edge>> callers_;
  std::vector<std::vector<Edge>> callees_;
};

}  // namespace litert::internal

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_
