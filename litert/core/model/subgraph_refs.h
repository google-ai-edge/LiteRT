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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_
#define ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_

#include <cstdint>
#include <optional>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"

// Registry of the TFLite option fields that hold subgraph indices.
//
// Ops reference other subgraphs (region bodies, control-flow branches,
// composite decompositions) through integer indices stored in their
// `BuiltinOptions` / `BuiltinOptions2` tables. Those indices are the canonical
// representation of subgraph references: they cross the compiler plugin C ABI
// and are copied verbatim by the serializer. New code should read and write
// subgraph indices through these functions so that a new index field only needs
// to be registered here.

namespace litert::internal {

// Subgraph indices referenced by `op`, in field order. Only
// `StableHLOCompositeOptions::decomposition_subgraph_index` is nullable: its -1
// means "no subgraph" and is omitted. A -1 in any other field (or any other
// invalid value) is kept so that validation can reject it. Ops whose option
// union is unset have no references.
absl::InlinedVector<int32_t, 2> SubgraphRefs(const LiteRtOpT& op);

// True if `SubgraphRefs(op)` is non-empty.
bool HasSubgraphRefs(const LiteRtOpT& op);

// True if `op` references subgraphs and is not a StableHLO composite.
bool IsRegionOp(const LiteRtOpT& op);

// Rewrites every subgraph index `i` in `op` to `old_to_new[i]`. A nullable -1
// is left unchanged. Fails without modifying `op` if any index is out of range
// of `old_to_new` or maps to a negative value.
Expected<void> RemapSubgraphRefs(LiteRtOpT& op,
                                 absl::Span<const int32_t> old_to_new);

// Rewrites every subgraph index equal to `from` in `op` to `to`. Returns the
// number of slots rewritten.
int RetargetSubgraphRefs(LiteRtOpT& op, int32_t from, int32_t to);

// The raw decomposition index of a StableHLO composite (possibly -1), or
// nullopt if `op` does not carry `StableHLOCompositeOptions`.
std::optional<int32_t> CompositeDecompositionIndex(const LiteRtOpT& op);

// Sets the decomposition index of a StableHLO composite to -1. Used when the
// composite is compiled as an opaque op and its decomposition is not needed.
// Does nothing for other ops.
void ClearCompositeDecomposition(LiteRtOpT& op);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_SUBGRAPH_REFS_H_
