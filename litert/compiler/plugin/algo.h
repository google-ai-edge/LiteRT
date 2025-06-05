// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_COMPILER_PLUGIN_ALGO_H_
#define ODML_LITERT_LITERT_COMPILER_PLUGIN_ALGO_H_

#include <vector>

#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"

namespace litert::internal {

// Identifies sub-DAGs of ops connected w.r.t. the use-def chain. Expects
// all "ops" belong to the same Subgraph. The ops in the input
// and output will always be the same.
std::vector<std::vector<LiteRtOp>> GroupPartitions(
    const std::vector<LiteRtOpWithPartitionIndex>& ops);

// Outlines "partition" from "root" into the empty subgraph "slice". Assumes
// the partition is a valid sub-DAG, and replaces it with a single
// tfl.custom_op in "root". A reference to that op is returned.
LiteRtOp OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
                          std::vector<LiteRtOp>& partition);

// Inline a subgraph to a destination op.
//
// This is a helper function to inline the all ops used in a subgraph to a
// destination op.
//
// Use cases:
// 1. Inline a decomposition subgraph to a composite op: When a composite op can
// not be directly compiled by a vendor plugin, it is possible that some(all)
// ops used in the decomposition subgraph can be compiled by the vendor plugin.
// In this case, we can inline the decomposition subgraph to the composite op,
// and compile the composite op with the vendor plugin. To give best effort
// compilation result, we inline the decomposition subgraph to replace the
// composite op before asking the vendor plugin to partition the main subgraph
// again.
//
// This function clones the ops and tensors from the decomposition subgraph into
// the main subgraph, and then restore the topology of the decomposition
// subgraph to the main subgraph.
//
// WARNING: The decomposition subgraph is not removed from the model. The caller
// should remove it if needed.
//
// Returns true if the inlining is successful.
Expected<void> InlineSubgraph(LiteRtModelT& model, LiteRtOpT& destination_op,
                              LiteRtSubgraph source_subgraph);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_COMPILER_PLUGIN_ALGO_H_
