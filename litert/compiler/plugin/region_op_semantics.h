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

#ifndef ODML_LITERT_LITERT_COMPILER_PLUGIN_REGION_OP_SEMANTICS_H_
#define ODML_LITERT_LITERT_COMPILER_PLUGIN_REGION_OP_SEMANTICS_H_

#include "litert/c/litert_op_code.h"

namespace litert::internal {

// How the CPU kernel of a region-bearing op executes the subgraphs it
// references. This decides whether the partitioner may offload ops inside
// those subgraphs when the op itself stays on CPU.
enum class RegionInvocation {
  // The kernel invokes each referenced subgraph as a whole graph (control
  // flow). Ops inside may be offloaded to NPU.
  kWholeSubgraph,
  // The kernel interprets the referenced subgraph internally, e.g. as a scalar
  // reduction or comparator. The subgraph must stay unchanged.
  kAtomic,
};

// Returns the invocation semantics of `code`. Op codes that are not known
// control-flow ops, including future ones, are `kAtomic`.
RegionInvocation RegionInvocationOf(LiteRtOpCode code);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_COMPILER_PLUGIN_REGION_OP_SEMANTICS_H_
