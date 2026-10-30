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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ORPHAN_CLEANUP_TRANSFORMATION_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ORPHAN_CLEANUP_TRANSFORMATION_H_

#include <vector>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"

// The LiteRT builder splices newly built ops at the position of the earliest
// erased op. Rewrites whose replaced sub-DAG starts before some of the new
// ops' inputs are defined therefore can only erase the tail of the matched
// DAG; the remaining (now dead) producers are registered here as an "orphan
// group" and erased by OrphanCleanupTransformation once they have no users.
//
// Only ops explicitly registered by a transformation are ever erased, so
// unrelated dead code is never touched. Subgraph outputs are not visible as
// tensor uses here, so callers must only register ops whose outputs are
// internal (never subgraph outputs).

// Registers ops that become dead once the registering rewrite is applied.
void RegisterOrphanGroup(const std::vector<LiteRtOp>& ops);

// Clears all registered orphan groups (called when transformations are
// (re-)registered for a new compilation).
void ResetOrphanRegistry();

#ifdef __cplusplus
extern "C" {
#endif

// Erases a registered orphan op whose outputs have no users, together with the
// registered producers (reachable through its inputs) that become dead.
//
// Before:                              After:
//   Mul   Mul    Mul   Mul
//     \   /        \   /                 (all ops of the dead sub-DAG that
//      Sub          Add                   belong to the orphan group are
//        \          /                     erased)
//         Concat  (0 users)
LiteRtStatus OrphanCleanupTransformation(const LiteRtCompilerContext* context,
                                         LiteRtBuilder builder_ptr,
                                         LiteRtOp op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_TRANSFORMATIONS_ORPHAN_CLEANUP_TRANSFORMATION_H_
