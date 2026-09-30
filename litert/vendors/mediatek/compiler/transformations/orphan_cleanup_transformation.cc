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

#include "litert/vendors/mediatek/compiler/transformations/orphan_cleanup_transformation.h"

#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/compiler/cc/litert_builder.h"
#include "litert/compiler/cc/litert_model.h"

using litert::compiler::Builder;
using litert::compiler::Op;

namespace {

// Set of registered orphan ops. Pointers are only dereferenced once reached
// through live graph edges, so stale entries (e.g. ops erased by another
// rewrite) are harmless.
absl::flat_hash_set<LiteRtOp>& Registry() {
  static auto* registry = new absl::flat_hash_set<LiteRtOp>();
  return *registry;
}

bool AllUsesIn(const Op& op, const absl::flat_hash_set<LiteRtOp>& users) {
  for (const auto& out : op.Outputs()) {
    for (const auto& use : out.Uses()) {
      if (!users.contains(use.user.Get())) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

void RegisterOrphanGroup(const std::vector<LiteRtOp>& ops) {
  for (LiteRtOp op : ops) {
    Registry().insert(op);
  }
}

void ResetOrphanRegistry() { Registry().clear(); }

extern "C" {

LiteRtStatus OrphanCleanupTransformation(const LiteRtCompilerContext* context,
                                         LiteRtBuilder builder_ptr,
                                         LiteRtOp op_ptr) {
  auto& registry = Registry();
  if (!registry.contains(op_ptr)) {
    return kLiteRtStatusPatternNoMatch;
  }
  Op root(context, op_ptr);
  absl::flat_hash_set<LiteRtOp> dead;
  if (!AllUsesIn(root, dead)) {
    return kLiteRtStatusPatternNoMatch;
  }
  dead.insert(op_ptr);

  // Grow the dead set with registered producers whose outputs are only used by
  // already-dead ops.
  std::vector<Op> worklist = {root};
  while (!worklist.empty()) {
    Op cur = worklist.back();
    worklist.pop_back();
    for (const auto& in : cur.Inputs()) {
      if (in.Get() == nullptr) continue;
      auto def = in.GetDefiningOp();
      if (!def || dead.contains(def->Get()) || !registry.contains(def->Get())) {
        continue;
      }
      if (AllUsesIn(*def, dead)) {
        dead.insert(def->Get());
        worklist.push_back(*def);
      }
    }
  }

  Builder builder(context, builder_ptr);
  for (LiteRtOp d : dead) {
    builder.EraseOp(Op(context, d));
    registry.erase(d);
  }
  return kLiteRtStatusOk;
}

}  // extern "C"
