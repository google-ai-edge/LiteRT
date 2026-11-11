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

#include "litert/core/model/subgraph_refs.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

constexpr bool kNullable = true;
constexpr bool kNotNullable = false;

// Calls `fn(slot, nullable)` for every subgraph index slot in a
// `BuiltinOptions` union. `Union` may be const.
template <typename Union, typename Fn>
void VisitBuiltinOptions(Union& opts, Fn&& fn) {
  switch (opts.type) {
    case tflite::BuiltinOptions_IfOptions:
      if (auto* o = opts.AsIfOptions()) {
        fn(o->then_subgraph_index, kNotNullable);
        fn(o->else_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions_WhileOptions:
      if (auto* o = opts.AsWhileOptions()) {
        fn(o->cond_subgraph_index, kNotNullable);
        fn(o->body_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions_CallOnceOptions:
      if (auto* o = opts.AsCallOnceOptions()) {
        fn(o->init_subgraph_index, kNotNullable);
      }
      return;
    default:
      return;
  }
}

// Calls `fn(slot, nullable)` for every subgraph index slot in a
// `BuiltinOptions2` union. `Union` may be const.
template <typename Union, typename Fn>
void VisitBuiltinOptions2(Union& opts, Fn&& fn) {
  switch (opts.type) {
    case tflite::BuiltinOptions2_StableHLOCompositeOptions:
      if (auto* o = opts.AsStableHLOCompositeOptions()) {
        fn(o->decomposition_subgraph_index, kNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloReduceOptions:
      if (auto* o = opts.AsStablehloReduceOptions()) {
        fn(o->body_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloReduceWindowOptions:
      if (auto* o = opts.AsStablehloReduceWindowOptions()) {
        fn(o->body_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloScatterOptions:
      if (auto* o = opts.AsStablehloScatterOptions()) {
        fn(o->update_computation_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloSortOptions:
      if (auto* o = opts.AsStablehloSortOptions()) {
        fn(o->comparator_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloCustomCallOptions:
      if (auto* o = opts.AsStablehloCustomCallOptions()) {
        for (auto& index : o->called_computations) {
          fn(index, kNotNullable);
        }
      }
      return;
    case tflite::BuiltinOptions2_StablehloWhileOptions:
      if (auto* o = opts.AsStablehloWhileOptions()) {
        fn(o->cond_subgraph_index, kNotNullable);
        fn(o->body_subgraph_index, kNotNullable);
      }
      return;
    case tflite::BuiltinOptions2_StablehloCaseOptions:
      if (auto* o = opts.AsStablehloCaseOptions()) {
        for (auto& index : o->branch_subgraph_indices) {
          fn(index, kNotNullable);
        }
      }
      return;
    default:
      return;
  }
}

bool IsNullRef(int32_t index, bool nullable) {
  return nullable && index == -1;
}

// Visits every option slot of `op` that holds a subgraph index. `nullable` is
// true only for slots where -1 means "no subgraph" (only
// `StableHLOCompositeOptions::decomposition_subgraph_index`). Ops whose option
// union is unset visit nothing.
void ForEachSubgraphRef(
    LiteRtOpT& op, absl::FunctionRef<void(int32_t& index, bool nullable)> fn) {
  TflOptions opts = TakeTflOptions(op);
  VisitBuiltinOptions(opts, fn);
  SetTflOptions(op, std::move(opts));

  TflOptions2 opts2 = TakeTflOptions2(op);
  VisitBuiltinOptions2(opts2, fn);
  SetTflOptions2(op, std::move(opts2));
}

void ForEachSubgraphRef(
    const LiteRtOpT& op,
    absl::FunctionRef<void(int32_t index, bool nullable)> fn) {
  VisitBuiltinOptions(GetTflOptions(op), fn);
  VisitBuiltinOptions2(GetTflOptions2(op), fn);
}

}  // namespace

absl::InlinedVector<int32_t, 2> SubgraphRefs(const LiteRtOpT& op) {
  absl::InlinedVector<int32_t, 2> refs;
  ForEachSubgraphRef(op, [&refs](int32_t index, bool nullable) {
    if (!IsNullRef(index, nullable)) {
      refs.push_back(index);
    }
  });
  return refs;
}

bool HasSubgraphRefs(const LiteRtOpT& op) { return !SubgraphRefs(op).empty(); }

bool IsRegionOp(const LiteRtOpT& op) {
  return op.OpCode() != kLiteRtOpCodeShloComposite && HasSubgraphRefs(op);
}

Expected<void> RemapSubgraphRefs(LiteRtOpT& op,
                                 absl::Span<const int32_t> old_to_new) {
  // Validate first so that a failure leaves `op` unchanged.
  for (const int32_t index : SubgraphRefs(op)) {
    if (index < 0 || static_cast<size_t>(index) >= old_to_new.size() ||
        old_to_new[index] < 0) {
      return Unexpected(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrCat("Cannot remap subgraph reference ", index));
    }
  }
  ForEachSubgraphRef(op, [&old_to_new](int32_t& index, bool nullable) {
    if (!IsNullRef(index, nullable)) {
      index = old_to_new[index];
    }
  });
  return {};
}

int RetargetSubgraphRefs(LiteRtOpT& op, int32_t from, int32_t to) {
  int num_rewritten = 0;
  ForEachSubgraphRef(op, [&](int32_t& index, bool nullable) {
    if (!IsNullRef(index, nullable) && index == from) {
      index = to;
      ++num_rewritten;
    }
  });
  return num_rewritten;
}

std::optional<int32_t> CompositeDecompositionIndex(const LiteRtOpT& op) {
  const auto* options = GetTflOptions2(op).AsStableHLOCompositeOptions();
  if (options == nullptr) {
    return std::nullopt;
  }
  return options->decomposition_subgraph_index;
}

void ClearCompositeDecomposition(LiteRtOpT& op) {
  if (GetTflOptions2(op).AsStableHLOCompositeOptions() == nullptr) {
    return;
  }
  TflOptions2 opts = TakeTflOptions2(op);
  opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index = -1;
  SetTflOptions2(op, std::move(opts));
}

}  // namespace litert::internal
