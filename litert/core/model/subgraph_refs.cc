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
#include <functional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

// Visits every subgraph index field in `opts` (BuiltinOptions). `Opts` may be
// const, in which case `fn` receives const references.
template <class Opts, class Fn>
void VisitTflOptions(Opts& opts, Fn&& fn) {
  switch (opts.type) {
    case ::tflite::BuiltinOptions_WhileOptions:
      if (auto* o = opts.AsWhileOptions(); o != nullptr) {
        fn(o->cond_subgraph_index, SubgraphRefKind::kControlFlow);
        fn(o->body_subgraph_index, SubgraphRefKind::kControlFlow);
      }
      break;
    case ::tflite::BuiltinOptions_IfOptions:
      if (auto* o = opts.AsIfOptions(); o != nullptr) {
        fn(o->then_subgraph_index, SubgraphRefKind::kControlFlow);
        fn(o->else_subgraph_index, SubgraphRefKind::kControlFlow);
      }
      break;
    case ::tflite::BuiltinOptions_CallOnceOptions:
      if (auto* o = opts.AsCallOnceOptions(); o != nullptr) {
        fn(o->init_subgraph_index, SubgraphRefKind::kControlFlow);
      }
      break;
    default:
      break;
  }
}

// Visits every subgraph index field in `opts` (BuiltinOptions2). `Opts` may be
// const, in which case `fn` receives const references.
template <class Opts, class Fn>
void VisitTflOptions2(Opts& opts, Fn&& fn) {
  switch (opts.type) {
    case ::tflite::BuiltinOptions2_StableHLOCompositeOptions:
      if (auto* o = opts.AsStableHLOCompositeOptions(); o != nullptr) {
        fn(o->decomposition_subgraph_index, SubgraphRefKind::kDecomposition);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloReduceOptions:
      if (auto* o = opts.AsStablehloReduceOptions(); o != nullptr) {
        fn(o->body_subgraph_index, SubgraphRefKind::kComputation);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloReduceWindowOptions:
      if (auto* o = opts.AsStablehloReduceWindowOptions(); o != nullptr) {
        fn(o->body_subgraph_index, SubgraphRefKind::kComputation);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloScatterOptions:
      if (auto* o = opts.AsStablehloScatterOptions(); o != nullptr) {
        fn(o->update_computation_subgraph_index, SubgraphRefKind::kComputation);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloSortOptions:
      if (auto* o = opts.AsStablehloSortOptions(); o != nullptr) {
        fn(o->comparator_subgraph_index, SubgraphRefKind::kComputation);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloCustomCallOptions:
      if (auto* o = opts.AsStablehloCustomCallOptions(); o != nullptr) {
        for (auto& index : o->called_computations) {
          fn(index, SubgraphRefKind::kComputation);
        }
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloWhileOptions:
      if (auto* o = opts.AsStablehloWhileOptions(); o != nullptr) {
        fn(o->cond_subgraph_index, SubgraphRefKind::kControlFlow);
        fn(o->body_subgraph_index, SubgraphRefKind::kControlFlow);
      }
      break;
    case ::tflite::BuiltinOptions2_StablehloCaseOptions:
      if (auto* o = opts.AsStablehloCaseOptions(); o != nullptr) {
        for (auto& index : o->branch_subgraph_indices) {
          fn(index, SubgraphRefKind::kControlFlow);
        }
      }
      break;
    default:
      break;
  }
}

bool TflOptionsHaveRefs(::tflite::BuiltinOptions type) {
  switch (type) {
    case ::tflite::BuiltinOptions_WhileOptions:
    case ::tflite::BuiltinOptions_IfOptions:
    case ::tflite::BuiltinOptions_CallOnceOptions:
      return true;
    default:
      return false;
  }
}

bool TflOptions2HaveRefs(::tflite::BuiltinOptions2 type) {
  switch (type) {
    case ::tflite::BuiltinOptions2_StableHLOCompositeOptions:
    case ::tflite::BuiltinOptions2_StablehloReduceOptions:
    case ::tflite::BuiltinOptions2_StablehloReduceWindowOptions:
    case ::tflite::BuiltinOptions2_StablehloScatterOptions:
    case ::tflite::BuiltinOptions2_StablehloSortOptions:
    case ::tflite::BuiltinOptions2_StablehloCustomCallOptions:
    case ::tflite::BuiltinOptions2_StablehloWhileOptions:
    case ::tflite::BuiltinOptions2_StablehloCaseOptions:
      return true;
    default:
      return false;
  }
}

}  // namespace

absl::string_view SubgraphRefKindName(SubgraphRefKind kind) {
  switch (kind) {
    case SubgraphRefKind::kDecomposition:
      return "decomposition";
    case SubgraphRefKind::kComputation:
      return "computation";
    case SubgraphRefKind::kControlFlow:
      return "control_flow";
  }
  return "unknown";
}

bool HasSubgraphRefs(const LiteRtOpT& op) {
  return TflOptionsHaveRefs(GetTflOptions(op).type) ||
         TflOptions2HaveRefs(GetTflOptions2(op).type);
}

bool HasRegionRefs(const LiteRtOpT& op) {
  if (!HasSubgraphRefs(op)) {
    return false;
  }
  for (const auto& ref : GetSubgraphRefs(op)) {
    if (IsRegionRefKind(ref.kind) && ref.index >= 0) {
      return true;
    }
  }
  return false;
}

SubgraphRefs GetSubgraphRefs(const LiteRtOpT& op) {
  SubgraphRefs refs;
  auto collect = [&refs](const int32_t index, SubgraphRefKind kind) {
    refs.push_back({index, kind});
  };
  VisitTflOptions(GetTflOptions(op), collect);
  VisitTflOptions2(GetTflOptions2(op), collect);
  return refs;
}

void UpdateSubgraphRefs(
    LiteRtOpT& op,
    absl::FunctionRef<void(int32_t& index, SubgraphRefKind kind)> fn) {
  if (TflOptionsHaveRefs(GetTflOptions(op).type)) {
    TflOptions opts = TakeTflOptions(op);
    VisitTflOptions(opts, fn);
    SetTflOptions(op, std::move(opts));
  }
  if (TflOptions2HaveRefs(GetTflOptions2(op).type)) {
    TflOptions2 opts = TakeTflOptions2(op);
    VisitTflOptions2(opts, fn);
    SetTflOptions2(op, std::move(opts));
  }
}

Expected<void> ValidateSubgraphRefs(const LiteRtModelT& model) {
  const size_t num_subgraphs = model.NumSubgraphs();
  for (size_t sg = 0; sg < num_subgraphs; ++sg) {
    const LiteRtSubgraphT& subgraph = model.Subgraph(sg);
    for (size_t op_ind = 0; op_ind < subgraph.Ops().size(); ++op_ind) {
      const LiteRtOpT& op = subgraph.Op(op_ind);
      if (!HasSubgraphRefs(op)) {
        continue;
      }
      for (const auto& ref : GetSubgraphRefs(op)) {
        const int32_t min_index =
            ref.kind == SubgraphRefKind::kDecomposition ? -1 : 0;
        if (ref.index < min_index || static_cast<int64_t>(ref.index) >=
                                         static_cast<int64_t>(num_subgraphs)) {
          return Error(
              kLiteRtStatusErrorInvalidArgument,
              absl::StrCat("Op ", op_ind, " (op code ", op.OpCode(),
                           ") in subgraph ", sg, " holds an invalid ",
                           SubgraphRefKindName(ref.kind),
                           " subgraph reference ", ref.index, " (model has ",
                           num_subgraphs, " subgraphs)"));
        }
      }
    }
  }
  return {};
}

Expected<SubgraphCallGraph> SubgraphCallGraph::Build(
    const LiteRtModelT& model) {
  LITERT_RETURN_IF_ERROR(ValidateSubgraphRefs(model));
  SubgraphCallGraph graph;
  const size_t num_subgraphs = model.NumSubgraphs();
  graph.callers_.resize(num_subgraphs);
  graph.callees_.resize(num_subgraphs);
  for (uint32_t sg = 0; sg < num_subgraphs; ++sg) {
    for (LiteRtOp op : model.Subgraph(sg).Ops()) {
      if (!HasSubgraphRefs(*op)) {
        continue;
      }
      for (const auto& ref : GetSubgraphRefs(*op)) {
        if (ref.index < 0) {
          continue;
        }
        const Edge edge{op, sg, static_cast<uint32_t>(ref.index), ref.kind};
        graph.callees_[sg].push_back(edge);
        graph.callers_[edge.callee].push_back(edge);
      }
    }
  }
  return graph;
}

bool SubgraphCallGraph::IsReferencedAs(uint32_t subgraph,
                                       SubgraphRefKind kind) const {
  return absl::c_any_of(callers_.at(subgraph),
                        [kind](const Edge& edge) { return edge.kind == kind; });
}

Expected<std::vector<uint32_t>> SubgraphCallGraph::TopologicalOrder() const {
  const size_t num_subgraphs = NumSubgraphs();
  std::vector<size_t> in_degree(num_subgraphs, 0);
  for (size_t sg = 0; sg < num_subgraphs; ++sg) {
    in_degree[sg] = callers_[sg].size();
  }
  // Min-heap so that the order is deterministic and stays as close as
  // possible to the original subgraph order.
  std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>>
      ready;
  for (uint32_t sg = 0; sg < num_subgraphs; ++sg) {
    if (in_degree[sg] == 0) {
      ready.push(sg);
    }
  }
  std::vector<uint32_t> order;
  order.reserve(num_subgraphs);
  while (!ready.empty()) {
    const uint32_t sg = ready.top();
    ready.pop();
    order.push_back(sg);
    for (const auto& edge : callees_[sg]) {
      if (--in_degree[edge.callee] == 0) {
        ready.push(edge.callee);
      }
    }
  }
  if (order.size() != num_subgraphs) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Subgraph references form a cycle");
  }
  return order;
}

}  // namespace litert::internal
