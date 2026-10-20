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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

// (caller, callee, kind) triple used to describe expected call graph edges.
using EdgeTuple = std::tuple<uint32_t, uint32_t, SubgraphRefKind>;

std::vector<EdgeTuple> AllEdges(const SubgraphCallGraph& graph) {
  std::vector<EdgeTuple> edges;
  for (uint32_t sg = 0; sg < graph.NumSubgraphs(); ++sg) {
    for (const auto& edge : graph.Callees(sg)) {
      edges.emplace_back(edge.caller, edge.callee, edge.kind);
    }
  }
  return edges;
}

LiteRtOpT& AddOpWithOptions2(LiteRtSubgraphT& subgraph, LiteRtOpCode code,
                             TflOptions2 options) {
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(code);
  SetTflOptions2(op, std::move(options));
  return op;
}

LiteRtOpT& AddOpWithOptions(LiteRtSubgraphT& subgraph, LiteRtOpCode code,
                            TflOptions options) {
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(code);
  SetTflOptions(op, std::move(options));
  return op;
}

TflOptions2 ReduceWindowOptions(int32_t body) {
  ::tflite::StablehloReduceWindowOptionsT opts;
  opts.body_subgraph_index = body;
  TflOptions2 options;
  options.Set(std::move(opts));
  return options;
}

TflOptions WhileOptions(int32_t cond, int32_t body) {
  ::tflite::WhileOptionsT opts;
  opts.cond_subgraph_index = cond;
  opts.body_subgraph_index = body;
  TflOptions options;
  options.Set(std::move(opts));
  return options;
}

TflOptions2 CompositeOptions(int32_t decomposition) {
  ::tflite::StableHLOCompositeOptionsT opts;
  opts.name = "composite";
  opts.decomposition_subgraph_index = decomposition;
  TflOptions2 options;
  options.Set(std::move(opts));
  return options;
}

int32_t ReduceWindowBody(const LiteRtOpT& op) {
  return GetTflOptions2(op)
      .AsStablehloReduceWindowOptions()
      ->body_subgraph_index;
}

//
// Option field coverage.
//

TEST(SubgraphRefsTest, NoRefs) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  EXPECT_FALSE(HasSubgraphRefs(op));
  EXPECT_FALSE(HasRegionRefs(op));
  EXPECT_TRUE(GetSubgraphRefs(op).empty());
}

TEST(SubgraphRefsTest, AllOptionFields) {
  struct Case {
    LiteRtOpCode code;
    TflOptions options;
    TflOptions2 options2;
    std::vector<SubgraphRefKind> kinds;
  };
  std::vector<Case> cases;
  {
    Case c{kLiteRtOpCodeTflWhile, WhileOptions(1, 2), {}, {}};
    c.kinds = {SubgraphRefKind::kControlFlow, SubgraphRefKind::kControlFlow};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::IfOptionsT opts;
    opts.then_subgraph_index = 1;
    opts.else_subgraph_index = 2;
    Case c{kLiteRtOpCodeTflIf, {}, {}, {}};
    c.options.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kControlFlow, SubgraphRefKind::kControlFlow};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::CallOnceOptionsT opts;
    opts.init_subgraph_index = 1;
    Case c{kLiteRtOpCodeTflCallOnce, {}, {}, {}};
    c.options.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kControlFlow};
    cases.push_back(std::move(c));
  }
  {
    Case c{kLiteRtOpCodeShloComposite, {}, CompositeOptions(1), {}};
    c.kinds = {SubgraphRefKind::kDecomposition};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloReduceOptionsT opts;
    opts.body_subgraph_index = 1;
    Case c{kLiteRtOpCodeShloReduce, {}, {}, {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kComputation};
    cases.push_back(std::move(c));
  }
  {
    Case c{kLiteRtOpCodeShloWindow, {}, ReduceWindowOptions(1), {}};
    c.kinds = {SubgraphRefKind::kComputation};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloScatterOptionsT opts;
    opts.update_computation_subgraph_index = 1;
    Case c{kLiteRtOpCodeShloScatter, {}, {}, {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kComputation};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloSortOptionsT opts;
    opts.comparator_subgraph_index = 1;
    Case c{kLiteRtOpCodeShloSort, {}, {}, {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kComputation};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloCustomCallOptionsT opts;
    opts.called_computations = {1, 2};
    Case c{kLiteRtOpCodeShloCustomCall, {}, {}, {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kComputation, SubgraphRefKind::kComputation};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloWhileOptionsT opts;
    opts.cond_subgraph_index = 1;
    opts.body_subgraph_index = 2;
    Case c{kLiteRtOpCodeShloWhile, {}, {}, {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kControlFlow, SubgraphRefKind::kControlFlow};
    cases.push_back(std::move(c));
  }
  {
    ::tflite::StablehloCaseOptionsT opts;
    opts.branch_subgraph_indices = {1, 2};
    Case c{static_cast<LiteRtOpCode>(::tflite::BuiltinOperator_STABLEHLO_CASE),
           {},
           {},
           {}};
    c.options2.Set(std::move(opts));
    c.kinds = {SubgraphRefKind::kControlFlow, SubgraphRefKind::kControlFlow};
    cases.push_back(std::move(c));
  }

  for (auto& c : cases) {
    SCOPED_TRACE(c.code);
    LiteRtModelT model;
    auto& sg = model.EmplaceSubgraph();
    auto& op = sg.EmplaceOp();
    op.SetOpCode(c.code);
    SetTflOptions(op, std::move(c.options));
    SetTflOptions2(op, std::move(c.options2));

    ASSERT_TRUE(HasSubgraphRefs(op));
    EXPECT_EQ(HasRegionRefs(op),
              c.kinds.front() != SubgraphRefKind::kDecomposition);

    const auto refs = GetSubgraphRefs(op);
    ASSERT_EQ(refs.size(), c.kinds.size());
    for (size_t i = 0; i < refs.size(); ++i) {
      EXPECT_EQ(refs[i].index, static_cast<int32_t>(i + 1));
      EXPECT_EQ(refs[i].kind, c.kinds[i]);
    }

    // Shift every reference and read it back.
    UpdateSubgraphRefs(op,
                       [](int32_t& index, SubgraphRefKind) { index += 10; });
    const auto updated = GetSubgraphRefs(op);
    ASSERT_EQ(updated.size(), c.kinds.size());
    for (size_t i = 0; i < updated.size(); ++i) {
      EXPECT_EQ(updated[i].index, static_cast<int32_t>(i + 11));
    }
  }
}

TEST(SubgraphRefsTest, CompositeWithoutDecompositionIsNotARegion) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  auto& op =
      AddOpWithOptions2(sg, kLiteRtOpCodeShloComposite, CompositeOptions(-1));
  EXPECT_TRUE(HasSubgraphRefs(op));
  EXPECT_FALSE(HasRegionRefs(op));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(model));
}

//
// Validation and call graph.
//

TEST(SubgraphRefsTest, ValidateRejectsOutOfRange) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  AddOpWithOptions2(sg, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));
  EXPECT_FALSE(ValidateSubgraphRefs(model));
  EXPECT_FALSE(SubgraphCallGraph::Build(model));

  model.EmplaceSubgraph();
  LITERT_EXPECT_OK(ValidateSubgraphRefs(model));
}

TEST(SubgraphRefsTest, ValidateRejectsNegativeRegion) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  AddOpWithOptions2(sg, kLiteRtOpCodeShloWindow, ReduceWindowOptions(-1));
  EXPECT_FALSE(ValidateSubgraphRefs(model));
}

TEST(SubgraphRefsTest, CallGraphRejectsCycle) {
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  auto& sg1 = model.EmplaceSubgraph();
  AddOpWithOptions(sg0, kLiteRtOpCodeTflWhile, WhileOptions(1, 1));
  AddOpWithOptions(sg1, kLiteRtOpCodeTflWhile, WhileOptions(0, 0));
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(model));
  EXPECT_FALSE(graph.TopologicalOrder());
}

TEST(SubgraphRefsTest, CallGraphTopologicalOrder) {
  // 0 -> 2 -> 1, 3 is unreferenced.
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  auto& sg2 = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  AddOpWithOptions(sg0, kLiteRtOpCodeTflWhile, WhileOptions(2, 2));
  AddOpWithOptions2(sg2, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));

  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(model));
  EXPECT_FALSE(graph.IsReferenced(0));
  EXPECT_TRUE(graph.IsReferenced(1));
  EXPECT_TRUE(graph.IsReferencedAs(1, SubgraphRefKind::kComputation));
  EXPECT_FALSE(graph.IsReferencedAs(1, SubgraphRefKind::kControlFlow));
  EXPECT_TRUE(graph.IsReferencedAs(2, SubgraphRefKind::kControlFlow));
  EXPECT_FALSE(graph.IsReferenced(3));
  LITERT_ASSERT_OK_AND_ASSIGN(auto order, graph.TopologicalOrder());
  EXPECT_THAT(order, ElementsAre(0, 2, 1, 3));
}

//
// TransferSubgraphTo / Yank.
//

TEST(SubgraphRefsTest, TransferRemapsRegionRefsInBothModels) {
  // sg0: reduce_window(body=1)          -> stays
  // sg1: reduce_window body             -> stays
  // sg2: partition with reduce_window(body=3) -> moved
  // sg3: body of sg2's op               -> moved
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  auto& sg1 = model.EmplaceSubgraph();
  auto& sg2 = model.EmplaceSubgraph();
  auto& sg3 = model.EmplaceSubgraph();
  auto& stay_op =
      AddOpWithOptions2(sg0, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));
  auto& move_op =
      AddOpWithOptions2(sg2, kLiteRtOpCodeShloWindow, ReduceWindowOptions(3));

  auto sliced = model.Yank({2, 3});

  EXPECT_THAT(model.Subgraphs(), ElementsAreArray({&sg0, &sg1}));
  EXPECT_THAT(sliced.Subgraphs(), ElementsAreArray({&sg2, &sg3}));
  EXPECT_EQ(ReduceWindowBody(stay_op), 1);
  EXPECT_EQ(ReduceWindowBody(move_op), 1);
  LITERT_EXPECT_OK(ValidateSubgraphRefs(model));
  LITERT_EXPECT_OK(ValidateSubgraphRefs(sliced));
}

TEST(SubgraphRefsTest, YankPreservesRequestedOrder) {
  // Entry partitions are created after the region bodies they reference; the
  // sliced model must list entries first.
  LiteRtModelT model;
  model.EmplaceSubgraph();               // 0: main
  auto& body = model.EmplaceSubgraph();  // 1: region body
  auto& p0 = model.EmplaceSubgraph();    // 2: partition 0
  auto& p1 = model.EmplaceSubgraph();    // 3: partition 1
  auto& op =
      AddOpWithOptions2(p1, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));

  auto sliced = model.Yank({2, 3, 1});

  EXPECT_THAT(sliced.Subgraphs(), ElementsAreArray({&p0, &p1, &body}));
  EXPECT_EQ(ReduceWindowBody(op), 2);
  LITERT_EXPECT_OK(ValidateSubgraphRefs(sliced));
  EXPECT_EQ(model.NumSubgraphs(), 1);
}

TEST(SubgraphRefsTest, TransferDropsCrossModelRefs) {
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  auto& op =
      AddOpWithOptions2(sg0, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));
  auto sliced = model.Yank({1});
  EXPECT_EQ(ReduceWindowBody(op), -1);
  EXPECT_FALSE(ValidateSubgraphRefs(model));
}

TEST(SubgraphRefsTest, TransferDropsMovedToRemainingRefs) {
  LiteRtModelT model;
  model.EmplaceSubgraph();
  auto& moved = model.EmplaceSubgraph();
  auto& op =
      AddOpWithOptions2(moved, kLiteRtOpCodeShloWindow, ReduceWindowOptions(0));
  auto sliced = model.Yank({1});
  EXPECT_EQ(ReduceWindowBody(op), -1);
  EXPECT_FALSE(ValidateSubgraphRefs(sliced));
}

TEST(SubgraphRefsTest, TransferIntoNonEmptyDest) {
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  auto& sg1 = model.EmplaceSubgraph();
  auto& sg2 = model.EmplaceSubgraph();
  auto& op =
      AddOpWithOptions2(sg2, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));

  LiteRtSubgraphT::Alloc dest;
  auto& existing = dest.EmplaceBack();
  model.TransferSubgraphTo(dest, {2, 1});

  EXPECT_THAT(model.Subgraphs(), ElementsAreArray({&sg0}));
  EXPECT_THAT(dest.Elements(), ElementsAreArray({&existing, &sg2, &sg1}));
  // Indices of transferred subgraphs account for the subgraphs already in
  // `dest`.
  EXPECT_EQ(ReduceWindowBody(op), 2);
}

TEST(SubgraphRefsTest, TransferIgnoresDuplicatesAndOutOfRangeIndices) {
  LiteRtModelT model;
  auto& sg0 = model.EmplaceSubgraph();
  auto& sg1 = model.EmplaceSubgraph();
  auto& op =
      AddOpWithOptions2(sg0, kLiteRtOpCodeShloWindow, ReduceWindowOptions(1));

  auto sliced = model.Yank({1, 1, 5});
  EXPECT_EQ(model.NumSubgraphs(), 1);
  EXPECT_THAT(sliced.Subgraphs(), ElementsAreArray({&sg1}));
  EXPECT_EQ(ReduceWindowBody(op), -1);

  // Only out of range indices: nothing is transferred.
  auto empty = model.Yank({7});
  EXPECT_EQ(empty.NumSubgraphs(), 0);
  EXPECT_EQ(model.NumSubgraphs(), 1);
}

TEST(SubgraphRefsTest, UpdateOpWithBothOptionTables) {
  LiteRtModelT model;
  auto& sg = model.EmplaceSubgraph();
  auto& op = sg.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflWhile);
  SetTflOptions(op, WhileOptions(1, 2));
  SetTflOptions2(op, CompositeOptions(3));

  EXPECT_THAT(GetSubgraphRefs(op), ::testing::SizeIs(3));
  UpdateSubgraphRefs(op, [](int32_t& index, SubgraphRefKind) { index *= 10; });
  std::vector<int32_t> indices;
  for (const auto& ref : GetSubgraphRefs(op)) {
    indices.push_back(ref.index);
  }
  EXPECT_THAT(indices, ElementsAre(10, 20, 30));
}

TEST(SubgraphRefsTest, KindNames) {
  EXPECT_EQ(SubgraphRefKindName(SubgraphRefKind::kDecomposition),
            "decomposition");
  EXPECT_EQ(SubgraphRefKindName(SubgraphRefKind::kComputation), "computation");
  EXPECT_EQ(SubgraphRefKindName(SubgraphRefKind::kControlFlow), "control_flow");
  EXPECT_FALSE(IsRegionRefKind(SubgraphRefKind::kDecomposition));
  EXPECT_TRUE(IsRegionRefKind(SubgraphRefKind::kComputation));
  EXPECT_TRUE(IsRegionRefKind(SubgraphRefKind::kControlFlow));
}

TEST(SubgraphRefsTest, NumEntrySubgraphs) {
  LiteRtModelT model;
  model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  EXPECT_EQ(model.NumEntrySubgraphs(), 2);
  model.SetNumEntrySubgraphs(1);
  EXPECT_EQ(model.NumEntrySubgraphs(), 1);
  model.SetNumEntrySubgraphs(5);
  EXPECT_EQ(model.NumEntrySubgraphs(), 2);
}

//
// Test models covering every kind of region produced by the converter.
//

struct RegionModelCase {
  std::string filename;
  size_t num_subgraphs;
  std::vector<EdgeTuple> edges;
};

class RegionModelTest : public ::testing::TestWithParam<RegionModelCase> {};

TEST_P(RegionModelTest, CallGraph) {
  const auto& param = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      LoadModelFromFile(litert::testing::GetTestFilePath(param.filename)));
  ASSERT_EQ(model->NumSubgraphs(), param.num_subgraphs);
  LITERT_ASSERT_OK(ValidateSubgraphRefs(*model));
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(*model));
  EXPECT_THAT(AllEdges(graph), UnorderedElementsAreArray(param.edges));
  LITERT_ASSERT_OK_AND_ASSIGN(auto order, graph.TopologicalOrder());
  EXPECT_EQ(order.front(), 0);
}

TEST_P(RegionModelTest, ReorderRoundTrip) {
  // Move the main subgraph to the end and back through serialization; all
  // region references must follow.
  const auto& param = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto model,
      LoadModelFromFile(litert::testing::GetTestFilePath(param.filename)));
  std::vector<size_t> reversed;
  for (size_t i = param.num_subgraphs; i > 0; --i) {
    reversed.push_back(i - 1);
  }
  auto yanked = model->Yank(reversed);
  ASSERT_EQ(yanked.NumSubgraphs(), param.num_subgraphs);
  LITERT_ASSERT_OK(ValidateSubgraphRefs(yanked));
  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, SubgraphCallGraph::Build(yanked));
  std::vector<EdgeTuple> expected;
  const uint32_t last = param.num_subgraphs - 1;
  for (const auto& [caller, callee, kind] : param.edges) {
    expected.emplace_back(last - caller, last - callee, kind);
  }
  EXPECT_THAT(AllEdges(graph), UnorderedElementsAreArray(expected));

  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized,
                              SerializeModel(std::move(yanked)));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded,
                              LoadModelFromBuffer(std::move(serialized)));
  LITERT_ASSERT_OK(ValidateSubgraphRefs(*reloaded));
  LITERT_ASSERT_OK_AND_ASSIGN(auto reloaded_graph,
                              SubgraphCallGraph::Build(*reloaded));
  EXPECT_THAT(AllEdges(reloaded_graph), UnorderedElementsAreArray(expected));
}

constexpr auto kComp = SubgraphRefKind::kComputation;
constexpr auto kCf = SubgraphRefKind::kControlFlow;
constexpr auto kDecomp = SubgraphRefKind::kDecomposition;

INSTANTIATE_TEST_SUITE_P(
    RegionModels, RegionModelTest,
    ::testing::Values(
        RegionModelCase{"shlo_reduce_window_region.tflite", 2, {{0, 1, kComp}}},
        RegionModelCase{"shlo_reduce_region.tflite", 2, {{0, 1, kComp}}},
        RegionModelCase{"shlo_sort_region.tflite", 2, {{0, 1, kComp}}},
        RegionModelCase{"shlo_scatter_region.tflite", 2, {{0, 1, kComp}}},
        RegionModelCase{
            "shlo_while_region.tflite", 3, {{0, 1, kCf}, {0, 2, kCf}}},
        RegionModelCase{
            "tfl_while_region.tflite", 3, {{0, 1, kCf}, {0, 2, kCf}}},
        RegionModelCase{"tfl_if_region.tflite", 3, {{0, 1, kCf}, {0, 2, kCf}}},
        RegionModelCase{"tfl_call_once_region.tflite", 2, {{0, 1, kCf}}},
        RegionModelCase{"tfl_while_shared_body.tflite",
                        3,
                        {{0, 1, kCf}, {0, 2, kCf}, {0, 1, kCf}, {0, 2, kCf}}},
        RegionModelCase{"nested_region.tflite",
                        4,
                        {{0, 1, kCf}, {0, 2, kCf}, {2, 3, kComp}}},
        RegionModelCase{
            "region_in_composite.tflite", 3, {{0, 1, kDecomp}, {1, 2, kComp}}},
        RegionModelCase{
            "npu_call_region.tflite", 3, {{0, 1, kDecomp}, {1, 2, kComp}}},
        RegionModelCase{
            "multi_island_reduce_window.tflite", 2, {{0, 1, kComp}}}),
    [](const ::testing::TestParamInfo<RegionModelCase>& info) {
      std::string name(info.param.filename);
      return name.substr(0, name.find('.'));
    });

}  // namespace
}  // namespace litert::internal
