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

#include "litert/core/model/subgraph_call_graph.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/matchers.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

void AddWhile(LiteRtSubgraphT& subgraph, int32_t cond, int32_t body) {
  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflWhile);
  tflite::WhileOptionsT while_opts;
  while_opts.cond_subgraph_index = cond;
  while_opts.body_subgraph_index = body;
  TflOptions options;
  options.type = tflite::BuiltinOptions_WhileOptions;
  options.Set(std::move(while_opts));
  SetTflOptions(op, std::move(options));
}

TEST(SubgraphCallGraphTest, BuildsDeduplicatedCallees) {
  LiteRtModelT model;
  auto& main = model.EmplaceSubgraph();
  auto& body = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  AddWhile(main, /*cond=*/2, /*body=*/1);
  AddWhile(main, /*cond=*/2, /*body=*/1);
  AddWhile(body, /*cond=*/2, /*body=*/2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, BuildSubgraphCallGraph(model));

  ASSERT_EQ(graph.callees.size(), 3);
  EXPECT_THAT(graph.callees[0], ElementsAre(2, 1));
  EXPECT_THAT(graph.callees[1], ElementsAre(2));
  EXPECT_THAT(graph.callees[2], IsEmpty());
  EXPECT_THAT(SubgraphClosure(graph, {1}), UnorderedElementsAre(1, 2));
  EXPECT_THAT(TopologicalOrder(graph), ElementsAre(0, 1, 2));
}

TEST(SubgraphCallGraphTest, TopologicalOrderPlacesCallersFirst) {
  LiteRtModelT model;
  model.EmplaceSubgraph();
  auto& body = model.EmplaceSubgraph();
  model.EmplaceSubgraph();
  auto& main = model.EmplaceSubgraph();
  // main(3) -> body(1) -> 0, main(3) -> 2.
  AddWhile(main, /*cond=*/2, /*body=*/1);
  AddWhile(body, /*cond=*/0, /*body=*/0);

  LITERT_ASSERT_OK_AND_ASSIGN(auto graph, BuildSubgraphCallGraph(model));

  EXPECT_THAT(TopologicalOrder(graph), ElementsAre(3, 2, 1, 0));
}

TEST(SubgraphCallGraphTest, RejectsOutOfRangeReference) {
  LiteRtModelT model;
  auto& main = model.EmplaceSubgraph();
  AddWhile(main, /*cond=*/0, /*body=*/5);

  auto graph = BuildSubgraphCallGraph(model);

  ASSERT_FALSE(graph.HasValue());
  EXPECT_EQ(graph.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(SubgraphCallGraphTest, RejectsCycle) {
  LiteRtModelT model;
  auto& main = model.EmplaceSubgraph();
  auto& body = model.EmplaceSubgraph();
  AddWhile(main, /*cond=*/1, /*body=*/1);
  AddWhile(body, /*cond=*/0, /*body=*/0);

  auto graph = BuildSubgraphCallGraph(model);

  ASSERT_FALSE(graph.HasValue());
  EXPECT_EQ(graph.Error().Status(), kLiteRtStatusErrorUnsupported);
}

TEST(SubgraphCallGraphTest, RejectsSelfReference) {
  LiteRtModelT model;
  auto& main = model.EmplaceSubgraph();
  AddWhile(main, /*cond=*/0, /*body=*/0);

  EXPECT_FALSE(BuildSubgraphCallGraph(model).HasValue());
}

}  // namespace
}  // namespace litert::internal
