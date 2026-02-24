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

#include "litert/vendors/examples/example_common.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/test/matchers.h"

namespace litert::example {
namespace {

using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;

static constexpr absl::string_view kSingleOpExampleGraph = R"(version:1
inputs:0,1
outputs:2
const_map:
tensors:[2x2],[2x2],[2x2]
ops:mul(0,1)(2))";

static constexpr absl::string_view kMultipleOpsExampleGraph = R"(version:1
inputs:0,1
outputs:3
const_map:
tensors:[2x2],[2x2],[2x2],[2x2]
ops:mul(0,1)(2)~sub(2,2)(3))";

TEST(ExampleTensorReprTest, Works) {
  ExampleTensor t(Dims{2, 2});
  EXPECT_EQ(absl::StrFormat("%v", t), "[2x2]");
}

TEST(ExampleOpReprTest, Works) {
  ExampleOp op;
  op.code = OpCode::kMul;
  op.inputs = Inds{0, 1};
  op.outputs = Inds{2};
  EXPECT_EQ(absl::StrFormat("%v", op), "mul(0,1)(2)");
}

TEST(SerializeExampleGraphTest, OneOp) {
  ExampleGraph graph;
  const auto t0 = graph.EmplaceTensor(Dims{2, 2});
  const auto t1 = graph.EmplaceTensor(Dims{2, 2});
  const auto t2 = graph.EmplaceTensor(Dims{2, 2});
  graph.EmplaceOp(OpCode::kMul, Inds{t0, t1}, Inds{t2});
  graph.SetInputs(t0, t1);
  graph.SetOutputs(t2);
  graph.SetVersion("1");
  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized, graph.Serialize());
  ASSERT_EQ(serialized.StrView(), kSingleOpExampleGraph);
}

TEST(SerializeExampleGraphTest, MultipleOps) {
  ExampleGraph graph;
  const auto t0 = graph.EmplaceTensor(Dims{2, 2});
  const auto t1 = graph.EmplaceTensor(Dims{2, 2});
  const auto t2 = graph.EmplaceTensor(Dims{2, 2});
  const auto t3 = graph.EmplaceTensor(Dims{2, 2});
  graph.EmplaceOp(OpCode::kMul, Inds{t0, t1}, Inds{t2});
  graph.EmplaceOp(OpCode::kSub, Inds{t2, t2}, Inds{t3});
  graph.SetInputs(t0, t1);
  graph.SetOutputs(t3);
  graph.SetVersion("1");
  LITERT_ASSERT_OK_AND_ASSIGN(auto serialized, graph.Serialize());
  ASSERT_EQ(serialized.StrView(), kMultipleOpsExampleGraph);
}

TEST(ParseTensorTest, Works) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor, ExampleTensor::Parse("[2x2]"));
  ASSERT_THAT(tensor.dims, ElementsAreArray({2, 2}));
}

TEST(ParseOpTest, Works) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto op, ExampleOp::Parse("mul(0,1)(2)"));
  ASSERT_EQ(op.code, OpCode::kMul);
  ASSERT_THAT(op.inputs, ElementsAreArray({0, 1}));
  ASSERT_THAT(op.outputs, ElementsAreArray({2}));
}

TEST(ParseExampleGraphTest, OneOp) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto graph,
      ExampleGraph::Parse(BufferRef<uint8_t>(kSingleOpExampleGraph.data(),
                                             kSingleOpExampleGraph.size())));
  EXPECT_THAT(graph.Inputs(), ElementsAreArray({0, 1}));
  EXPECT_THAT(graph.Outputs(), ElementsAreArray({2}));
  ASSERT_EQ(graph.Tensors().size(), 3);
  EXPECT_THAT(graph.version(), Eq("1"));
  EXPECT_THAT(graph.Tensors(), Each(Eq(ExampleTensor(Dims{2, 2}))));
  EXPECT_THAT(graph.Ops(),
              ElementsAreArray({ExampleOp{OpCode::kMul, Inds{0, 1}, Inds{2}}}));
}

TEST(ParseExampleGraphTest, MultipleOps) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto graph,
      ExampleGraph::Parse(BufferRef<uint8_t>(kMultipleOpsExampleGraph.data(),
                                             kMultipleOpsExampleGraph.size())));

  EXPECT_THAT(graph.Inputs(), ElementsAreArray({0, 1}));
  EXPECT_THAT(graph.Outputs(), ElementsAreArray({3}));
  ASSERT_EQ(graph.Tensors().size(), 4);
  EXPECT_THAT(graph.version(), Eq("1"));
  EXPECT_THAT(graph.Tensors(), Each(Eq(ExampleTensor(Dims{2, 2}))));
  EXPECT_THAT(graph.Ops(),
              ElementsAreArray({ExampleOp{OpCode::kMul, Inds{0, 1}, Inds{2}},
                                ExampleOp{OpCode::kSub, Inds{2, 2}, Inds{3}}}));
}

TEST(ExecuteTest, SingleOp) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto graph,
      ExampleGraph::Parse(BufferRef<uint8_t>(kSingleOpExampleGraph.data(),
                                             kSingleOpExampleGraph.size())));
  const std::vector<float> lhs = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> rhs = {5.0f, 6.0f, 7.0f, 8.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto outputs, Execute(graph, {lhs, rhs}));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(outputs.front(), ElementsAreArray({5.0f, 12.0f, 21.0f, 32.0f}));
}

TEST(ExecuteTest, MultipleOps) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto graph,
      ExampleGraph::Parse(BufferRef<uint8_t>(kMultipleOpsExampleGraph.data(),
                                             kMultipleOpsExampleGraph.size())));
  const std::vector<float> lhs = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> rhs = {5.0f, 6.0f, 7.0f, 8.0f};
  LITERT_ASSERT_OK_AND_ASSIGN(auto outputs, Execute(graph, {lhs, rhs}));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(outputs.front(), Each(Eq(0.0f)));
}

}  // namespace
}  // namespace litert::example
