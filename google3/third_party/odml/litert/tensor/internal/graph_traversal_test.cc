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

#include "third_party/odml/litert/tensor/internal/graph_traversal.h"

#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/internal/arithmetic_helpers.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/utils/source_location.h"

namespace litert {
namespace tensor {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// A simple operation that does nothing.
class NoOp : public graph::Operation {
 public:
  absl::string_view GetName() const override { return "NoOp"; }
};

TEST(GetExecutionPlan, LinearGraph) {
  Tensor a = Tensor().SetName("a");
  auto op_a = std::make_shared<NoOp>();
  op_a->outputs_group = a.GetRaw().group;
  a.GetRaw().group->producer = op_a;

  Tensor b = Abs(a);
  b.SetName("b");
  Tensor c = Abs(b);
  c.SetName("c");

  const auto op_c = graph::GetProducer(c.GetRaw()).value();
  const auto op_b = graph::GetProducer(b.GetRaw()).value();

  auto plan_or = GetExecutionPlan({c});
  ASSERT_TRUE(plan_or.ok());
  const auto plan = plan_or.value();

  EXPECT_THAT(plan, ElementsAre(op_a.get(), op_b.get(), op_c.get()));
}

TEST(GetExecutionPlan, DiamondGraph) {
  Tensor a = Tensor().SetName("a");
  auto op_a = std::make_shared<NoOp>();
  op_a->outputs_group = a.GetRaw().group;
  a.GetRaw().group->producer = op_a;

  Tensor b = Abs(a);
  b.SetName("b");
  Tensor c = Neg(a);
  c.SetName("c");
  Tensor d = Add(b, c);
  d.SetName("d");

  const auto op_d = graph::GetProducer(d.GetRaw()).value();
  const auto op_c = graph::GetProducer(c.GetRaw()).value();
  const auto op_b = graph::GetProducer(b.GetRaw()).value();

  auto plan_or = GetExecutionPlan({d});
  ASSERT_TRUE(plan_or.ok());
  const auto plan = plan_or.value();

  EXPECT_THAT(plan, UnorderedElementsAre(op_d.get(), op_c.get(), op_b.get(),
                                         op_a.get()));
}

TEST(GetExecutionPlan, MultipleOutputs) {
  auto op_a = std::make_shared<NoOp>();
  TensorHandle a_handle = AddOutput(op_a, source_location::current());
  Tensor a = TensorHandle(a_handle);
  a.SetName("a");

  Tensor b = Abs(a);
  b.SetName("b");
  Tensor c = Neg(a);
  c.SetName("c");

  const auto op_c = graph::GetProducer(c.GetRaw()).value();
  const auto op_b = graph::GetProducer(b.GetRaw()).value();

  auto plan_or = GetExecutionPlan({b, c});
  ASSERT_TRUE(plan_or.ok());
  const auto plan = plan_or.value();
  const std::vector<const graph::Operation*> plan_vec(plan.begin(), plan.end());

  // op_a must come after op_b and op_c. The relative order of op_b and op_c is
  // not important for correctness, but should be deterministic.
  ASSERT_EQ(plan_vec.size(), 3);
  EXPECT_THAT(plan_vec,
              UnorderedElementsAre(op_a.get(), op_b.get(), op_c.get()));
}

TEST(GetExecutionPlan, UnconnectedGraph) {
  Tensor a = Tensor().SetName("a");
  auto op_a = std::make_shared<NoOp>();
  op_a->outputs_group = a.GetRaw().group;
  a.GetRaw().group->producer = op_a;

  // Unconnected operation.
  Tensor b = Tensor().SetName("b");
  auto op_b = std::make_shared<NoOp>();
  op_b->outputs_group = b.GetRaw().group;
  b.GetRaw().group->producer = op_b;

  auto plan_or = GetExecutionPlan({a});
  ASSERT_TRUE(plan_or.ok());
  const auto plan = plan_or.value();

  EXPECT_THAT(plan, ElementsAre(op_a.get()));
}

TEST(GetExecutionPlan, EmptyOutputs) {
  auto plan_or = GetExecutionPlan({});
  ASSERT_TRUE(plan_or.ok());
  const auto plan = plan_or.value();
  EXPECT_THAT(plan, IsEmpty());
}

TEST(GetExecutionPlan, CycleDetection) {
  // Create a cycle manually.
  auto tensor_a = graph::NewTensor(source_location::current());
  auto tensor_b = graph::NewTensor(source_location::current());
  auto op_a = std::make_shared<NoOp>();
  auto op_b = std::make_shared<NoOp>();
  op_a->inputs = {tensor_b};
  op_b->inputs = {tensor_a};
  op_a->outputs_group = tensor_a.group;
  op_b->outputs_group = tensor_b.group;
  // Link producers:
  tensor_a.group->producer = op_a;
  tensor_b.group->producer = op_b;

  const auto plan = GetExecutionPlan({TensorHandle(tensor_a)});
  EXPECT_FALSE(plan.ok());
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(plan.status().message(),
                                "Cycle detected in the graph."));

  // Break the cycle to avoid memory leaks.
  op_a->inputs.clear();
  op_b->inputs.clear();
}

}  // namespace
}  // namespace tensor
}  // namespace litert
