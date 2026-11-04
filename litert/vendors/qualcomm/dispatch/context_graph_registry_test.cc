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

#include "litert/vendors/qualcomm/dispatch/context_graph_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/test/matchers.h"

namespace litert::qnn {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::testing::litert::IsError;

struct FakeContext {};

TEST(ContextGraphRegistryTest, PlansExpectedAndRequiredGraphs) {
  const std::vector<std::string> available = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected = {"a", "other"};
  ContextGraphRegistry<FakeContext> registry(available, expected);

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("b"));
  EXPECT_THAT(graphs, UnorderedElementsAre("a", "b"));
}

TEST(ContextGraphRegistryTest, EmptyPlanMeansEnableAllGraphs) {
  const std::vector<std::string> available = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available, {});

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("a"));
  EXPECT_THAT(graphs, IsEmpty());
}

TEST(ContextGraphRegistryTest, RejectsRequiredGraphOutsideBinary) {
  const std::vector<std::string> available = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available, {"a"});

  EXPECT_THAT(registry.PlanAdditionalGraphsToEnable("other"), IsError());
}

TEST(ContextGraphRegistryTest, DoesNotPlanGraphsAlreadyRegistered) {
  const std::vector<std::string> available = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available, expected);
  auto first_context = std::make_unique<FakeContext>();
  auto* first_context_ptr = first_context.get();
  const std::vector<std::string> first_graphs = {"a", "b"};
  registry.RegisterContext(std::move(first_context), first_graphs);

  EXPECT_EQ(registry.FindContext("a"), first_context_ptr);
  EXPECT_EQ(registry.FindContext("c"), nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("c"));
  EXPECT_THAT(graphs, ElementsAre("c"));
}

TEST(ContextGraphRegistryTest, RegisterAllGraphsMapsEveryAvailableGraph) {
  const std::vector<std::string> available = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available, {"a"});
  auto context = std::make_unique<FakeContext>();
  auto* context_ptr = context.get();

  registry.RegisterContextWithAllAvailableGraphs(std::move(context));

  EXPECT_EQ(registry.FindContext("a"), context_ptr);
  EXPECT_EQ(registry.FindContext("b"), context_ptr);
}

}  // namespace
}  // namespace litert::qnn
