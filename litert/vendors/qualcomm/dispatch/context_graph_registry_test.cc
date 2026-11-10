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

using ::testing::UnorderedElementsAre;
using ::testing::litert::IsError;

struct FakeContext {};

TEST(ContextGraphRegistryTest, PlansExpectedAndRequiredGraphs) {
  const std::vector<std::string> available_graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected_graphs = {"a", "other"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, expected_graphs);

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("b"));
  EXPECT_THAT(graphs, UnorderedElementsAre("a", "b"));
}

TEST(ContextGraphRegistryTest, EmptyExpectedGraphsPlansAllAvailableGraphs) {
  const std::vector<std::string> available_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, {});

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("a"));
  EXPECT_THAT(graphs, UnorderedElementsAre("a", "b"));
}

TEST(ContextGraphRegistryTest, RejectsUnavailableRequiredGraph) {
  const std::vector<std::string> available_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, {"a"});

  EXPECT_THAT(registry.PlanAdditionalGraphsToEnable("other"),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(ContextGraphRegistryTest, DoesNotPlanGraphsAlreadyRegistered) {
  const std::vector<std::string> available_graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, expected_graphs);
  const std::vector<std::string> enabled_graphs = {"a", "b"};
  auto& context =
      registry.RegisterContext(std::make_unique<FakeContext>(), enabled_graphs);

  EXPECT_EQ(registry.FindContext("a"), &context);
  EXPECT_EQ(registry.FindContext("b"), &context);
  EXPECT_EQ(registry.FindContext("c"), nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("a"));
  EXPECT_TRUE(graphs.empty());
}

TEST(ContextGraphRegistryTest, RegisterAllGraphsMapsEveryAvailableGraph) {
  const std::vector<std::string> available_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, {"a"});

  auto& context = registry.RegisterContextWithAllAvailableGraphs(
      std::make_unique<FakeContext>());

  EXPECT_EQ(registry.FindContext("a"), &context);
  EXPECT_EQ(registry.FindContext("b"), &context);
  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("b"));
  EXPECT_TRUE(graphs.empty());
}

}  // namespace
}  // namespace litert::qnn
