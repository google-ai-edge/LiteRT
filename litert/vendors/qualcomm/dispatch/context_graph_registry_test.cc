// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/dispatch/context_graph_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/test/matchers.h"

namespace litert::qnn {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::litert::IsError;

struct FakeContext {};

TEST(ContextGraphRegistryTest, PlansExpectedAndRequiredGraphs) {
  const absl::flat_hash_set<std::string> available_graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected_graphs = {"a", "other"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, expected_graphs);

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("b"));
  EXPECT_THAT(graphs, UnorderedElementsAre("a", "b"));
}

TEST(ContextGraphRegistryTest, EmptyExpectedGraphsPlansAllAvailableGraphs) {
  const absl::flat_hash_set<std::string> available_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, {});

  LITERT_ASSERT_OK_AND_ASSIGN(auto graphs,
                              registry.PlanAdditionalGraphsToEnable("a"));
  EXPECT_THAT(graphs, UnorderedElementsAre("a", "b"));
}

TEST(ContextGraphRegistryTest, RejectsUnavailableRequiredGraph) {
  const absl::flat_hash_set<std::string> available_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, {"a"});

  EXPECT_THAT(registry.PlanAdditionalGraphsToEnable("other"),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(ContextGraphRegistryTest, DoesNotPlanGraphsAlreadyRegistered) {
  const absl::flat_hash_set<std::string> available_graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> expected_graphs = {"a", "b"};
  ContextGraphRegistry<FakeContext> registry(available_graphs, expected_graphs);
  const std::array<std::string, 2> enabled_graphs = {"a", "b"};
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
  const absl::flat_hash_set<std::string> available_graphs = {"a", "b"};
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
