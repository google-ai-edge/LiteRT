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

#include "litert/vendors/qualcomm/dispatch/active_functions.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl

namespace litert::qnn {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(SelectGraphsToEnableTest, NoActiveFunctionsEnablesAll) {
  const std::vector<std::string> graphs = {"a", "b"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, {}, "a"), IsEmpty());
}

TEST(SelectGraphsToEnableTest, SelectsIntersection) {
  const std::vector<std::string> graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> active = {"a", "c", "other"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "a"), ElementsAre("a", "c"));
}

TEST(SelectGraphsToEnableTest, AlwaysIncludesRequestedFunction) {
  const std::vector<std::string> graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> active = {"a"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "b"), ElementsAre("a", "b"));
}

TEST(SelectGraphsToEnableTest, AllActiveEnablesAll) {
  const std::vector<std::string> graphs = {"a", "b"};
  const absl::flat_hash_set<std::string> active = {"a", "b"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "a"), IsEmpty());
}

TEST(SelectGraphsToEnableTest, EmptyBinaryGraphsReturnsEmpty) {
  const absl::flat_hash_set<std::string> active = {"a"};
  EXPECT_THAT(SelectGraphsToEnable({}, active, "a"), IsEmpty());
}

TEST(SelectGraphsToEnableTest, SingleGraphBinaryAlwaysEnablesAll) {
  const std::vector<std::string> graphs = {"single"};
  const absl::flat_hash_set<std::string> active = {"single"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "single"), IsEmpty());
}

TEST(SelectGraphsToEnableTest,
     RequestedFunctionNotInActiveFunctionsStillSelected) {
  const std::vector<std::string> graphs = {"a", "b", "c"};
  const absl::flat_hash_set<std::string> active = {"b"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "c"), ElementsAre("b", "c"));
}

TEST(SelectGraphsToEnableTest, ActiveFunctionsFromOtherBinaryIgnored) {
  const std::vector<std::string> graphs = {"a", "b"};
  const absl::flat_hash_set<std::string> active = {"other_1", "other_2"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "a"), ElementsAre("a"));
}

TEST(SelectGraphsToEnableTest, NoMatchingGraphsReturnsEmpty) {
  const std::vector<std::string> graphs = {"a", "b"};
  const absl::flat_hash_set<std::string> active = {"other_1", "other_2"};
  EXPECT_THAT(SelectGraphsToEnable(graphs, active, "not_in_graphs"), IsEmpty());
}

}  // namespace
}  // namespace litert::qnn
