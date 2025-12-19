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

#include "litert/cc/options/litert_darwinn_options.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_opaque_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
using testing::litert::IsOkAndHolds;

TEST(DarwinnRuntimeOptionsTest, CreateAndFindRuntimeOptions) {
  // Create runtime options
  LITERT_ASSERT_OK_AND_ASSIGN(auto runtime_options,
                              DarwinnRuntimeOptions::Create());

  // Set some values
  LITERT_EXPECT_OK(runtime_options.SetInferencePowerState(3));
  LITERT_EXPECT_OK(runtime_options.SetInferencePriority(5));

  // Verify the values
  EXPECT_THAT(runtime_options.GetInferencePowerState(), IsOkAndHolds(3));
  EXPECT_THAT(runtime_options.GetInferencePriority(), IsOkAndHolds(5));

  // Find options in the list
  OpaqueOptions opaque_options = std::move(runtime_options);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto found_options,
      FindOpaqueOptions<DarwinnRuntimeOptions>(opaque_options));

  // Verify found options have the same values
  EXPECT_THAT(found_options.GetInferencePowerState(), IsOkAndHolds(3));
}

TEST(DarwinnRuntimeOptionsTest, CheckDefaultValues) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, DarwinnRuntimeOptions::Create());

  // Check default values
  EXPECT_THAT(options.GetInferencePowerState(), IsOkAndHolds(6));
  EXPECT_THAT(options.GetInferenceMemoryPowerState(), IsOkAndHolds(3));
  EXPECT_THAT(options.GetInferencePriority(), IsOkAndHolds(-1));
  EXPECT_THAT(options.GetPreferCoherent(), IsOkAndHolds(false));
}

}  // namespace
}  // namespace litert
