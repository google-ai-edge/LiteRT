// Copyright 2025 Google LLC.
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

#include "litert/cc/options/darwinn_options.h"

#include <utility>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"

namespace litert {
namespace google_tensor {
namespace {

// Test creating LiteRT options with Darwinn runtime options
TEST(DarwinnOptionsTest, IntegrateWithLiteRtOptions) {
  // Create base LiteRT options
  LiteRtOptions litert_options = nullptr;
  ASSERT_EQ(LiteRtCreateOptions(&litert_options), kLiteRtStatusOk);

  // Create and configure Darwinn runtime options
  auto darwinn_options = DarwinnRuntimeOptions::Create();
  ASSERT_TRUE(darwinn_options);

  EXPECT_TRUE(darwinn_options->SetInferencePowerState(3));
  EXPECT_TRUE(darwinn_options->SetInferenceMemoryPowerState(2));
  EXPECT_TRUE(darwinn_options->SetInferencePriority(10));
  EXPECT_TRUE(darwinn_options->SetAtomicInference(false));
  EXPECT_TRUE(darwinn_options->SetPreferCoherent(false));

  // Create a C++ wrapper for the LiteRT options
  Options cc_options(litert_options, OwnHandle::kNo);

  // Add the Darwinn options to the LiteRT options
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(*darwinn_options)));

  // Verify we can find the Darwinn options in the chain
  auto opaque_options = cc_options.GetOpaqueOptions();
  ASSERT_TRUE(opaque_options);
  auto found_darwinn =
      FindOpaqueOptions<DarwinnRuntimeOptions>(*opaque_options);
  ASSERT_TRUE(found_darwinn);

  // Verify the values are preserved
  auto power_state = found_darwinn->GetInferencePowerState();
  ASSERT_TRUE(power_state);
  EXPECT_EQ(*power_state, 3);

  auto priority = found_darwinn->GetInferencePriority();
  ASSERT_TRUE(priority);
  EXPECT_EQ(*priority, 10);

  auto prefer_coherent = found_darwinn->GetPreferCoherent();
  ASSERT_TRUE(prefer_coherent);
  EXPECT_EQ(*prefer_coherent, false);

  // Clean up
  LiteRtDestroyOptions(litert_options);
}

}  // namespace
}  // namespace google_tensor
}  // namespace litert
