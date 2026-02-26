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

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/google/litert_darwinn_options.h"
#include "litert/vendors/google_tensor/dispatch/litert_darwinn_options.h"

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
  const char* identifier;
  void* payload;
  void (*payload_deleter)(void*);
  EXPECT_TRUE(darwinn_options->GetOpaqueDarwinnRuntimeOptionsData(
      &identifier, &payload, &payload_deleter));

  auto darwinn_opaque_options =
      ::litert::OpaqueOptions::Create(identifier, payload, payload_deleter);
  ASSERT_TRUE(darwinn_opaque_options);
  EXPECT_TRUE(cc_options.AddOpaqueOptions(std::move(*darwinn_opaque_options)));

  // Verify we can find the Darwinn options in the chain
  auto opaque_options = cc_options.GetOpaqueOptions();
  ASSERT_TRUE(opaque_options);
  auto darwinn_options_data = litert::FindOpaqueData<const char>(
      *opaque_options, litert::LiteRtDarwinnRuntimeOptionsT::Identifier());
  ASSERT_TRUE(darwinn_options_data);
  litert::LiteRtDarwinnRuntimeOptionsT found_darwinn;
  absl::string_view data_str(*darwinn_options_data);
  EXPECT_EQ(litert::internal::ParseLiteRtDarwinnRuntimeOptions(
                data_str.data(), data_str.size(), &found_darwinn),
            kLiteRtStatusOk);

  // Verify the values are preserved
  EXPECT_EQ(found_darwinn.inference_power_state, 3);
  EXPECT_EQ(found_darwinn.inference_priority, 10);
  EXPECT_EQ(found_darwinn.prefer_coherent, false);

  // Clean up
  LiteRtDestroyOptions(litert_options);
}

}  // namespace
}  // namespace google_tensor
}  // namespace litert
