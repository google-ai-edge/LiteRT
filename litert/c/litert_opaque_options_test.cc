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

#include "litert/c/litert_opaque_options.h"

#include <cstdint>
#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/version.h"
#include "litert/test/matchers.h"

namespace {

using testing::StrEq;
using testing::litert::IsError;

static constexpr int kDummyOptionValue = 3;

struct DummyOpaqueOptions {
  static constexpr const char* const kIdentifier = "dummy-accelerator";

  int dummy_option = kDummyOptionValue;

  // Allocates and sets the basic structure for the accelerator options.
  static litert::Expected<LiteRtOpaqueOptions> CreateOptions() {
    auto* payload = new DummyOpaqueOptions;
    auto payload_destructor = [](void* payload) {
      delete reinterpret_cast<DummyOpaqueOptions*>(payload);
    };
    return CreateOptions(kIdentifier, payload, payload_destructor);
  }

  static litert::Expected<LiteRtOpaqueOptions> CreateOptions(
      const char* identifier, void* payload,
      void (*payload_destructor)(void*)) {
    LiteRtOpaqueOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
        identifier, payload, payload_destructor, &options));
    return options;
  }
};

class LiteRtOpaqueOptionsTest : public testing::Test {
 public:
  void SetUp() override {
    auto options = DummyOpaqueOptions::CreateOptions();
    EXPECT_TRUE(options);
    options_ = *options;
  }

  void TearDown() override {
    LiteRtDestroyOpaqueOptions(options_);
    options_ = nullptr;
  }

  LiteRtOpaqueOptions options_ = nullptr;
};

TEST_F(LiteRtOpaqueOptionsTest, CreateAndDestroyDoesntLeak) {}

TEST_F(LiteRtOpaqueOptionsTest, GetIdentifier) {
  const char* identifier = nullptr;
  LITERT_EXPECT_OK(LiteRtGetOpaqueOptionsIdentifier(options_, &identifier));
  EXPECT_THAT(identifier, StrEq(DummyOpaqueOptions::kIdentifier));
  EXPECT_THAT(LiteRtGetOpaqueOptionsIdentifier(nullptr, &identifier),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_EQ(LiteRtGetOpaqueOptionsIdentifier(options_, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtOpaqueOptionsTest, GetPayloadHashFailsIfUnset) {
  uint64_t hash = 0;
  // Fails as 'unsupported' because no hash function is set.
  EXPECT_THAT(LiteRtGetOpaqueOptionsHash(options_, &hash),
              IsError(kLiteRtStatusErrorUnsupported));
}

TEST_F(LiteRtOpaqueOptionsTest, GetPayloadHashWorks) {
  auto std_hash = [](const void* payload) -> uint64_t {
    const DummyOpaqueOptions* dummy_options =
        reinterpret_cast<const DummyOpaqueOptions*>(payload);
    return std::hash<int>()(dummy_options->dummy_option);
  };
  LITERT_EXPECT_OK(LiteRtSetOpaqueOptionsHash(options_, std_hash));

  uint64_t hash = 0;
  LITERT_EXPECT_OK(LiteRtGetOpaqueOptionsHash(options_, &hash));
  EXPECT_EQ(hash, std::hash<int>()(kDummyOptionValue));
}

TEST_F(LiteRtOpaqueOptionsTest, CreatingAndDestroyingAListWorks) {
  auto appended_options1 = DummyOpaqueOptions::CreateOptions();
  ASSERT_TRUE(appended_options1);
  auto appended_options2 = DummyOpaqueOptions::CreateOptions();
  ASSERT_TRUE(appended_options2);

  EXPECT_EQ(LiteRtAppendOpaqueOptions(&options_, *appended_options1),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtAppendOpaqueOptions(&options_, *appended_options2),
            kLiteRtStatusOk);

  // Iterate through the list to check that the links have been correctly added.

  LiteRtOpaqueOptions options_it = options_;
  ASSERT_EQ(LiteRtGetNextOpaqueOptions(&options_it), kLiteRtStatusOk);
  EXPECT_EQ(options_it, *appended_options1);

  ASSERT_EQ(LiteRtGetNextOpaqueOptions(&options_it), kLiteRtStatusOk);
  EXPECT_EQ(options_it, *appended_options2);

  // The list is destroyed in the `TearDown()` function.
}

TEST_F(LiteRtOpaqueOptionsTest, FindData) {
  constexpr auto* appended_options_id = "appended_options_id";
  void* appended_options_data = reinterpret_cast<void*>(12345);
  constexpr auto appended_options_destructor = [](void*) {};

  auto appended_options = DummyOpaqueOptions::CreateOptions(
      appended_options_id, appended_options_data, appended_options_destructor);

  EXPECT_EQ(LiteRtAppendOpaqueOptions(&options_, *appended_options),
            kLiteRtStatusOk);

  void* payload_data;
  EXPECT_EQ(
      LiteRtFindOpaqueOptionsData(options_, appended_options_id, &payload_data),
      kLiteRtStatusOk);

  EXPECT_EQ(payload_data, appended_options_data);

  // The list is destroyed in the `TearDown()` function.
}

TEST_F(LiteRtOpaqueOptionsTest, Pop) {
  constexpr auto* appended_options_id = "appended_options_id";
  void* appended_options_data = reinterpret_cast<void*>(12345);
  constexpr auto appended_options_destructor = [](void*) {};

  auto appended_options = DummyOpaqueOptions::CreateOptions(
      appended_options_id, appended_options_data, appended_options_destructor);

  EXPECT_EQ(LiteRtAppendOpaqueOptions(&options_, *appended_options),
            kLiteRtStatusOk);

  void* payload_data;
  EXPECT_EQ(
      LiteRtFindOpaqueOptionsData(options_, appended_options_id, &payload_data),
      kLiteRtStatusOk);

  // After poping the last item, we shouldn't be able to find it any longer.
  EXPECT_EQ(LiteRtPopOpaqueOptions(&options_), kLiteRtStatusOk);
  EXPECT_NE(
      LiteRtFindOpaqueOptionsData(options_, appended_options_id, &payload_data),
      kLiteRtStatusOk);

  // The list is destroyed in the `TearDown()` function.
}

}  // namespace
