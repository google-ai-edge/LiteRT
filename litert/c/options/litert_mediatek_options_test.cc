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
#include "litert/c/options/litert_mediatek_options.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/test/matchers.h"

namespace litert::mediatek {
namespace {
TEST(LiteRtMediatekOptionsTest, CreateAndGet) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, NeronSDKVersionType) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetNeronSDKVersionType(
      options_data, &sdk_version_type));
  ASSERT_EQ(sdk_version_type,
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8);

  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetNeronSDKVersionType(
      options_data, kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetNeronSDKVersionType(
      options_data, &sdk_version_type));
  ASSERT_EQ(sdk_version_type,
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, GetWithInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data = nullptr;
  EXPECT_EQ(LiteRtMediatekOptionsGet(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGet(nullptr, &options_data),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, GetWithInvalidIdentifier) {
  LiteRtOpaqueOptions options;
  int payload_int = 17;
  void* payload = &payload_int;
  LITERT_ASSERT_OK(LiteRtCreateOpaqueOptions(
      "invalid_identifier", payload, [](void*) {}, &options));
  LiteRtMediatekOptions options_data;
  EXPECT_EQ(LiteRtMediatekOptionsGet(options, &options_data),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDestroyOpaqueOptions(options);
}

TEST(MediatekOptionsTest, CppApi) {
  auto options = MediatekOptions::Create();
  ASSERT_TRUE(options);
  EXPECT_EQ(options->GetNeronSDKVersionType(),
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8);
  options->SetNeronSDKVersionType(
      kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);
  EXPECT_EQ(options->GetNeronSDKVersionType(),
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);
}

}  // namespace
}  // namespace litert::mediatek
