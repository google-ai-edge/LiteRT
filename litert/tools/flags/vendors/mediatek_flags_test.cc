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

#include "litert/tools/flags/vendors/mediatek_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/marshalling.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_mediatek_options.h"

namespace litert::mediatek {
namespace {

TEST(NeronSDKVersionTypeFlagTest, Malformed) {
  std::string error;
  LiteRtMediatekOptionsNeronSDKVersionType value;

  EXPECT_FALSE(absl::ParseFlag("wenxin", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("+", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("steven", &value, &error));
}

TEST(NeronSDKVersionTypeFlagTest, Parse) {
  std::string error;
  LiteRtMediatekOptionsNeronSDKVersionType value;

  {
    static constexpr absl::string_view kVersion = "version8";
    static constexpr LiteRtMediatekOptionsNeronSDKVersionType kVersionEnum =
        kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8;
    EXPECT_TRUE(absl::ParseFlag(kVersion, &value, &error));
    EXPECT_EQ(value, kVersionEnum);
    EXPECT_EQ(kVersion, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kVersion = "version7";
    static constexpr LiteRtMediatekOptionsNeronSDKVersionType kLevelEnum =
        kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7;
    EXPECT_TRUE(absl::ParseFlag(kVersion, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kVersion, absl::UnparseFlag(value));
  }
}

TEST(MediatekOptionsFromFlagsTest, DefaultValue) {
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetNeronSDKVersionType(),
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8);
  EXPECT_FALSE(options.Value().GetEnableGemmaCompilerOptimizations());
}

TEST(MediatekOptionsFromFlagsTest, SetFlagToVersion7) {
  absl::SetFlag(&FLAGS_mediatek_sdk_version_type,
                kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();

  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetNeronSDKVersionType(),
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);
}

TEST(MediatekOptionsFromFlagsTest, SetEnableGemmaCompilerOptimizationsToTrue) {
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, true);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_TRUE(options.Value().GetEnableGemmaCompilerOptimizations());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, false);
}

TEST(MediatekOptionsFromFlagsTest, SetEnableGemmaCompilerOptimizationsToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, false);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_FALSE(options.Value().GetEnableGemmaCompilerOptimizations());
}

}  // namespace

}  // namespace litert::mediatek
