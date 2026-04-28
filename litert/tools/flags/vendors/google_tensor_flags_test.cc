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

#include "litert/tools/flags/vendors/google_tensor_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_google_tensor_options.h"

namespace litert::google_tensor {
namespace {

TEST(TruncationTypeFlagTest, Malformed) {
  std::string error;
  LrtGoogleTensorOptionsTruncationType value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
}

TEST(TruncationTypeFlagTest, Parse) {
  std::string error;
  LrtGoogleTensorOptionsTruncationType value;

  {
    static constexpr absl::string_view kLevel = "auto";
    static constexpr LrtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeAuto;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "no_truncation";
    static constexpr LrtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeNoTruncation;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kLevel = "bfloat16";
    static constexpr LrtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeBfloat16;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
  {
    static constexpr absl::string_view kLevel = "half";
    static constexpr LrtGoogleTensorOptionsTruncationType kLevelEnum =
        kLiteRtGoogleTensorFloatTruncationTypeHalf;
    EXPECT_TRUE(AbslParseFlag(kLevel, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kLevel, AbslUnparseFlag(value));
  }
}

TEST(PerformanceModeFlagTest, Malformed) {
  std::string error;
  GoogleTensorOptions::PerformanceMode value;

  EXPECT_FALSE(AbslParseFlag("oogabooga", &value, &error));
  EXPECT_FALSE(AbslParseFlag("highperf", &value, &error));
}

TEST(PerformanceModeFlagTest, Parse) {
  std::string error;
  GoogleTensorOptions::PerformanceMode value;

  {
    static constexpr absl::string_view kMode = "extreme_power_saver";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kExtremePowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "power_saver";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kPowerSaver;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "balanced";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kBalanced;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "high_performance";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kHighPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "sustained_performance";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kSustainedPerformance;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }

  {
    static constexpr absl::string_view kMode = "burst";
    static constexpr GoogleTensorOptions::PerformanceMode kModeEnum =
        GoogleTensorOptions::PerformanceMode::kBurst;
    EXPECT_TRUE(AbslParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, AbslUnparseFlag(value));
  }
}

TEST(UpdateGoogleTensorOptionsFromFlagsTest, SetPerformanceMode) {
  Expected<GoogleTensorOptions> options = GoogleTensorOptions::Create();
  ASSERT_TRUE(options.HasValue());
  // Default value should be balanced.
  ASSERT_TRUE(UpdateGoogleTensorOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetPerformanceMode(),
            GoogleTensorOptions::PerformanceMode::kBalanced);
  // Update flag value to burst.
  absl::SetFlag(&FLAGS_google_tensor_performance_mode,
                GoogleTensorOptions::PerformanceMode::kBurst);
  ASSERT_TRUE(UpdateGoogleTensorOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetPerformanceMode(),
            GoogleTensorOptions::PerformanceMode::kBurst);

  // Reset flag to default to avoid affecting other tests.
  absl::SetFlag(&FLAGS_google_tensor_performance_mode,
                GoogleTensorOptions::PerformanceMode::kBalanced);
}

}  // namespace

}  // namespace litert::google_tensor
