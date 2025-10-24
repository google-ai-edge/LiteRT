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
#include "litert/c/options/litert_mediatek_options.h"
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

  {
    static constexpr absl::string_view kVersion = "version9";
    static constexpr LiteRtMediatekOptionsNeronSDKVersionType kLevelEnum =
        kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9;
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
  EXPECT_EQ(
      options.Value().GetPerformanceMode(),
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "");
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

TEST(MediatekOptionsFromFlagsTest, SetPerformanceMode) {
  absl::SetFlag(
      &FLAGS_mediatek_performance_mode_type,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();

  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetPerformanceMode(),
            kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(
      &FLAGS_mediatek_performance_mode_type,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);
}

TEST(MediatekOptionsFromFlagsTest, SetEnableL1CacheOptimizationsToTrue) {
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, true);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_TRUE(options.Value().GetEnableL1CacheOptimizations());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, false);
}

TEST(MediatekOptionsFromFlagsTest, SetEnableL1CacheOptimizationsToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, false);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_FALSE(options.Value().GetEnableL1CacheOptimizations());
}

TEST(MediatekOptionsFromFlagsTest, SetDisableDlaDirRemovalToTrue) {
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, true);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_TRUE(options.Value().GetDisableDlaDirRemoval());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, false);
}

TEST(MediatekOptionsFromFlagsTest, SetDisableDlaDirRemovalToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, false);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_FALSE(options.Value().GetDisableDlaDirRemoval());
}

TEST(MediatekOptionsFromFlagsTest, SetOptimizationHint) {
  absl::SetFlag(&FLAGS_mediatek_optimization_hint,
                kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();

  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetOptimizationHint(),
            kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_optimization_hint,
                kLiteRtMediatekNeuronAdapterOptimizationHintNormal);
}

TEST(MediatekOptionsFromFlagsTest, SetMediatekDlaDir) {
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "/data/local/tmp");
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "/data/local/tmp");
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "");
}

TEST(MediatekOptionsFromFlagsTest, SetMediatekDlaDirMalformed) {
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "this is not a path");
  Expected<MediatekOptions> options = MediatekOptionsFromFlags();
  ASSERT_TRUE(options.HasValue());
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "this is not a path");
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "");
}

TEST(NeuronAdapterPerformanceModeFlagTest, Malformed) {
  std::string error;
  LiteRtMediatekNeuronAdapterPerformanceMode value;

  EXPECT_FALSE(absl::ParseFlag("not", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("a real", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("flag", &value, &error));
}

TEST(NeuronAdapterPerformanceModeFlagTest, Parse) {
  std::string error;
  LiteRtMediatekNeuronAdapterPerformanceMode value;
  {
    static constexpr absl::string_view kMode = "low_power";
    static constexpr LiteRtMediatekNeuronAdapterPerformanceMode kModeEnum =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "fast_single_answer";
    static constexpr LiteRtMediatekNeuronAdapterPerformanceMode kModeEnum =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "sustained_speed";
    static constexpr LiteRtMediatekNeuronAdapterPerformanceMode kModeEnum =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "turbo_boost";
    static constexpr LiteRtMediatekNeuronAdapterPerformanceMode kModeEnum =
        kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferTurboBoost;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
}

TEST(NeuronAdapterOptimizationHintFlagTest, Malformed) {
  std::string error;
  LiteRtMediatekNeuronAdapterOptimizationHint value;

  EXPECT_FALSE(absl::ParseFlag("not", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("a real", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("flag", &value, &error));
}

TEST(NeuronAdapterOptimizationHintFlagTest, Parse) {
  std::string error;
  LiteRtMediatekNeuronAdapterOptimizationHint value;
  {
    static constexpr absl::string_view kMode = "normal";
    static constexpr LiteRtMediatekNeuronAdapterOptimizationHint kModeEnum =
        kLiteRtMediatekNeuronAdapterOptimizationHintNormal;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "low_latency";
    static constexpr LiteRtMediatekNeuronAdapterOptimizationHint kModeEnum =
        kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "deep_fusion";
    static constexpr LiteRtMediatekNeuronAdapterOptimizationHint kModeEnum =
        kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "batch_processing";
    static constexpr LiteRtMediatekNeuronAdapterOptimizationHint kModeEnum =
        kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
}

}  // namespace

}  // namespace litert::mediatek
