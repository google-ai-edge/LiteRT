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
  MediatekOptions::NeronSDKVersion value;

  EXPECT_FALSE(absl::ParseFlag("wenxin", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("+", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("steven", &value, &error));
}

TEST(NeronSDKVersionTypeFlagTest, Parse) {
  std::string error;
  MediatekOptions::NeronSDKVersion value;

  {
    static constexpr absl::string_view kVersion = "version8";
    static constexpr MediatekOptions::NeronSDKVersion kVersionEnum =
        MediatekOptions::NeronSDKVersion::kVersion8;
    EXPECT_TRUE(absl::ParseFlag(kVersion, &value, &error));
    EXPECT_EQ(value, kVersionEnum);
    EXPECT_EQ(kVersion, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kVersion = "version7";
    static constexpr MediatekOptions::NeronSDKVersion kLevelEnum =
        MediatekOptions::NeronSDKVersion::kVersion7;
    EXPECT_TRUE(absl::ParseFlag(kVersion, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kVersion, absl::UnparseFlag(value));
  }

  {
    static constexpr absl::string_view kVersion = "version9";
    static constexpr MediatekOptions::NeronSDKVersion kLevelEnum =
        MediatekOptions::NeronSDKVersion::kVersion9;
    EXPECT_TRUE(absl::ParseFlag(kVersion, &value, &error));
    EXPECT_EQ(value, kLevelEnum);
    EXPECT_EQ(kVersion, absl::UnparseFlag(value));
  }
}

TEST(UpdateMediatekOptionsFromFlagsTest, DefaultValue) {
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetNeronSDKVersionType(),
            MediatekOptions::NeronSDKVersion::kVersion8);
  EXPECT_FALSE(options.Value().GetEnableGemmaCompilerOptimizations());
  EXPECT_EQ(
      options.Value().GetPerformanceMode(),
      MediatekOptions::PerformanceMode::kSustainedSpeed);
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "");
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetFlagToVersion7) {
  absl::SetFlag(&FLAGS_mediatek_sdk_version_type,
                MediatekOptions::NeronSDKVersion::kVersion7);
  Expected<MediatekOptions> options = MediatekOptions::Create();

  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetNeronSDKVersionType(),
            MediatekOptions::NeronSDKVersion::kVersion7);
}

TEST(UpdateMediatekOptionsFromFlagsTest,
     SetEnableGemmaCompilerOptimizationsToTrue) {
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, true);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetEnableGemmaCompilerOptimizations());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, false);
}

TEST(UpdateMediatekOptionsFromFlagsTest,
     SetEnableGemmaCompilerOptimizationsToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_enable_gemma_compiler_optimizations, false);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_FALSE(options.Value().GetEnableGemmaCompilerOptimizations());
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetPerformanceMode) {
  absl::SetFlag(
      &FLAGS_mediatek_performance_mode_type,
      MediatekOptions::PerformanceMode::kLowPower);
  Expected<MediatekOptions> options = MediatekOptions::Create();

  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetPerformanceMode(),
            MediatekOptions::PerformanceMode::kLowPower);
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(
      &FLAGS_mediatek_performance_mode_type,
      MediatekOptions::PerformanceMode::kSustainedSpeed);
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetEnableL1CacheOptimizationsToTrue) {
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, true);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetEnableL1CacheOptimizations());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, false);
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetEnableL1CacheOptimizationsToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_enable_l1_cache_optimizations, false);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_FALSE(options.Value().GetEnableL1CacheOptimizations());
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetDisableDlaDirRemovalToTrue) {
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, true);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetDisableDlaDirRemoval());
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, false);
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetDisableDlaDirRemovalToFalse) {
  // Explicitly set to false (even though it's the default) to ensure it's
  // picked up
  absl::SetFlag(&FLAGS_mediatek_disable_dla_dir_removal, false);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_FALSE(options.Value().GetDisableDlaDirRemoval());
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetUseGetSupportedOperationsToFalse) {
  absl::SetFlag(&FLAGS_mediatek_use_get_supported_operations, false);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_FALSE(options.Value().GetUseGetSupportedOperations());
  // Reset to default.
  absl::SetFlag(&FLAGS_mediatek_use_get_supported_operations, true);
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetUseGetSupportedOperationsToTrue) {
  absl::SetFlag(&FLAGS_mediatek_use_get_supported_operations, true);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetUseGetSupportedOperations());
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetOptimizationHint) {
  absl::SetFlag(&FLAGS_mediatek_optimization_hint,
                MediatekOptions::OptimizationHint::kLowLatency);
  Expected<MediatekOptions> options = MediatekOptions::Create();

  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetOptimizationHint(),
            MediatekOptions::OptimizationHint::kLowLatency);
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_optimization_hint,
                MediatekOptions::OptimizationHint::kNormal);
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetMediatekDlaDir) {
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "/data/local/tmp");
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "/data/local/tmp");
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "");
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetMediatekDlaDirMalformed) {
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "this is not a path");
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetMediatekDlaDir(), "this is not a path");
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_dla_dir, "");
}

TEST(NeuronAdapterPerformanceModeFlagTest, Malformed) {
  std::string error;
  MediatekOptions::PerformanceMode value;

  EXPECT_FALSE(absl::ParseFlag("not", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("a real", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("flag", &value, &error));
}

TEST(NeuronAdapterPerformanceModeFlagTest, Parse) {
  std::string error;
  MediatekOptions::PerformanceMode value;
  {
    static constexpr absl::string_view kMode = "low_power";
    static constexpr MediatekOptions::PerformanceMode kModeEnum =
        MediatekOptions::PerformanceMode::kLowPower;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "fast_single_answer";
    static constexpr MediatekOptions::PerformanceMode kModeEnum =
        MediatekOptions::PerformanceMode::kFastSingleAnswer;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "sustained_speed";
    static constexpr MediatekOptions::PerformanceMode kModeEnum =
        MediatekOptions::PerformanceMode::kSustainedSpeed;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "turbo_boost";
    static constexpr MediatekOptions::PerformanceMode kModeEnum =
        MediatekOptions::PerformanceMode::kTurboBoost;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
}

TEST(NeuronAdapterOptimizationHintFlagTest, Malformed) {
  std::string error;
  MediatekOptions::OptimizationHint value;

  EXPECT_FALSE(absl::ParseFlag("not", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("a real", &value, &error));
  EXPECT_FALSE(absl::ParseFlag("flag", &value, &error));
}

TEST(NeuronAdapterOptimizationHintFlagTest, Parse) {
  std::string error;
  MediatekOptions::OptimizationHint value;
  {
    static constexpr absl::string_view kMode = "normal";
    static constexpr MediatekOptions::OptimizationHint kModeEnum =
        MediatekOptions::OptimizationHint::kNormal;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "low_latency";
    static constexpr MediatekOptions::OptimizationHint kModeEnum =
        MediatekOptions::OptimizationHint::kLowLatency;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "deep_fusion";
    static constexpr MediatekOptions::OptimizationHint kModeEnum =
        MediatekOptions::OptimizationHint::kDeepFusion;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
  {
    static constexpr absl::string_view kMode = "batch_processing";
    static constexpr MediatekOptions::OptimizationHint kModeEnum =
        MediatekOptions::OptimizationHint::kBatchProcessing;
    EXPECT_TRUE(absl::ParseFlag(kMode, &value, &error));
    EXPECT_EQ(value, kModeEnum);
    EXPECT_EQ(kMode, absl::UnparseFlag(value));
  }
}

TEST(UpdateMediatekOptionsFromFlagsTest, SetAotCompilationOptions) {
  const std::string test_options =
      "a test flag that does absolutely nothing.";
  absl::SetFlag(&FLAGS_mediatek_aot_compilation_options, test_options);
  Expected<MediatekOptions> options = MediatekOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateMediatekOptionsFromFlags(options.Value()).HasValue());
  EXPECT_EQ(options.Value().GetAotCompilationOptions(), test_options);
  // Reset flag to default to avoid affecting other tests
  absl::SetFlag(&FLAGS_mediatek_aot_compilation_options, "");
}

}  // namespace

}  // namespace litert::mediatek
