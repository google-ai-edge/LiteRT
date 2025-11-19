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

#include <cstdint>

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

TEST(LiteRtMediatekOptionsTest, GemmaCompilerOptimizations) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  bool gemma_optimizations;
  // Check default value (false)
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
      options_data, &gemma_optimizations));
  ASSERT_FALSE(gemma_optimizations);

  // Set to true
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetGemmaCompilerOptimizations(options_data, true));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
      options_data, &gemma_optimizations));
  ASSERT_TRUE(gemma_optimizations);

  // Set to false
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetGemmaCompilerOptimizations(options_data, false));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
      options_data, &gemma_optimizations));
  ASSERT_FALSE(gemma_optimizations);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, GemmaCompilerOptimizationsInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  bool gemma_optimizations;

  EXPECT_EQ(LiteRtMediatekOptionsSetGemmaCompilerOptimizations(nullptr, true),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(
      LiteRtMediatekOptionsGetGemmaCompilerOptimizations(options_data, nullptr),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGetGemmaCompilerOptimizations(
                nullptr, &gemma_optimizations),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, PerformanceMode) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;
  // Check default value
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsGetPerformanceMode(options_data, &performance_mode));

  ASSERT_EQ(
      performance_mode,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);

  // Set to LowPower
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetPerformanceMode(
      options_data,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower));
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsGetPerformanceMode(options_data, &performance_mode));
  ASSERT_EQ(performance_mode,
            kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);

  // Set to FastSingleAnswer
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetPerformanceMode(
      options_data,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer));
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsGetPerformanceMode(options_data, &performance_mode));
  ASSERT_EQ(
      performance_mode,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, PerformanceModeInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;

  EXPECT_EQ(
      LiteRtMediatekOptionsSetPerformanceMode(
          nullptr,
          kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower),
      kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtMediatekOptionsGetPerformanceMode(options_data, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGetPerformanceMode(nullptr, &performance_mode),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, L1CacheOptimizations) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  bool l1_cache_optimizations;
  // Check default value (false)
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetL1CacheOptimizations(
      options_data, &l1_cache_optimizations));
  ASSERT_FALSE(l1_cache_optimizations);

  // Set to true
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetL1CacheOptimizations(options_data, true));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetL1CacheOptimizations(
      options_data, &l1_cache_optimizations));
  ASSERT_TRUE(l1_cache_optimizations);

  // Set to false
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetL1CacheOptimizations(options_data, false));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetL1CacheOptimizations(
      options_data, &l1_cache_optimizations));
  ASSERT_FALSE(l1_cache_optimizations);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, L1CacheOptimizationsInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  bool l1_cache_optimizations;

  EXPECT_EQ(LiteRtMediatekOptionsSetL1CacheOptimizations(nullptr, true),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtMediatekOptionsGetL1CacheOptimizations(options_data, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGetL1CacheOptimizations(
                nullptr, &l1_cache_optimizations),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, OptimizationHint) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;
  // Check default value
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetOptimizationHint(
      options_data, &optimization_hint));

  ASSERT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintNormal);

  // Set to LowLatency
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetOptimizationHint(
      options_data, kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetOptimizationHint(
      options_data, &optimization_hint));
  ASSERT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);

  // Set to DeepFusion
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetOptimizationHint(
      options_data, kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetOptimizationHint(
      options_data, &optimization_hint));
  ASSERT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion);

  // Set to BatchProcessing
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetOptimizationHint(
      options_data,
      kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetOptimizationHint(
      options_data, &optimization_hint));
  ASSERT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, OptimizationHintInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;

  EXPECT_EQ(
      LiteRtMediatekOptionsSetOptimizationHint(
          nullptr, kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency),
      kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtMediatekOptionsGetOptimizationHint(options_data, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtMediatekOptionsGetOptimizationHint(nullptr, &optimization_hint),
      kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, DisableDlaDirRemoval) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  bool disable_dla_dir_removal;
  // Check default value (false)
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetDisableDlaDirRemoval(
      options_data, &disable_dla_dir_removal));
  ASSERT_FALSE(disable_dla_dir_removal);

  // Set to true
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetDisableDlaDirRemoval(options_data, true));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetDisableDlaDirRemoval(
      options_data, &disable_dla_dir_removal));
  ASSERT_TRUE(disable_dla_dir_removal);

  // Set to false
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetDisableDlaDirRemoval(options_data, false));
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGetDisableDlaDirRemoval(
      options_data, &disable_dla_dir_removal));
  ASSERT_FALSE(disable_dla_dir_removal);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, DisableDlaDirRemovalInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  bool disable_dla_dir_removal;

  EXPECT_EQ(LiteRtMediatekOptionsSetDisableDlaDirRemoval(nullptr, true),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtMediatekOptionsGetDisableDlaDirRemoval(options_data, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGetDisableDlaDirRemoval(
                nullptr, &disable_dla_dir_removal),
            kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, MediatekDlaDir) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  const char* mediatek_dla_dir;
  // Check default value (empty string)
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsGetMediatekDlaDir(options_data, &mediatek_dla_dir));
  ASSERT_STREQ(mediatek_dla_dir, "");

  // Set to a value
  const char* test_dir = "/test/dir";
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsSetMediatekDlaDir(options_data, test_dir));
  LITERT_ASSERT_OK(
      LiteRtMediatekOptionsGetMediatekDlaDir(options_data, &mediatek_dla_dir));
  ASSERT_STREQ(mediatek_dla_dir, test_dir);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtMediatekOptionsTest, MediatekDlaDirInvalidArguments) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));
  const char* mediatek_dla_dir;

  EXPECT_EQ(LiteRtMediatekOptionsSetMediatekDlaDir(nullptr, "/test/dir"),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtMediatekOptionsGetMediatekDlaDir(options_data, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtMediatekOptionsGetMediatekDlaDir(nullptr, &mediatek_dla_dir),
            kLiteRtStatusErrorInvalidArgument);

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
  options->SetNeronSDKVersionType(
      kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9);
  EXPECT_EQ(options->GetNeronSDKVersionType(),
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion9);

  // Test Gemma Compiler Optimizations
  EXPECT_FALSE(options->GetEnableGemmaCompilerOptimizations());
  options->SetEnableGemmaCompilerOptimizations(true);
  EXPECT_TRUE(options->GetEnableGemmaCompilerOptimizations());
  options->SetEnableGemmaCompilerOptimizations(false);
  EXPECT_FALSE(options->GetEnableGemmaCompilerOptimizations());

  // Test Performance Mode
  EXPECT_EQ(
      options->GetPerformanceMode(),
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);
  options->SetPerformanceMode(
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);
  EXPECT_EQ(options->GetPerformanceMode(),
            kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);
  options->SetPerformanceMode(
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer);
  EXPECT_EQ(
      options->GetPerformanceMode(),
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer);

  // Test L1 Cache Optimizations
  EXPECT_FALSE(options->GetEnableL1CacheOptimizations());
  options->SetEnableL1CacheOptimizations(true);
  EXPECT_TRUE(options->GetEnableL1CacheOptimizations());
  options->SetEnableL1CacheOptimizations(false);
  EXPECT_FALSE(options->GetEnableL1CacheOptimizations());

  // Test Optimization Hint
  EXPECT_EQ(options->GetOptimizationHint(),
            kLiteRtMediatekNeuronAdapterOptimizationHintNormal);
  options->SetOptimizationHint(
      kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);
  EXPECT_EQ(options->GetOptimizationHint(),
            kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);
  options->SetOptimizationHint(
      kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion);
  EXPECT_EQ(options->GetOptimizationHint(),
            kLiteRtMediatekNeuronAdapterOptimizationHintDeepFusion);
  options->SetOptimizationHint(
      kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing);
  EXPECT_EQ(options->GetOptimizationHint(),
            kLiteRtMediatekNeuronAdapterOptimizationHintBatchProcessing);

  // Test Disable DLA Dir Removal
  EXPECT_FALSE(options->GetDisableDlaDirRemoval());
  options->SetDisableDlaDirRemoval(true);
  EXPECT_TRUE(options->GetDisableDlaDirRemoval());
  options->SetDisableDlaDirRemoval(false);
  EXPECT_FALSE(options->GetDisableDlaDirRemoval());

  // // Test Mediatek DLA Dir
  EXPECT_EQ(options->GetMediatekDlaDir(), "");
  options->SetMediatekDlaDir("/test/dir");
  EXPECT_EQ(options->GetMediatekDlaDir(), "/test/dir");
}

TEST(LiteRtMediatekOptionsTest, OptionsHash) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsCreate(&options));
  LiteRtMediatekOptions options_data;
  LITERT_ASSERT_OK(LiteRtMediatekOptionsGet(options, &options_data));

  // Check that we can compute the hash of the options.
  uint64_t initial_hash;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options, &initial_hash));

  // Modify an option and check if the hash changes.
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
  uint64_t new_hash;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options, &new_hash));
  EXPECT_NE(initial_hash, new_hash);

  // Reset the option and check if the hash is the same as the original hash.
  LITERT_ASSERT_OK(LiteRtMediatekOptionsSetNeronSDKVersionType(
      options_data, kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8));
  uint64_t reset_hash;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options, &reset_hash));
  EXPECT_EQ(initial_hash, reset_hash);

  LiteRtDestroyOpaqueOptions(options);
}

}  // namespace
}  // namespace litert::mediatek
