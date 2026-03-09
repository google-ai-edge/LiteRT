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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert::mediatek {
namespace {

void SerializeAndParse(LrtMediatekOptions* payload,
                       LrtMediatekOptions** parsed) {
  const char* identifier;
  void* raw_payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueMediatekOptionsData(
      payload, &identifier, &raw_payload, &payload_deleter));
  EXPECT_STREQ(identifier, "mediatek");
  std::string* toml_str = reinterpret_cast<std::string*>(raw_payload);

  LITERT_ASSERT_OK(LrtCreateMediatekOptionsFromToml(toml_str->c_str(), parsed));

  payload_deleter(raw_payload);
}

TEST(LrtMediatekOptionsTest, GetOpaqueDataEmpty) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueMediatekOptionsData(options, &identifier,
                                                   &payload, &payload_deleter));

  EXPECT_STREQ(identifier, "mediatek");
  absl::string_view toml_str = *reinterpret_cast<std::string*>(payload);
  EXPECT_TRUE(toml_str.empty());

  payload_deleter(payload);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, NeronSDKVersionType) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type;
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsNeronSDKVersionType(options, &sdk_version_type));
  EXPECT_EQ(sdk_version_type,
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8);

  LITERT_ASSERT_OK(LrtSetMediatekOptionsNeronSDKVersionType(
      options, kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7));
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsNeronSDKVersionType(options, &sdk_version_type));
  EXPECT_EQ(sdk_version_type,
            kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  LiteRtMediatekOptionsNeronSDKVersionType parsed_ver;
  LrtGetMediatekOptionsNeronSDKVersionType(parsed, &parsed_ver);
  EXPECT_EQ(parsed_ver, kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, GemmaCompilerOptimizations) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  bool gemma_optimizations;
  LITERT_ASSERT_OK(LrtGetMediatekOptionsGemmaCompilerOptimizations(
      options, &gemma_optimizations));
  EXPECT_FALSE(gemma_optimizations);

  LITERT_ASSERT_OK(
      LrtSetMediatekOptionsGemmaCompilerOptimizations(options, true));
  LITERT_ASSERT_OK(LrtGetMediatekOptionsGemmaCompilerOptimizations(
      options, &gemma_optimizations));
  EXPECT_TRUE(gemma_optimizations);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_gemma;
  LrtGetMediatekOptionsGemmaCompilerOptimizations(parsed, &parsed_gemma);
  EXPECT_TRUE(parsed_gemma);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, PerformanceMode) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  LiteRtMediatekNeuronAdapterPerformanceMode performance_mode;
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsPerformanceMode(options, &performance_mode));
  EXPECT_EQ(
      performance_mode,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferSustainedSpeed);

  LITERT_ASSERT_OK(LrtSetMediatekOptionsPerformanceMode(
      options,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower));
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsPerformanceMode(options, &performance_mode));
  EXPECT_EQ(performance_mode,
            kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  LiteRtMediatekNeuronAdapterPerformanceMode parsed_perf;
  LrtGetMediatekOptionsPerformanceMode(parsed, &parsed_perf);
  EXPECT_EQ(parsed_perf,
            kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferLowPower);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, L1CacheOptimizations) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  bool l1_cache_optimizations;
  LITERT_ASSERT_OK(LrtGetMediatekOptionsL1CacheOptimizations(
      options, &l1_cache_optimizations));
  EXPECT_FALSE(l1_cache_optimizations);

  LITERT_ASSERT_OK(LrtSetMediatekOptionsL1CacheOptimizations(options, true));
  LITERT_ASSERT_OK(LrtGetMediatekOptionsL1CacheOptimizations(
      options, &l1_cache_optimizations));
  EXPECT_TRUE(l1_cache_optimizations);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_l1;
  LrtGetMediatekOptionsL1CacheOptimizations(parsed, &parsed_l1);
  EXPECT_TRUE(parsed_l1);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, OptimizationHint) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  LiteRtMediatekNeuronAdapterOptimizationHint optimization_hint;
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsOptimizationHint(options, &optimization_hint));
  EXPECT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintNormal);

  LITERT_ASSERT_OK(LrtSetMediatekOptionsOptimizationHint(
      options, kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency));
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsOptimizationHint(options, &optimization_hint));
  EXPECT_EQ(optimization_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  LiteRtMediatekNeuronAdapterOptimizationHint parsed_hint;
  LrtGetMediatekOptionsOptimizationHint(parsed, &parsed_hint);
  EXPECT_EQ(parsed_hint,
            kLiteRtMediatekNeuronAdapterOptimizationHintLowLatency);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, DisableDlaDirRemoval) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  bool disable_dla_dir_removal;
  LITERT_ASSERT_OK(LrtGetMediatekOptionsDisableDlaDirRemoval(
      options, &disable_dla_dir_removal));
  EXPECT_FALSE(disable_dla_dir_removal);

  LITERT_ASSERT_OK(LrtSetMediatekOptionsDisableDlaDirRemoval(options, true));
  LITERT_ASSERT_OK(LrtGetMediatekOptionsDisableDlaDirRemoval(
      options, &disable_dla_dir_removal));
  EXPECT_TRUE(disable_dla_dir_removal);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_disable;
  LrtGetMediatekOptionsDisableDlaDirRemoval(parsed, &parsed_disable);
  EXPECT_TRUE(parsed_disable);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, MediatekDlaDir) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  const char* mediatek_dla_dir;
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsMediatekDlaDir(options, &mediatek_dla_dir));
  EXPECT_STREQ(mediatek_dla_dir, "");

  const char* test_dir = "/test/dir";
  LITERT_ASSERT_OK(LrtSetMediatekOptionsMediatekDlaDir(options, test_dir));
  LITERT_ASSERT_OK(
      LrtGetMediatekOptionsMediatekDlaDir(options, &mediatek_dla_dir));
  EXPECT_STREQ(mediatek_dla_dir, test_dir);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  const char* parsed_dla_dir;
  LrtGetMediatekOptionsMediatekDlaDir(parsed, &parsed_dla_dir);
  EXPECT_STREQ(parsed_dla_dir, test_dir);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, AotCompilationOptions) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  const char* aot_compilation_options;
  LITERT_ASSERT_OK(LrtGetMediatekOptionsAotCompilationOptions(
      options, &aot_compilation_options));
  EXPECT_STREQ(aot_compilation_options, "");

  const char* test_options = "--test_flag=value --another_flag";
  LITERT_ASSERT_OK(
      LrtSetMediatekOptionsAotCompilationOptions(options, test_options));
  LITERT_ASSERT_OK(LrtGetMediatekOptionsAotCompilationOptions(
      options, &aot_compilation_options));
  EXPECT_STREQ(aot_compilation_options, test_options);

  LrtMediatekOptions* parsed;
  SerializeAndParse(options, &parsed);
  const char* parsed_aot;
  LrtGetMediatekOptionsAotCompilationOptions(parsed, &parsed_aot);
  EXPECT_STREQ(parsed_aot, test_options);

  LrtDestroyMediatekOptions(parsed);
  LrtDestroyMediatekOptions(options);
}

TEST(LrtMediatekOptionsTest, GetOpaqueDataPopulated) {
  LrtMediatekOptions* options;
  LITERT_ASSERT_OK(LrtCreateMediatekOptions(&options));

  LITERT_ASSERT_OK(LrtSetMediatekOptionsPerformanceMode(
      options,
      kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferFastSingleAnswer));
  LITERT_ASSERT_OK(LrtSetMediatekOptionsAotCompilationOptions(
      options, "--test_compilation"));

  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueMediatekOptionsData(options, &identifier,
                                                   &payload, &payload_deleter));

  EXPECT_STREQ(identifier, "mediatek");
  absl::string_view toml_str = *reinterpret_cast<std::string*>(payload);
  EXPECT_TRUE(absl::StrContains(toml_str, "performance_mode = 1"));
  EXPECT_TRUE(absl::StrContains(
      toml_str, "aot_compilation_options = \"--test_compilation\""));

  payload_deleter(payload);
  LrtDestroyMediatekOptions(options);
}

}  // namespace
}  // namespace litert::mediatek
