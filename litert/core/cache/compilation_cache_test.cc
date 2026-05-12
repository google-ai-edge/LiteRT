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

#include "litert/core/cache/compilation_cache.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/options.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace litert::internal {

CompilationCache::CompilerPluginInfo GetTestCompilerPluginInfo() {
  return {
      .api_version = {.major = 1, .minor = 0, .patch = 0},
      .hw_accelerators = kLiteRtHwAcceleratorNpu,
      .manufacturer = "test_manufacturer",
      .sdk_version = "1.0.0",
  };
}

LiteRtOptionsT GetTestOptions() {
  return {
      .version = {.major = 1, .minor = 0, .patch = 0},
      .hardware_accelerators = kLiteRtHwAcceleratorNpu,
  };
}

TEST(CompilationCacheTest, CacheMiss) {
  // GIVEN: a compilation cache and a model
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  // WHEN: the model has not been saved to the cache
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));

  // THEN: the model is not found in the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_miss,
                         compilation_cache.TryLoadModel(cache_key));
  EXPECT_FALSE(cache_miss.has_value());
}

TEST(CompilationCacheTest, CacheHit) {
  // GIVEN: a compilation cache and a model
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  // WHEN: the model is saved to the cache
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key));

  // THEN: the model can be found in the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(cache_key));
  EXPECT_TRUE(cache_hit.has_value());
}

TEST(CompilationCacheTest, CacheHitWithModelName) {
  // GIVEN: a compilation cache and a model
  const std::string cache_root_path =
      ::testing::TempDir() + "/CacheHitWithModelName";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  // WHEN: the model is saved to the cache with a model name
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  std::string model_name = "test_model";
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model, cache_key, model_name));

  // THEN: the model can be found in the cache using the model name
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit_with_name,
                         compilation_cache.TryLoadModel(cache_key, model_name));
  EXPECT_TRUE(cache_hit_with_name.has_value());

  // AND: loading WITHOUT name fails because it was stored WITH name
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_miss,
                         compilation_cache.TryLoadModel(cache_key, ""));
  EXPECT_FALSE(cache_miss.has_value());
}

TEST(CompilationCacheTest, CacheHitFallbackToHashOnly) {
  // GIVEN: a compilation cache and a model
  const std::string cache_root_path =
      ::testing::TempDir() + "/CacheHitFallbackToHashOnly";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  // WHEN: the model is saved to the cache WITHOUT a model name
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key, ""));

  // THEN: the model can be found in the cache even if we PROVIDE a model name
  // (due to fallback)
  std::string model_name = "test_model";
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(cache_key, model_name));
  EXPECT_TRUE(cache_hit.has_value());
}

TEST(CompilationCacheTest, CompilerPluginVersionChange_CacheMiss) {
  // GIVEN: a compilation cache and a model, saved to the cache
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  CompilationCache::CompilerPluginInfo compiler_plugin_info =
      GetTestCompilerPluginInfo();

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key));

  // WHEN: the vendor plugin API version has been updated.
  compiler_plugin_info.api_version.minor++;

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key_updated,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));

  // THEN: the model can not be loaded from the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(cache_key_updated));
  EXPECT_FALSE(cache_hit.has_value());
}

TEST(CompilationCacheTest, CompilerPluginSdkVersionChange_CacheMiss) {
  // GIVEN: a compilation cache and a model, saved to the cache
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  CompilationCache::CompilerPluginInfo compiler_plugin_info =
      GetTestCompilerPluginInfo();
  compiler_plugin_info.sdk_version = "1.0.0";

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key));

  // WHEN: the vendor plugin SDK version has been updated.
  compiler_plugin_info.sdk_version = "1.0.1";

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key_updated,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));

  // THEN: the model can not be loaded from the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(cache_key_updated));
  EXPECT_FALSE(cache_hit.has_value());
}

TEST(CompilationCacheTest, MultipleCompilerPlugins) {
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));
  CompilationCache::CompilerPluginInfo compiler_plugin_info_first =
      GetTestCompilerPluginInfo();
  CompilationCache::CompilerPluginInfo compiler_plugin_info_second = {
      .api_version = {.major = 2, .minor = 1, .patch = 0},
      .hw_accelerators = kLiteRtHwAcceleratorNpu,
      .manufacturer = "test_manufacturer_second",
  };

  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key_first,
      CompilationCache::GetModelHash(
          *model, GetTestOptions(),
          {compiler_plugin_info_first, compiler_plugin_info_second}));

  compiler_plugin_info_second.api_version.minor++;
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key_second,
      CompilationCache::GetModelHash(
          *model, GetTestOptions(),
          {compiler_plugin_info_first, compiler_plugin_info_second}));

  EXPECT_NE(cache_key_first.config_hash, cache_key_second.config_hash);
  EXPECT_EQ(cache_key_first.content_hash, cache_key_second.content_hash);

  compiler_plugin_info_second.api_version.minor--;
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key_third,
      CompilationCache::GetModelHash(
          *model, GetTestOptions(),
          {compiler_plugin_info_first, compiler_plugin_info_second}));

  EXPECT_EQ(cache_key_first.config_hash, cache_key_third.config_hash);
  EXPECT_EQ(cache_key_first.content_hash, cache_key_third.content_hash);
}

TEST(CompilationCacheTest, DifferentLiteRtOptions_CacheMiss) {
  // GIVEN: a compilation cache and a model, saved to the cache
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  LiteRtOptionsT options = GetTestOptions();

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key,
                         CompilationCache::GetModelHash(
                             *model, options, GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key));

  // WHEN: LiteRT's major version has been updated.
  options.version.major++;

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key_updated,
                         CompilationCache::GetModelHash(
                             *model, options, GetTestCompilerPluginInfo()));

  // THEN: the model can not be loaded from the cache.
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(cache_key_updated));
  EXPECT_FALSE(cache_hit.has_value());
}

TEST(CompilationCacheTest, DifferentModelContent_DifferentCachePath) {
  const std::string cache_root_path =
      ::testing::TempDir() + "/DifferentModelContent";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_1,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_2,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_add_dynamic_shape.tflite")));

  std::string model_name_1 = "model_a";
  std::string model_name_2 = "model_b";

  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key_1,
      CompilationCache::GetModelHash(*model_1, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));

  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key_2,
      CompilationCache::GetModelHash(*model_2, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));

  EXPECT_NE(cache_key_1.content_hash, cache_key_2.content_hash);

  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_1, cache_key_1, model_name_1));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_2, cache_key_2, model_name_2));

  std::string path_1 = litert::internal::Join(
      {cache_root_path, model_name_1, absl::StrCat(cache_key_1.content_hash),
       absl::StrCat(cache_key_1.config_hash, ".tflite")});
  std::string path_2 = litert::internal::Join(
      {cache_root_path, model_name_2, absl::StrCat(cache_key_2.content_hash),
       absl::StrCat(cache_key_2.config_hash, ".tflite")});

  EXPECT_TRUE(litert::internal::Exists(path_1));
  EXPECT_TRUE(litert::internal::Exists(path_2));
}

TEST(CompilationCacheTest, SameModelContent_DifferentOptions_SameDirectory) {
  const std::string cache_root_path =
      ::testing::TempDir() + "/SameModelContent";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  compilation_cache.SetMaxConfigsPerModel(2);

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  std::string model_name = "shared_name";

  auto options_1 = GetTestOptions();
  auto options_2 = GetTestOptions();
  options_2.version.major++;

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key_1,
                         CompilationCache::GetModelHash(
                             *model, options_1, GetTestCompilerPluginInfo()));

  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey cache_key_2,
                         CompilationCache::GetModelHash(
                             *model, options_2, GetTestCompilerPluginInfo()));

  EXPECT_EQ(cache_key_1.content_hash, cache_key_2.content_hash);
  EXPECT_NE(cache_key_1.config_hash, cache_key_2.config_hash);

  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model, cache_key_1, model_name));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model, cache_key_2, model_name));

  std::string dir_1 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(cache_key_1.content_hash)});
  std::string dir_2 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(cache_key_2.content_hash)});

  EXPECT_EQ(dir_1, dir_2);

  std::string path_1 = litert::internal::Join(
      {dir_1, absl::StrCat(cache_key_1.config_hash, ".tflite")});
  std::string path_2 = litert::internal::Join(
      {dir_2, absl::StrCat(cache_key_2.config_hash, ".tflite")});

  EXPECT_TRUE(litert::internal::Exists(path_1));
  EXPECT_TRUE(litert::internal::Exists(path_2));
}

TEST(CompilationCacheTest, UnnamedModel_StoredInMemDirectory) {
  const std::string cache_root_path = ::testing::TempDir() + "/UnnamedModel";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey cache_key,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));

  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, cache_key, ""));

  std::string path = litert::internal::Join(
      {cache_root_path, "mem", absl::StrCat(cache_key.content_hash),
       absl::StrCat(cache_key.config_hash, ".tflite")});

  EXPECT_TRUE(litert::internal::Exists(path));
}

TEST(CompilationCacheTest, BuildInventoryComplex) {
  const std::string cache_root_path =
      ::testing::TempDir() + "/BuildInventoryComplex";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  compilation_cache.SetMaxConfigsPerModel(2);

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_1,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_2,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_add_dynamic_shape.tflite")));

  std::string model_name = "model_a";

  // 1. Model A, Content 1, Config 1
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_a_c1_o1,
      CompilationCache::GetModelHash(*model_1, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_1, key_a_c1_o1, model_name));

  // 2. Model A, Content 1, Config 2
  auto options_2 = GetTestOptions();
  options_2.version.major++;
  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey key_a_c1_o2,
                         CompilationCache::GetModelHash(
                             *model_1, options_2, GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_1, key_a_c1_o2, model_name));

  // 3. Model B, Content 2, Config 1
  std::string model_name_b = "model_b";
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_b_c2_o1,
      CompilationCache::GetModelHash(*model_2, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_2, key_b_c2_o1, model_name_b));

  // 4. Unnamed Model, Content 1, Config 1
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_unnamed,
      CompilationCache::GetModelHash(*model_1, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model_1, key_unnamed, ""));

  // Now build inventory
  LITERT_ASSIGN_OR_ABORT(auto inventory, compilation_cache.BuildInventory());

  ASSERT_EQ(inventory.size(), 4);

  // Helper to find entry in inventory
  auto find_entry = [&](uint64_t content_hash, uint64_t config_hash,
                        absl::string_view model_id) {
    for (const auto& entry : inventory) {
      if (entry.content_hash == content_hash &&
          entry.config_hash == config_hash && entry.model_id == model_id) {
        return true;
      }
    }
    return false;
  };

  EXPECT_TRUE(find_entry(key_a_c1_o1.content_hash, key_a_c1_o1.config_hash,
                         model_name));
  EXPECT_TRUE(find_entry(key_a_c1_o2.content_hash, key_a_c1_o2.config_hash,
                         model_name));
  EXPECT_TRUE(find_entry(key_b_c2_o1.content_hash, key_b_c2_o1.config_hash,
                         model_name_b));
  EXPECT_TRUE(
      find_entry(key_unnamed.content_hash, key_unnamed.config_hash, "mem"));
}

TEST(CompilationCacheTest, Case1_ModelUpdate_RemovesOldDirectory) {
  const std::string cache_root_path =
      ::testing::TempDir() + "/Case1_ModelUpdate";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_1,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model_2,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_add_dynamic_shape.tflite")));

  std::string model_name = "my_model";

  // 1. Save Model 1
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_1,
      CompilationCache::GetModelHash(*model_1, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_1, key_1, model_name));

  // Verify directory 1 exists
  std::string dir_1 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(key_1.content_hash)});
  EXPECT_TRUE(litert::internal::Exists(dir_1));

  // 2. Save Model 2 (same name, different content)
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_2,
      CompilationCache::GetModelHash(*model_2, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(
      compilation_cache.SaveModel(*model_2, key_2, model_name));

  // Verify directory 2 exists
  std::string dir_2 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(key_2.content_hash)});
  EXPECT_TRUE(litert::internal::Exists(dir_2));

  // Verify directory 1 is REMOVED
  EXPECT_FALSE(litert::internal::Exists(dir_1));
}

TEST(CompilationCacheTest, Case2_ConfigLimit_RemovesOldestConfig) {
  const std::string cache_root_path =
      ::testing::TempDir() + "/Case2_ConfigLimit";
  LITERT_ABORT_IF_ERROR(litert::internal::MkDir(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));

  compilation_cache.SetMaxConfigsPerModel(2);

  LITERT_ASSIGN_OR_ABORT(std::unique_ptr<LiteRtModelT> model,
                         LoadModelFromFile(litert::testing::GetTestFilePath(
                             "simple_model.tflite")));

  std::string model_name = "my_model";

  // 1. Save Config 1
  LITERT_ASSIGN_OR_ABORT(
      CompilationCache::CacheKey key_1,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, key_1, model_name));
  // We sleep to ensure that the next SaveModel call results in a strictly later
  // filesystem modification time. Some filesystems have 1s granularity.
  absl::SleepFor(absl::Milliseconds(100));

  // 2. Save Config 2
  auto options_2 = GetTestOptions();
  options_2.version.major++;
  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey key_2,
                         CompilationCache::GetModelHash(
                             *model, options_2, GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, key_2, model_name));
  absl::SleepFor(absl::Milliseconds(100));

  std::string path_1 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(key_1.content_hash),
       absl::StrCat(key_1.config_hash, ".tflite")});
  std::string path_2 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(key_2.content_hash),
       absl::StrCat(key_2.config_hash, ".tflite")});
  EXPECT_TRUE(litert::internal::Exists(path_1));
  EXPECT_TRUE(litert::internal::Exists(path_2));

  // 3. Save Config 3 (exceeds limit)
  auto options_3 = GetTestOptions();
  options_3.version.major += 2;
  LITERT_ASSIGN_OR_ABORT(CompilationCache::CacheKey key_3,
                         CompilationCache::GetModelHash(
                             *model, options_3, GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, key_3, model_name));

  std::string path_3 = litert::internal::Join(
      {cache_root_path, model_name, absl::StrCat(key_3.content_hash),
       absl::StrCat(key_3.config_hash, ".tflite")});
  EXPECT_TRUE(litert::internal::Exists(path_3));

  // Verify ONE of the old ones is removed (should be Config 1)
  EXPECT_FALSE(litert::internal::Exists(path_1));
  EXPECT_TRUE(litert::internal::Exists(path_2));
}

}  // namespace litert::internal
