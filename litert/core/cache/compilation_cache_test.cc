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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/cc/litert_macros.h"
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
      const std::size_t model_hash,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));

  // THEN: the model is not found in the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_miss,
                         compilation_cache.TryLoadModel(model_hash));
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
      const std::size_t model_hash,
      CompilationCache::GetModelHash(*model, GetTestOptions(),
                                     GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, model_hash));

  // THEN: the model can be found in the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(model_hash));
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

  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, model_hash));

  // WHEN: the vendor plugin API version has been updated.
  compiler_plugin_info.api_version.minor++;

  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash_with_updated_api_version,
                         CompilationCache::GetModelHash(
                             *model, GetTestOptions(), compiler_plugin_info));

  // THEN: the model can not be loaded from the cache
  LITERT_ASSIGN_OR_ABORT(
      std::optional<LiteRtModelT::Ptr> cache_hit,
      compilation_cache.TryLoadModel(model_hash_with_updated_api_version));
  EXPECT_FALSE(cache_hit.has_value());
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

  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash,
                         CompilationCache::GetModelHash(
                             *model, options, GetTestCompilerPluginInfo()));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, model_hash));

  // WHEN: LiteRT's major version has been updated.
  options.version.major++;

  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash_with_updated_api_version,
                         CompilationCache::GetModelHash(
                             *model, options, GetTestCompilerPluginInfo()));

  // THEN: the model can not be loaded from the cache.
  LITERT_ASSIGN_OR_ABORT(
      std::optional<LiteRtModelT::Ptr> cache_hit,
      compilation_cache.TryLoadModel(model_hash_with_updated_api_version));
  EXPECT_FALSE(cache_hit.has_value());
}

}  // namespace litert::internal
