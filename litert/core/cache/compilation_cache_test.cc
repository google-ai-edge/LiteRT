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
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"

namespace litert::internal {

TEST(CompilationCacheTest, CacheMiss) {
  // GIVEN: a compilation cache and a model
  const std::string cache_root_path = ::testing::TempDir();
  LITERT_ASSIGN_OR_ABORT(CompilationCache compilation_cache,
                         CompilationCache::Create(cache_root_path));
  LITERT_ASSIGN_OR_ABORT(
      std::unique_ptr<LiteRtModelT> model,
      LoadModelFromFile(litert::testing::GetTestFilePath(kModelFileName)));

  // WHEN: the model has not been saved to the cache
  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash,
                         CompilationCache::GetModelHash(*model));

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
  LITERT_ASSIGN_OR_ABORT(const std::size_t model_hash,
                         CompilationCache::GetModelHash(*model));
  LITERT_ABORT_IF_ERROR(compilation_cache.SaveModel(*model, model_hash));

  // THEN: the model can be found in the cache
  LITERT_ASSIGN_OR_ABORT(std::optional<LiteRtModelT::Ptr> cache_hit,
                         compilation_cache.TryLoadModel(model_hash));
  EXPECT_TRUE(cache_hit.has_value());
}

}  // namespace litert::internal
