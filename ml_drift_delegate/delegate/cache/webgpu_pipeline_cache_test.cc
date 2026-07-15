// Copyright 2026 Google LLC.
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

#include "ml_drift_delegate/delegate/cache/webgpu_pipeline_cache.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/cache/simple_cache.h"

namespace litert::ml_drift {
namespace {

class WebGpuPipelineCacheTest : public ::testing::Test {
 protected:
  void SetUp() override { cache_file_path_ = std::tmpnam(nullptr); }

  void TearDown() override { std::remove(cache_file_path_.c_str()); }

  std::string cache_file_path_;
};

TEST_F(WebGpuPipelineCacheTest, LoadFromSafeFile) {
  WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
  std::vector<uint8_t> buffer(10);
  EXPECT_EQ(cache.Load(123, absl::MakeSpan(buffer)), 0);
}

TEST_F(WebGpuPipelineCacheTest, StoreAndLoad) {
  WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
  std::vector<uint8_t> data = {1, 2, 3, 4, 5};
  EXPECT_TRUE(cache.Store(123, data));

  std::vector<uint8_t> buffer(5);
  size_t size = cache.Load(123, absl::MakeSpan(buffer));
  EXPECT_EQ(size, 5);
  EXPECT_EQ(buffer, data);
}

TEST_F(WebGpuPipelineCacheTest, LoadSizeOnly) {
  WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
  std::vector<uint8_t> data = {1, 2, 3, 4, 5};
  EXPECT_TRUE(cache.Store(123, data));
  EXPECT_EQ(cache.Load(123, {}), 5);
}

TEST_F(WebGpuPipelineCacheTest, Overflow) {
  WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 1);
  std::vector<uint8_t> data1 = {1, 2, 3, 4, 5};
  EXPECT_TRUE(cache.Store(123, data1));
  std::vector<uint8_t> data2 = {6, 7, 8, 9, 10};
  EXPECT_FALSE(cache.Store(456, data2));
  // Verify data1 is still in the cache.
  std::vector<uint8_t> buffer(5);
  EXPECT_EQ(cache.Load(123, absl::MakeSpan(buffer)), 5);
  EXPECT_EQ(buffer, data1);
  // Verify data2 is not in the cache.
  EXPECT_EQ(cache.Load(456, absl::MakeSpan(buffer)), 0);
  // Verify data1 can be updated.
  data1 = {11, 12, 13, 14, 15};
  EXPECT_TRUE(cache.Store(123, data1));
  EXPECT_EQ(cache.Load(123, absl::MakeSpan(buffer)), 5);
  EXPECT_EQ(buffer, data1);
}

TEST_F(WebGpuPipelineCacheTest, Persistence) {
  // 1. Create cache, store data, destroy cache (flushes to file)
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> data = {10, 20, 30};
    EXPECT_TRUE(cache.Store(456, data));
  }

  // 2. Load cache from same file, verify data exists
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> buffer(3);
    size_t size = cache.Load(456, absl::MakeSpan(buffer));
    EXPECT_EQ(size, 3);
    std::vector<uint8_t> expected = {10, 20, 30};
    EXPECT_EQ(buffer, expected);
  }
}

TEST_F(WebGpuPipelineCacheTest, OverwriteData) {
  WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
  std::vector<uint8_t> data1 = {1, 1, 1};
  EXPECT_TRUE(cache.Store(789, data1));

  std::vector<uint8_t> data2 = {2, 2, 2, 2};
  EXPECT_TRUE(cache.Store(789, data2));

  std::vector<uint8_t> buffer(4);
  size_t size = cache.Load(789, absl::MakeSpan(buffer));
  EXPECT_EQ(size, 4);
  EXPECT_EQ(buffer, data2);
}

TEST_F(WebGpuPipelineCacheTest, AppendToExistingCache) {
  // 1. Create initial cache
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> data1 = {0xA, 0xB};
    EXPECT_TRUE(cache.Store(1001, data1));
  }

  // 2. Open existing cache and add new item
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> data2 = {0xC, 0xD, 0xE};
    EXPECT_TRUE(cache.Store(1002, data2));
  }

  // 3. Verify both items exist
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);

    std::vector<uint8_t> buffer1(2);
    EXPECT_EQ(cache.Load(1001, absl::MakeSpan(buffer1)), 2);
    std::vector<uint8_t> expected1 = {0xA, 0xB};
    EXPECT_EQ(buffer1, expected1);

    std::vector<uint8_t> buffer2(3);
    EXPECT_EQ(cache.Load(1002, absl::MakeSpan(buffer2)), 3);
    std::vector<uint8_t> expected2 = {0xC, 0xD, 0xE};
    EXPECT_EQ(buffer2, expected2);
  }
}

TEST_F(WebGpuPipelineCacheTest, OverwritePersistentData) {
  // 1. Create initial cache
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> data1 = {0xA, 0xB};
    EXPECT_TRUE(cache.Store(1001, data1));
  }

  // 2. Load cache from same file, verify data exists, then overwrite it.
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> buffer(2);
    size_t size = cache.Load(1001, absl::MakeSpan(buffer));
    EXPECT_EQ(size, 2);
    std::vector<uint8_t> expected = {0xA, 0xB};
    EXPECT_EQ(buffer, expected);

    std::vector<uint8_t> data2 = {0xC, 0xD, 0xE};
    EXPECT_TRUE(cache.Store(1001, data2));
  }

  // 3. Load cache from same file, verify data overwritten
  {
    WebGpuPipelineCache cache(SimpleCache(cache_file_path_), 100);
    std::vector<uint8_t> buffer(3);
    EXPECT_EQ(cache.Load(1001, absl::MakeSpan(buffer)), 3);
    std::vector<uint8_t> expected = {0xC, 0xD, 0xE};
    EXPECT_EQ(buffer, expected);
  }
}

}  // namespace
}  // namespace litert::ml_drift
