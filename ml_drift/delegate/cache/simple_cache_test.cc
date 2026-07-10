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

#include "third_party/odml/litert/ml_drift/delegate/cache/simple_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/file_util.h"
#include "third_party/odml/infra/ml_drift_delegate/serialization_weight_cache/mmap_handle.h"

namespace litert::ml_drift {
namespace {

class SimpleCacheTest : public ::testing::Test {
 protected:
  void SetUp() override { cache_file_path_ = std::tmpnam(nullptr); }

  void TearDown() override { std::remove(cache_file_path_.c_str()); }

  std::string cache_file_path_;
};

TEST_F(SimpleCacheTest, DefaultConstructor) {
  SimpleCache cache;
  EXPECT_FALSE(cache.IsValid());
}

TEST_F(SimpleCacheTest, LoadWithFdEmpty) {
  ::ml_drift::FileDescriptor fd = ::ml_drift::FileDescriptor::Open(
      cache_file_path_.c_str(), O_RDWR | O_CREAT, 0644);
  SimpleCache cache(std::move(fd));
  EXPECT_TRUE(cache.IsValid());

  EXPECT_THAT(cache.Load([](absl::Span<const uint8_t> data,
                            ::ml_drift::MMapHandle& mmap_handle) {
    return absl::OkStatus();
  }),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SimpleCacheTest, LoadWithFilePathEmpty) {
  SimpleCache cache(cache_file_path_);
  EXPECT_TRUE(cache.IsValid());

  EXPECT_THAT(cache.Load([](absl::Span<const uint8_t> data,
                            ::ml_drift::MMapHandle& mmap_handle) {
    return absl::OkStatus();
  }),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SimpleCacheTest, StoreAndLoadWithFd) {
  ::ml_drift::FileDescriptor fd = ::ml_drift::FileDescriptor::Open(
      cache_file_path_.c_str(), O_RDWR | O_CREAT, 0644);
  SimpleCache cache(std::move(fd));
  EXPECT_TRUE(cache.IsValid());

  std::vector<uint8_t> data = {1, 2, 3, 4, 5};
  ABSL_EXPECT_OK(cache.Store(data));

  ABSL_EXPECT_OK(cache.Load([&data](absl::Span<const uint8_t> loaded_data,
                                    ::ml_drift::MMapHandle& mmap_handle) {
    EXPECT_EQ(loaded_data.size(), data.size());
    EXPECT_EQ(loaded_data, data);
    return absl::OkStatus();
  }));
}

TEST_F(SimpleCacheTest, StoreAndLoadWithFilePath) {
  SimpleCache cache(cache_file_path_);
  EXPECT_TRUE(cache.IsValid());

  std::vector<uint8_t> data = {1, 2, 3, 4, 5};
  ABSL_EXPECT_OK(cache.Store(data));

  ABSL_EXPECT_OK(cache.Load([&data](absl::Span<const uint8_t> loaded_data,
                                    ::ml_drift::MMapHandle& mmap_handle) {
    EXPECT_EQ(loaded_data.size(), data.size());
    EXPECT_EQ(loaded_data, data);
    return absl::OkStatus();
  }));
}

TEST_F(SimpleCacheTest, Persistence) {
  std::vector<uint8_t> data = {10, 20, 30};

  // 1. Create cache, store data, destroy cache (flushes to file)
  {
    SimpleCache cache(cache_file_path_);
    ABSL_EXPECT_OK(cache.Store(data));
  }

  // 2. Load cache from same file, verify data exists
  {
    SimpleCache cache(cache_file_path_);
    ABSL_EXPECT_OK(cache.Load([&data](absl::Span<const uint8_t> loaded_data,
                                      ::ml_drift::MMapHandle& mmap_handle) {
      EXPECT_EQ(loaded_data.size(), data.size());
      EXPECT_EQ(loaded_data, data);
      return absl::OkStatus();
    }));
  }
}

TEST_F(SimpleCacheTest, OverwriteData) {
  std::vector<uint8_t> data1 = {0xA, 0xB};
  std::vector<uint8_t> data2 = {0xC, 0xD, 0xE};

  // 1. Create initial cache
  {
    SimpleCache cache(cache_file_path_);
    ABSL_EXPECT_OK(cache.Store(data1));
  }

  // 2. Load cache from same file, verify data exists, then overwrite it.
  {
    SimpleCache cache(cache_file_path_);
    ABSL_EXPECT_OK(cache.Load([&data1](absl::Span<const uint8_t> loaded_data,
                                       ::ml_drift::MMapHandle& mmap_handle) {
      EXPECT_EQ(loaded_data.size(), data1.size());
      EXPECT_EQ(loaded_data, data1);
      return absl::OkStatus();
    }));

    ABSL_EXPECT_OK(cache.Store(data2));
  }

  // 3. Load cache from same file, verify data overwritten
  {
    SimpleCache cache(cache_file_path_);
    ABSL_EXPECT_OK(cache.Load([&data2](absl::Span<const uint8_t> loaded_data,
                                       ::ml_drift::MMapHandle& mmap_handle) {
      EXPECT_EQ(loaded_data.size(), data2.size());
      EXPECT_EQ(loaded_data, data2);
      return absl::OkStatus();
    }));
  }
}

}  // namespace
}  // namespace litert::ml_drift
