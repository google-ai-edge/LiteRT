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

#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/testing_util.h"

namespace mldrift {

namespace {

using ml_drift::MMapHandle;
using testing::ElementsAreArray;
using testing::Ge;
using testing_util::TempFileDesc;

TEST(MMapHandleTest, DefaultConstructs) {
  MMapHandle handle;
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MapNonExistingFileFails) {
  // I hope this path doesn't exist...
  const char* file_path = "sdbgfd";
  MMapHandle handle;
  EXPECT_THAT(handle.Map(file_path),
              testing::status::StatusIs(
                  ::util::error::INVALID_ARGUMENT,
                  testing::HasSubstr("Cannot mmap invalid file descriptor")));
}

TEST(MMapHandleTest, MapExistingFileWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file(testing::TempDir() + "/weight_cache_test_file.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(),  // NOLINT: b/332641196
                  payload.c_str(), size(payload)),
            size(payload));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath()));
  EXPECT_TRUE(handle.IsMapped());
  EXPECT_NE(handle.data(), nullptr);
  EXPECT_THAT(handle.size(), Ge(size(payload)));
  EXPECT_THAT(handle, ElementsAreArray(payload));

  handle.UnMap();
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
}

TEST(MMapHandleTest, MoveConstructs) {
  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file(testing::TempDir() + "/weight_cache_test_file.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), size(payload)),
            size(payload));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath()));

  MMapHandle handle2(std::move(handle));

  // We are checking that the moved from handle has lost control over the data.
  // NOLINTBEGIN(bugprone-use-after-move)
  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);
  // NOLINTEND(bugprone-use-after-move)

  EXPECT_TRUE(handle2.IsMapped());
  EXPECT_NE(handle2.data(), nullptr);
  EXPECT_THAT(handle2.size(), Ge(size(payload)));
  EXPECT_THAT(handle2, ElementsAreArray(payload));
}

TEST(MMapHandleTest, MapWithOffset) {
  const std::string payload = "This is some data in the file.";
  const std::string payload2 = "Some other data appended to the the offset.";

  TempFileDesc tmp_file(testing::TempDir() + "/weight_cache_test_file.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), size(payload)),
            size(payload));
  ASSERT_EQ(write(tmp_file.GetFd(), payload2.c_str(), size(payload2)),
            size(payload2));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath(), /*offset=*/size(payload)));
  EXPECT_EQ(handle.size(), size(payload2));
  EXPECT_THAT(std::string((const char*)handle.data(), handle.size()),
              testing::StrEq(payload2));
}

TEST(MMapHandleTest, MapWithOffsetAndSize) {
  const std::string payload = "This is some data in the file.";
  const std::string payload2 = "Some other data appended to the the offset.";
  const std::string payload3 = "And some final data that should be ignored.";

  TempFileDesc tmp_file(testing::TempDir() + "/weight_cache_test_file.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), size(payload)),
            size(payload));
  ASSERT_EQ(write(tmp_file.GetFd(), payload2.c_str(), size(payload2)),
            size(payload2));
  ASSERT_EQ(write(tmp_file.GetFd(), payload2.c_str(), size(payload2)),
            size(payload3));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath(), /*offset=*/size(payload),
                       /*size=*/size(payload2)));
  EXPECT_EQ(handle.size(), size(payload2));
  EXPECT_THAT(std::string((const char*)handle.data(), handle.size()),
              testing::StrEq(payload2));
}

TEST(MMapHandleTest, MapTakeOwnership) {
  using std::size;

  const std::string payload = "This is some data in the file.";

  TempFileDesc tmp_file(testing::TempDir() + "/weight_cache_test_file.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(),  // NOLINT: b/332641196
                  payload.c_str(), size(payload)),
            size(payload));
  tmp_file.Close();

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath()));
  EXPECT_TRUE(handle.IsMapped());
  EXPECT_NE(handle.data(), nullptr);
  EXPECT_THAT(handle.size(), Ge(size(payload)));
  EXPECT_THAT(handle, ElementsAreArray(payload));

  auto release_data_callback = handle.TakeOwnership();
  EXPECT_NE(release_data_callback, nullptr);

  EXPECT_FALSE(handle.IsMapped());
  EXPECT_EQ(handle.data(), nullptr);
  EXPECT_EQ(handle.size(), 0);

  release_data_callback.reset();
}

TEST(MMapHandleTest, MapZeroBytesFails) {
  TempFileDesc tmp_file(testing::TempDir() + "/mmap_zero_bytes.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  tmp_file.Close();

  MMapHandle handle;
  EXPECT_THAT(
      handle.Map(tmp_file.GetCPath()),
      testing::status::StatusIs(::util::error::INVALID_ARGUMENT,
                                testing::HasSubstr("Cannot mmap 0 bytes")));
}

TEST(MMapHandleTest, MapBeyondEOFFails) {
  const std::string payload = "data";
  TempFileDesc tmp_file(testing::TempDir() + "/mmap_beyond_eof.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), payload.size()),
            payload.size());
  tmp_file.Close();

  MMapHandle handle;
  EXPECT_THAT(
      handle.Map(tmp_file.GetCPath(), /*offset=*/0, /*size=*/100),
      testing::status::StatusIs(::util::error::INVALID_ARGUMENT,
                                testing::HasSubstr("Cannot mmap beyond EOF")));
}

TEST(MMapHandleTest, MapIntegerOverflowFails) {
  const std::string payload = "data";
  TempFileDesc tmp_file(testing::TempDir() + "/mmap_integer_overflow.XXXXXX");
  ASSERT_TRUE(tmp_file.IsOpen());
  ASSERT_EQ(write(tmp_file.GetFd(), payload.c_str(), payload.size()),
            payload.size());
  tmp_file.Close();

  MMapHandle handle;
  // Use a massive offset that will wrap around when size is added.
  size_t huge_offset = std::numeric_limits<size_t>::max() - 2;
  size_t size = 4;  // offset + size = 1 (wraps around)

  EXPECT_THAT(
      handle.Map(tmp_file.GetCPath(), huge_offset, size),
      testing::status::StatusIs(::util::error::INVALID_ARGUMENT,
                                testing::HasSubstr("Cannot mmap beyond EOF")));
}

}  // namespace
}  // namespace mldrift
