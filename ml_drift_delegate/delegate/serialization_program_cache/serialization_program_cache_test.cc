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

#include "ml_drift_delegate/delegate/serialization_program_cache/serialization_program_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196
#include <unistd.h>

#include <string>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"

namespace ml_drift {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

FileDescriptor OpenTestFile(const std::string& name) {
  std::string path = testing::TempDir() + "/" + name;
  return FileDescriptor::Open(
      path.c_str(),
      O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
      0644);
}

TEST(SerializationProgramCacheTest, ReadNonExistentKeyFails) {
  FileDescriptor fd = OpenTestFile("program_cache_test_notfound.bin");
  ASSERT_TRUE(fd.IsValid());
  SerializationProgramCache cache(fd.Release());

  EXPECT_THAT(cache.LookUp(123), StatusIs(absl::StatusCode::kNotFound));
}

TEST(SerializationProgramCacheTest, WriteAndRead) {
  FileDescriptor fd = OpenTestFile("program_cache_test_rw.bin");
  ASSERT_TRUE(fd.IsValid());
  SerializationProgramCache cache(fd.Release());

  std::string value = "program_bytecode";
  ASSERT_OK(cache.Insert(1, value));

  EXPECT_THAT(cache.LookUp(1), IsOkAndHolds(value));
}

TEST(SerializationProgramCacheTest, UpdateValue) {
  FileDescriptor fd = OpenTestFile("program_cache_test_update.bin");
  ASSERT_TRUE(fd.IsValid());
  SerializationProgramCache cache(fd.Release());

  ASSERT_OK(cache.Insert(1, "val1"));
  EXPECT_THAT(cache.LookUp(1), IsOkAndHolds("val1"));

  ASSERT_OK(cache.Insert(1, "val2_longer"));
  EXPECT_THAT(cache.LookUp(1), IsOkAndHolds("val2_longer"));
}

TEST(SerializationProgramCacheTest, MultipleKeys) {
  FileDescriptor fd = OpenTestFile("program_cache_test_multi.bin");
  ASSERT_TRUE(fd.IsValid());
  SerializationProgramCache cache(fd.Release());

  ASSERT_OK(cache.Insert(1, "val1"));
  ASSERT_OK(cache.Insert(2, "val2"));

  EXPECT_THAT(cache.LookUp(1), IsOkAndHolds("val1"));
  EXPECT_THAT(cache.LookUp(2), IsOkAndHolds("val2"));
}

TEST(SerializationProgramCacheTest, Persistence) {
  std::string path = testing::TempDir() + "/program_cache_test_persist.bin";
  {
    FileDescriptor fd =
        FileDescriptor::Open(path.c_str(),
                             O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
                             0644);
    ASSERT_TRUE(fd.IsValid());
    SerializationProgramCache cache(fd.Release());
    ASSERT_OK(cache.Insert(1, "persist"));
  }
  {
    FileDescriptor fd = FileDescriptor::Open(path.c_str(), O_RDWR);
    ASSERT_TRUE(fd.IsValid());
    SerializationProgramCache cache(fd.Release());
    EXPECT_THAT(cache.LookUp(1), IsOkAndHolds("persist"));
  }
}

TEST(SerializationProgramCacheTest, ConstructorWithPath) {
  std::string path = testing::TempDir() + "/program_cache_test_ctor_path.bin";
  // Ensure file doesn't exist initially.
  unlink(path.c_str());

  {
    SerializationProgramCache cache(path);
    std::string value = "test_data";
    ASSERT_OK(cache.Insert(100, value));
    EXPECT_THAT(cache.LookUp(100), IsOkAndHolds(value));
  }

  // Re-open using path constructor
  {
    SerializationProgramCache cache(path);
    EXPECT_THAT(cache.LookUp(100), IsOkAndHolds("test_data"));
  }
}

TEST(SerializationProgramCacheTest, ConstructorWithDirAndToken) {
  std::string dir = testing::TempDir();
  std::string token = "my_model_token";
  std::string expected_path = dir + "/my_model_token_mldrift_program_cache.bin";

  // Ensure file doesn't exist initially.
  unlink(expected_path.c_str());

  {
    SerializationProgramCache cache(dir, token);
    std::string value = "token_data";
    ASSERT_OK(cache.Insert(200, value));
    EXPECT_THAT(cache.LookUp(200), IsOkAndHolds(value));
  }

  // Check if file exists at expected path
  ASSERT_EQ(access(expected_path.c_str(), F_OK), 0);

  // Re-open using the same constructor
  {
    SerializationProgramCache cache(dir, token);
    EXPECT_THAT(cache.LookUp(200), IsOkAndHolds("token_data"));
  }
}

}  // namespace
}  // namespace ml_drift
