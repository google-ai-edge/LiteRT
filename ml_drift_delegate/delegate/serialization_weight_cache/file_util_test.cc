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

#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <string>
#include <type_traits>
#include <utility>

#include "testing/base/public/gunit.h"

namespace ml_drift {
namespace {

TEST(FileDescriptorTest, DefaultConstructedIsInvalid) {
  FileDescriptor fd;
  EXPECT_FALSE(fd.IsValid());
}

TEST(FileDescriptorTest, ConstructAndRelease) {
  const int kFd = 53;
  // Construct from int.
  FileDescriptor fd(kFd);
  EXPECT_TRUE(fd.IsValid());
  EXPECT_EQ(fd.Value(), kFd);
  // Move construction
  FileDescriptor fd2(std::move(fd));
  EXPECT_FALSE(
      fd.IsValid());  // NOLINT: This "fd" is not a real file descriptor
  EXPECT_TRUE(fd2.IsValid());
  EXPECT_EQ(fd2.Value(), kFd);
  // We release because we don't own kFd.
  EXPECT_EQ(fd2.Release(), kFd);
  EXPECT_FALSE(fd2.IsValid());
  EXPECT_FALSE(std::is_copy_constructible_v<FileDescriptor>);
}

TEST(FileDescriptorTest, OpenWriteRewindAndReadWorks) {
  const std::string tmp_file = testing::TempDir() + __FUNCTION__;
  FileDescriptor fd =
      FileDescriptor::Open(tmp_file.c_str(),
                           O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
                           0644);
  ASSERT_TRUE(fd.IsValid());
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_TRUE(fd.Write(src_data.data(), src_data.size()));
  EXPECT_EQ(fd.SetPos(0), 0);
  std::string dst_data(src_data.size(), ' ');
  EXPECT_TRUE(fd.Read(dst_data.data(), src_data.size()));
  EXPECT_EQ(dst_data, src_data);
}

TEST(FileDescriptorTest, WriteFailureReturnsFalse) {
  const std::string tmp_file = testing::TempDir() + __FUNCTION__;
  FileDescriptor fd =
      FileDescriptor::Open(tmp_file.c_str(),
                           O_CREAT | O_TRUNC | O_RDONLY,  // NOLINT: b/332641196
                           0644);
  ASSERT_TRUE(fd.IsValid());
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_FALSE(fd.Write(src_data.data(), src_data.size()));
}

TEST(FileDescriptorTest, ReadFailureReturnsFalse) {
  const std::string tmp_file = testing::TempDir() + __FUNCTION__;
  FileDescriptor fd =
      FileDescriptor::Open(tmp_file.c_str(),
                           O_CREAT | O_TRUNC | O_WRONLY,  // NOLINT: b/332641196
                           0644);
  ASSERT_TRUE(fd.IsValid());
  std::string dst_data(5, ' ');
  EXPECT_FALSE(fd.Read(dst_data.data(), dst_data.size()));
}

TEST(FileDescriptorTest, TruncateSuccess) {
  const std::string tmp_file = testing::TempDir() + __FUNCTION__;
  FileDescriptor fd =
      FileDescriptor::Open(tmp_file.c_str(),
                           O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
                           0644);
  ASSERT_TRUE(fd.IsValid());
  const std::string src_data = "The quick brown fox jumps over the lazy dog.";
  EXPECT_TRUE(fd.Write(src_data.data(), src_data.size()));
  EXPECT_TRUE(fd.Truncate(10));
  EXPECT_EQ(fd.SetPos(0), 0);
  std::string dst_data(10, ' ');
  EXPECT_TRUE(fd.Read(dst_data.data(), dst_data.size()));
  EXPECT_EQ(dst_data, src_data.substr(0, 10));
  // EOF is reached when reading more than the size of the file.
  EXPECT_FALSE(fd.Read(dst_data.data(), 1));
}

}  // namespace
}  // namespace ml_drift
