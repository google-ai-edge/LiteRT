
// Copyright 2024 Google LLC.
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

#include "litert/core/filesystem.h"

#include <fstream>
#include <ios>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::internal {
namespace {

using ::testing::UnorderedElementsAre;

static constexpr absl::string_view kPrefix = "a/prefix";
static constexpr absl::string_view kInfix = "an/infix";
static constexpr absl::string_view kSuffix = "suffix.ext";
static constexpr absl::string_view kPath = "a/prefix.ext";
static constexpr absl::string_view kStem = "prefix";


TEST(FilesystemTest, JoinTwo) {
  const auto path = Join({kPrefix, kSuffix});
  EXPECT_EQ(path, absl::StrFormat("%s/%s", kPrefix, kSuffix));
}

TEST(FilesystemTest, JoinMany) {
  const auto path = Join({kPrefix, kInfix, kSuffix});
  EXPECT_EQ(path, absl::StrFormat("%s/%s/%s", kPrefix, kInfix, kSuffix));
}

TEST(FilesystemTest, Stem) {
  const auto stem = Stem(kPath);
  EXPECT_EQ(stem, kStem);
}

void WriteFile(absl::string_view path, absl::string_view content) {
  std::ofstream ofs((std::string(path)), std::ios::binary);
  ofs << content;
}

TEST(FilesystemTest, MkDirExistsIsDir) {
  const std::string dir = Join({::testing::TempDir(), "test_dir"});
  EXPECT_FALSE(Exists(dir));
  auto status = MkDir(dir);
  ASSERT_TRUE(status);
  EXPECT_TRUE(Exists(dir));
  EXPECT_TRUE(IsDir(dir));
  EXPECT_FALSE(IsDir(Join({dir, "foo"})));
}

TEST(FilesystemTest, TouchExists) {
  const std::string file = Join({::testing::TempDir(), "test_file"});
  EXPECT_FALSE(Exists(file));
  Touch(file);
  EXPECT_TRUE(Exists(file));
  EXPECT_FALSE(IsDir(file));
}

TEST(FilesystemTest, Size) {
  const std::string file = Join({::testing::TempDir(), "test_file_size"});
  WriteFile(file, "1234");
  auto size = Size(file);
  ASSERT_TRUE(size);
  EXPECT_EQ(*size, 4);
}

TEST(FilesystemTest, LoadBinaryFile) {
  const std::string file = Join({::testing::TempDir(), "test_file_load"});
  const std::string content = "12345";
  WriteFile(file, content);
  auto buffer = LoadBinaryFile(file);
  ASSERT_TRUE(buffer);
  EXPECT_EQ(buffer->Size(), 5);
  EXPECT_EQ(absl::string_view(buffer->StrData(), buffer->Size()), content);
}

TEST(FilesystemTest, ListDir) {
  const std::string dir = Join({::testing::TempDir(), "list_dir_test"});
  auto status = MkDir(dir);
  ASSERT_TRUE(status);
  const std::string file1 = Join({dir, "file1.txt"});
  const std::string file2 = Join({dir, "file2.txt"});
  Touch(file1);
  Touch(file2);
  auto list = ListDir(dir);
  ASSERT_TRUE(list);
  EXPECT_THAT(*list, UnorderedElementsAre(file1, file2));
}

TEST(FilesystemTest, Filename) {
  const std::string dir = Join({::testing::TempDir(), "filename_test"});
  auto status = MkDir(dir);
  ASSERT_TRUE(status);
  const std::string file = Join({dir, "file1.txt"});
  Touch(file);
  auto filename = Filename(file);
  ASSERT_TRUE(filename);
  EXPECT_EQ(*filename, "file1.txt");
}

TEST(FilesystemTest, Parent) {
  const std::string dir = Join({::testing::TempDir(), "parent_test"});
  auto status = MkDir(dir);
  ASSERT_TRUE(status);
  const std::string file = Join({dir, "file1.txt"});
  Touch(file);
  auto parent = Parent(file);
  ASSERT_TRUE(parent);
  EXPECT_EQ(*parent, dir);
}

TEST(FilesystemTest, RmDir) {
  const std::string dir = Join({::testing::TempDir(), "rm_dir_test"});
  auto status = MkDir(dir);
  ASSERT_TRUE(status);
  const std::string file = Join({dir, "file1.txt"});
  Touch(file);
  EXPECT_TRUE(Exists(dir));
  auto rm_status = RmDir(dir);
  ASSERT_TRUE(rm_status);
  EXPECT_FALSE(Exists(dir));
}

}  // namespace
}  // namespace litert::internal

