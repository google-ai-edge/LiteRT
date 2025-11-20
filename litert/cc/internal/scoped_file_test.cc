// Copyright 2025 The ODML Authors.
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

#include "litert/cc/internal/scoped_file.h"

#include <fcntl.h>

#include <cerrno>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert {
namespace {

#if defined(_WIN32)
#define read _read
#define close _close
#endif  // defined(_WIN32)

void WriteFile(absl::string_view path, absl::string_view contents) {
  std::ofstream ofstr(std::string(path), std::ios::out);
  ofstr << contents;
}

[[maybe_unused]]
std::string ReadFile(absl::string_view path) {
  std::ifstream ifstr{std::string(path)};
  std::stringstream sstr;
  sstr << ifstr.rdbuf();
  return sstr.str();
}

TEST(ScopedFile, FailsOpeningNonExistentFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "bad.txt";
  ASSERT_FALSE(ScopedFile::Open(path.string()).ok());
}

TEST(ScopedFile, GetSize) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = ScopedFile::Open(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  EXPECT_TRUE(file->IsValid());

  auto size = file->GetSize();
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);

  size = ScopedFile::GetSize(file->file());
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);
}

TEST(ScopedFile, GetSizeOfWritableFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = ScopedFile::OpenWritable(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  EXPECT_TRUE(file->IsValid());

  auto size = file->GetSize();
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);

  size = ScopedFile::GetSize(file->file());
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);
}

TEST(ScopedFile, MoveInvalidatesFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  absl::StatusOr<ScopedFile> file = ScopedFile::Open(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  EXPECT_TRUE(file->IsValid());

  ScopedFile other_file = std::move(*file);
  EXPECT_TRUE(other_file.IsValid());
  EXPECT_FALSE(file->IsValid());  // NOLINT: use after move is intended to check
                                  // the state.
}

TEST(ScopedFile, GetSizeOfInvalidFile) {
  ScopedFile uninitialized_file;
  auto status = uninitialized_file.GetSize();
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  status = ScopedFile::GetSize(uninitialized_file.file());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(ScopedFile, Duplicate) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), "foo bar");

  auto file = ScopedFile::Open(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  auto duplicated = file->Duplicate();
  ASSERT_EQ(duplicated.status(), absl::OkStatus());

  auto size = file->GetSize();
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);
  size = duplicated->GetSize();
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);

  // Delete original, duplicated file should still be valid.
  file = ScopedFile();
  size = duplicated->GetSize();
  ASSERT_EQ(size.status(), absl::OkStatus());
  EXPECT_EQ(*size, 7);
}

TEST(ScopedFile, ReleaseFailsWithAnInvalidFile) {
  ScopedFile uninitialized_file;
  EXPECT_FALSE(uninitialized_file.Release().ok());
}

#if defined(_WIN32)

TEST(ScopedFile, ReleaseFailsWithAnAsyncFile) {
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  const DWORD share_mode =
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  const DWORD access = GENERIC_READ | GENERIC_WRITE;
  HANDLE hfile =
      ::CreateFileW(path.native().c_str(), access, share_mode, nullptr,
                    OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
  ASSERT_NE(hfile, INVALID_HANDLE_VALUE);
  ScopedFile file(hfile);
  EXPECT_FALSE(file.Release().ok());
}

#endif  // defined(_WIN32)

TEST(ScopedFile, ReleaseWorksForAReadOnlyFile) {
  const char reference_data[] = "foo bar";
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), reference_data);

  auto file = ScopedFile::Open(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  auto fd = file->Release();
  ASSERT_EQ(fd.status(), absl::OkStatus());
  EXPECT_GE(*fd, 0);

  EXPECT_FALSE(file->IsValid());

  char data[sizeof(reference_data)] = {'a'};
  EXPECT_EQ(read(*fd, data, sizeof(reference_data) - 1),
            sizeof(reference_data) - 1)
      << strerror(errno);

  close(*fd);
}

TEST(ScopedFile, ReleaseWorksForAWritableFile) {
  const char reference_data[] = "foo bar";
  auto path = std::filesystem::path(::testing::TempDir()) / "file.txt";
  WriteFile(path.string(), reference_data);

  auto file = ScopedFile::OpenWritable(path.string());
  ASSERT_EQ(file.status(), absl::OkStatus());
  auto fd = file->Release();
  ASSERT_EQ(fd.status(), absl::OkStatus());
  EXPECT_GE(*fd, 0);

  char data[sizeof(reference_data)] = {'a'};
  EXPECT_EQ(read(*fd, data, sizeof(reference_data) - 1),
            sizeof(reference_data) - 1)
      << strerror(errno);

  EXPECT_EQ(write(*fd, reference_data, sizeof(reference_data) - 1),
            sizeof(reference_data) - 1)
      << strerror(errno);

  close(*fd);

  std::string file_contents = ReadFile(path.string());
  EXPECT_EQ(file_contents, std::string(reference_data) + reference_data);
}

}  // namespace
}  // namespace litert
