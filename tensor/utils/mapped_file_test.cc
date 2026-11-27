/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/utils/mapped_file.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::tensor {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;

std::string WriteTempFile(absl::string_view name, absl::string_view contents) {
  const std::string path = testing::TempDir() + "/" + std::string(name);
  std::ofstream out(path, std::ios::binary);
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return path;
}

TEST(MappedFileTest, MapsWholeFile) {
  const std::string contents = "GGUF mapped bytes";
  const std::string path = WriteTempFile("mapped.bin", contents);

  absl::StatusOr<MappedFile> mapped = MappedFile::OpenReadOnly(path);
  ASSERT_THAT(mapped, IsOk());
  EXPECT_TRUE(mapped->is_mapped());
  ASSERT_EQ(mapped->size(), contents.size());
  EXPECT_THAT(mapped->span(), ElementsAreArray(contents));
}

TEST(MappedFileTest, MoveTransfersOwnershipAndResetUnmaps) {
  const std::string path = WriteTempFile("moved.bin", "abc");
  absl::StatusOr<MappedFile> mapped = MappedFile::OpenReadOnly(path);
  ASSERT_THAT(mapped, IsOk());
  const uint8_t* data = mapped->data();

  MappedFile moved = *std::move(mapped);
  EXPECT_EQ(moved.data(), data);
  EXPECT_EQ(moved.size(), 3);

  moved.Reset();
  EXPECT_FALSE(moved.is_mapped());
  EXPECT_EQ(moved.size(), 0);
}

TEST(MappedFileTest, AdviseDontNeedKeepsMappingReadable) {
  const std::string contents(1 << 16, 'x');
  const std::string path = WriteTempFile("advise.bin", contents);
  absl::StatusOr<MappedFile> mapped = MappedFile::OpenReadOnly(path);
  ASSERT_THAT(mapped, IsOk());

  mapped->AdviseDontNeed(/*offset=*/1, /*length=*/contents.size());
  mapped->AdviseDontNeed(/*offset=*/contents.size() + 1, /*length=*/1);
  size_t count = 0;
  for (uint8_t byte : mapped->span()) count += byte == 'x';
  EXPECT_EQ(count, contents.size());
}

TEST(MappedFileTest, MissingFileIsNotFound) {
  EXPECT_THAT(MappedFile::OpenReadOnly(testing::TempDir() + "/missing.bin"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(MappedFileTest, EmptyFileIsInvalidArgument) {
  const std::string path = WriteTempFile("empty.bin", "");
  EXPECT_THAT(MappedFile::OpenReadOnly(path),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MappedFileTest, DirectoryIsInvalidArgument) {
  EXPECT_THAT(MappedFile::OpenReadOnly(testing::TempDir()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor
