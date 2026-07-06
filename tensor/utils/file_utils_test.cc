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

#include "tensor/utils/file_utils.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <system_error>  // NOLINT

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::tensor {
namespace {

TEST(FileUtilsTest, CreateAndRemoveTempFile) {
  auto temp_file_or = CreateTempFile("test_prefix");
  ASSERT_TRUE(temp_file_or.ok());
  std::string temp_file = *temp_file_or;

  // Verify file exists
  EXPECT_TRUE(std::filesystem::exists(temp_file));

  // Write some data to it
  {
    std::ofstream out(temp_file);
    out << "test data";
  }

  // Remove file
  auto status = RemoveFile(temp_file);
  EXPECT_TRUE(status.ok());

  // Verify file no longer exists
  EXPECT_FALSE(std::filesystem::exists(temp_file));
}

TEST(FileUtilsTest, RemoveNonExistentFileIsOk) {
  std::error_code ec;
  auto temp_dir = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec);
  auto non_existent = temp_dir / "definitely_not_exist_123456";

  auto status = RemoveFile(non_existent.generic_string());
  EXPECT_TRUE(status.ok());
}

}  // namespace
}  // namespace litert::tensor
