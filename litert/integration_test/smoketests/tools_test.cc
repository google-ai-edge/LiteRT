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

#include <filesystem>  // NOLINT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl

ABSL_FLAG(std::string, models_dir, "", "Dir where to find tflite models.");

using ::testing::StartsWith;

namespace litert::test {
namespace {

TEST(ToolsTest, FileIsOnDevice) {
  const auto models_dir = absl::GetFlag(FLAGS_models_dir);
  ASSERT_FALSE(models_dir.empty());
  ASSERT_THAT(models_dir, StartsWith("/data/local/tmp/runfiles"));

  auto tfl_count = 0;
  for (const auto& file : std::filesystem::directory_iterator(models_dir)) {
    if (file.path().extension() == ".tflite") {
      ++tfl_count;
    }
  }

  ASSERT_GE(tfl_count, 1);
}

}  // namespace
}  // namespace litert::test
