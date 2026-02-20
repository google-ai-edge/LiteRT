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

#include "litert/runtime/litert_runtime_options.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "toml.hpp"  // from @tomlplusplus

namespace {

using litert::internal::ParseLiteRtRuntimeOptions;

TEST(LiteRtRuntimeOptionsTest, ParseWorks) {
  std::string toml_str = R"(
    enable_profiling = true
    compress_quantization_zero_points = false
    error_reporter_mode = 1
  )";

  LiteRtRuntimeOptionsT options;
  EXPECT_EQ(
      ParseLiteRtRuntimeOptions(toml_str.data(), toml_str.size(), &options),
      kLiteRtStatusOk);

  // Verify against expected values
  EXPECT_TRUE(options.enable_profiling);
  EXPECT_FALSE(options.compress_quantization_zero_points);
  EXPECT_EQ(options.error_reporter_mode,
            static_cast<LiteRtErrorReporterMode>(1));

  // Verify against tomlplusplus
  auto toml_tbl = toml::parse(toml_str);
  EXPECT_EQ(options.enable_profiling,
            toml_tbl["enable_profiling"].value<bool>().value());
  EXPECT_EQ(
      options.compress_quantization_zero_points,
      toml_tbl["compress_quantization_zero_points"].value<bool>().value());
  EXPECT_EQ(static_cast<int>(options.error_reporter_mode),
            toml_tbl["error_reporter_mode"].value<int>().value());
}

TEST(LiteRtRuntimeOptionsTest, ParseWorksWithCommentsAndWhitespace) {
  std::string toml_str = R"(
    # This is a comment
    enable_profiling = true
    compress_quantization_zero_points = true
    error_reporter_mode = 0
  )";

  LiteRtRuntimeOptionsT options;
  EXPECT_EQ(
      ParseLiteRtRuntimeOptions(toml_str.data(), toml_str.size(), &options),
      kLiteRtStatusOk);

  EXPECT_TRUE(options.enable_profiling);
  EXPECT_TRUE(options.compress_quantization_zero_points);
  EXPECT_EQ(options.error_reporter_mode,
            static_cast<LiteRtErrorReporterMode>(0));
}

TEST(LiteRtRuntimeOptionsTest, ParseInvalidBool) {
  std::string toml_str = "enable_profiling = not_a_bool";
  LiteRtRuntimeOptionsT options;

  EXPECT_EQ(
      ParseLiteRtRuntimeOptions(toml_str.data(), toml_str.size(), &options),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_FALSE(options.enable_profiling);  // Default is unset (false)
}

}  // namespace
