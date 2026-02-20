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

#include "litert/compiler/plugin/litert_compiler_options.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"

namespace litert {
namespace internal {
namespace {

TEST(LiteRtCompilerOptionsTest, ParseLiteRtCompilerOptions) {
  auto toml_string = R"(
    partition_strategy = 1
    dummy_option = true
  )";

  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(
                toml_string, std::string(toml_string).size(), &options),
            kLiteRtStatusOk);

  EXPECT_EQ(options.partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  EXPECT_TRUE(options.dummy_option);
}

TEST(LiteRtCompilerOptionsTest, ParseLiteRtCompilerOptionsDefaults) {
  auto toml_string = "";
  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(toml_string, 0, &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyDefault);
  EXPECT_FALSE(options.dummy_option);
}

}  // namespace
}  // namespace internal
}  // namespace litert
