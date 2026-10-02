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
#include "litert/tools/flags/apply_plugin_flags.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_compiler_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_compiler_options.h"
#include "litert/test/matchers.h"

namespace litert {
namespace {
TEST(ApplyPluginFlagsTest, MalformedPartitionStrategyFailsToParse) {
  std::string error;
  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  EXPECT_FALSE(AbslParseFlag("not", &partition_strategy, &error));
  EXPECT_FALSE(AbslParseFlag("a real", &partition_strategy, &error));
  EXPECT_FALSE(AbslParseFlag("flag", &partition_strategy, &error));
}

TEST(ApplyPluginFlagsTest, ParsePartitionStrategySuccess) {
  std::string error;
  LiteRtCompilerOptionsPartitionStrategy partition_strategy;
  EXPECT_TRUE(AbslParseFlag("default", &partition_strategy, &error));
  EXPECT_EQ(partition_strategy, kLiteRtCompilerOptionsPartitionStrategyDefault);
  EXPECT_EQ(AbslUnparseFlag(partition_strategy), "default");
  EXPECT_TRUE(AbslParseFlag("weakly_connected", &partition_strategy, &error));
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  EXPECT_EQ(AbslUnparseFlag(partition_strategy), "weakly_connected");
}

TEST(ApplyPluginFlagsTest, UnparsePartitionStrategySuccess) {
  EXPECT_EQ(AbslUnparseFlag(kLiteRtCompilerOptionsPartitionStrategyDefault),
            "default");
  EXPECT_EQ(
      AbslUnparseFlag(kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected),
      "weakly_connected");
}

TEST(ApplyPluginFlagsTest, ParseFlagsAndGetPartitionStrategySuccess) {
  absl::SetFlag(&FLAGS_partition_strategy,
                kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);

  Expected<CompilerOptions> options = CompilerOptions::Create();
  ASSERT_TRUE(options.HasValue());
  ASSERT_TRUE(UpdateCompilerOptionsFromFlags(options.Value()).HasValue());
  EXPECT_TRUE(options.Value().GetPartitionStrategy().HasValue());
  LITERT_ASSIGN_OR_ABORT(auto partition_strategy,
                         options.Value().GetPartitionStrategy());
  EXPECT_EQ(partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
}

// Serializes CompilerOptions to its TOML payload.
std::string ToToml(const CompilerOptions& options) {
  const char* identifier;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  if (LrtGetOpaqueCompilerOptionsData(options.Get(), &identifier, &payload,
                                      &payload_deleter) != kLiteRtStatusOk) {
    return "";
  }
  std::string toml_str(static_cast<const char*>(payload));
  if (payload_deleter) payload_deleter(payload);
  return toml_str;
}

class ApplyPluginInputShapeFlagsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::SetFlag(&FLAGS_signature, "");
    absl::SetFlag(&FLAGS_input, {});
    absl::SetFlag(&FLAGS_input_name, {});
    absl::SetFlag(&FLAGS_signature_name, {});
  }
};

TEST_F(ApplyPluginInputShapeFlagsTest, PositionalInputShapes) {
  absl::SetFlag(&FLAGS_signature, "serving_default");
  absl::SetFlag(&FLAGS_input, {"1:224:224:3", "1:-1"});

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());
  LITERT_ASSERT_OK(UpdateCompilerOptionsFromFlags(options));

  EXPECT_THAT(
      ToToml(options),
      ::testing::HasSubstr(
          "positional_input_shapes = "
          "[\"serving_default@1:224:224:3\", \"serving_default@1:-1\"]"));
}

TEST_F(ApplyPluginInputShapeFlagsTest, NamedInputShapes) {
  absl::SetFlag(&FLAGS_input_name, {"arg0@1:224:224:3"});
  absl::SetFlag(&FLAGS_signature_name, {"image@1:224:224:3"});

  LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());
  LITERT_ASSERT_OK(UpdateCompilerOptionsFromFlags(options));

  const std::string toml_str = ToToml(options);
  EXPECT_THAT(toml_str, ::testing::HasSubstr(
                            "tensor_input_shapes = [\"@arg0@1:224:224:3\"]"));
  EXPECT_THAT(toml_str,
              ::testing::HasSubstr(
                  "signature_input_shapes = [\"@image@1:224:224:3\"]"));
}

TEST_F(ApplyPluginInputShapeFlagsTest, MalformedPositionalSpecsFail) {
  for (const char* bad : {"", "1:abc:3", "1::3", "a@1:2"}) {
    absl::SetFlag(&FLAGS_input, {bad});
    LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());
    EXPECT_FALSE(UpdateCompilerOptionsFromFlags(options).HasValue())
        << "spec: " << bad;
  }
}

TEST_F(ApplyPluginInputShapeFlagsTest, MalformedNamedSpecsFail) {
  for (const char* bad : {"missing_at_sign", "@1:2:3", "arg0@", "arg0@1:x"}) {
    absl::SetFlag(&FLAGS_input_name, {bad});
    absl::SetFlag(&FLAGS_signature_name, {});
    LITERT_ASSERT_OK_AND_ASSIGN(auto options, CompilerOptions::Create());
    EXPECT_FALSE(UpdateCompilerOptionsFromFlags(options).HasValue())
        << "input_name spec: " << bad;

    absl::SetFlag(&FLAGS_input_name, {});
    absl::SetFlag(&FLAGS_signature_name, {bad});
    LITERT_ASSERT_OK_AND_ASSIGN(auto options2, CompilerOptions::Create());
    EXPECT_FALSE(UpdateCompilerOptionsFromFlags(options2).HasValue())
        << "signature_name spec: " << bad;
  }
}

}  // namespace
}  // namespace litert
