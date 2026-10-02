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

#include <cstdint>
#include <string>
#include <vector>

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
    max_partitions = 3
  )";

  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(
                toml_string, std::string(toml_string).size(), &options),
            kLiteRtStatusOk);

  EXPECT_EQ(options.partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyWeaklyConnected);
  EXPECT_TRUE(options.dummy_option);
  EXPECT_EQ(options.max_partitions, 3);
}

TEST(LiteRtCompilerOptionsTest, ParseLiteRtCompilerOptionsDefaults) {
  auto toml_string = "";
  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(toml_string, 0, &options),
            kLiteRtStatusOk);
  EXPECT_EQ(options.partition_strategy,
            kLiteRtCompilerOptionsPartitionStrategyDefault);
  EXPECT_FALSE(options.dummy_option);
  EXPECT_EQ(options.max_partitions, 0);
  EXPECT_FALSE(options.HasInputShapes());
}

TEST(LiteRtCompilerOptionsTest, ParseInputShapesFromToml) {
  auto toml_string = R"(
    positional_input_shapes = ["serving_default@1:224:224:3", "@1:10"]
    tensor_input_shapes = ["serving_default@arg0@1:224:224:3"]
    signature_input_shapes = ["serving_default@image@1:224:224:3"]
  )";

  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(
                toml_string, std::string(toml_string).size(), &options),
            kLiteRtStatusOk);

  EXPECT_TRUE(options.HasInputShapes());

  ASSERT_EQ(options.positional_input_shapes.size(), 2);
  EXPECT_EQ(options.positional_input_shapes[0].signature_key,
            "serving_default");
  EXPECT_EQ(options.positional_input_shapes[0].dims,
            (std::vector<int32_t>{1, 224, 224, 3}));
  EXPECT_EQ(options.positional_input_shapes[1].signature_key, "");
  EXPECT_EQ(options.positional_input_shapes[1].dims,
            (std::vector<int32_t>{1, 10}));

  ASSERT_EQ(options.tensor_input_shapes.size(), 1);
  EXPECT_EQ(options.tensor_input_shapes[0].signature_key, "serving_default");
  EXPECT_EQ(options.tensor_input_shapes[0].name, "arg0");
  EXPECT_EQ(options.tensor_input_shapes[0].dims,
            (std::vector<int32_t>{1, 224, 224, 3}));

  ASSERT_EQ(options.signature_input_shapes.size(), 1);
  EXPECT_EQ(options.signature_input_shapes[0].signature_key, "serving_default");
  EXPECT_EQ(options.signature_input_shapes[0].name, "image");
  EXPECT_EQ(options.signature_input_shapes[0].dims,
            (std::vector<int32_t>{1, 224, 224, 3}));
}

TEST(LiteRtCompilerOptionsTest, ParseMultiSignatureInputShapes) {
  auto toml_string = R"(
    signature_input_shapes = ["prefill@tokens@1:512", "decode@tokens@1:1"]
  )";

  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(
                toml_string, std::string(toml_string).size(), &options),
            kLiteRtStatusOk);

  EXPECT_TRUE(options.HasInputShapes());
  ASSERT_EQ(options.signature_input_shapes.size(), 2);

  EXPECT_EQ(options.signature_input_shapes[0].signature_key, "prefill");
  EXPECT_EQ(options.signature_input_shapes[0].name, "tokens");
  EXPECT_EQ(options.signature_input_shapes[0].dims,
            (std::vector<int32_t>{1, 512}));

  EXPECT_EQ(options.signature_input_shapes[1].signature_key, "decode");
  EXPECT_EQ(options.signature_input_shapes[1].name, "tokens");
  EXPECT_EQ(options.signature_input_shapes[1].dims,
            (std::vector<int32_t>{1, 1}));
}

TEST(LiteRtCompilerOptionsTest, ParseRejectsMalformedInputShapeEntries) {
  for (const char* toml : {
           R"(positional_input_shapes = ["sig@"])",
           R"(positional_input_shapes = ["sig@1:x"])",
           R"(tensor_input_shapes = ["sig@@1:2"])",
           R"(tensor_input_shapes = ["1:2"])",
           R"(signature_input_shapes = ["sig@name@"])",
           R"(signature_input_shapes = ["sig@name@1:-2:x"])",
       }) {
    LiteRtCompilerOptionsT options;
    EXPECT_NE(
        ParseLiteRtCompilerOptions(toml, std::string(toml).size(), &options),
        kLiteRtStatusOk)
        << "toml: " << toml;
  }
}

TEST(LiteRtCompilerOptionsTest, RoundTripsThroughCApi) {
  LrtCompilerOptions* c_options = nullptr;
  ASSERT_EQ(LrtCreateCompilerOptions(&c_options), kLiteRtStatusOk);
  const int32_t dims[] = {1, -1, 768};
  ASSERT_EQ(LrtAddCompilerOptionsSignatureInputShape(c_options, "decode",
                                                     "tokens", dims, 3),
            kLiteRtStatusOk);
  ASSERT_EQ(
      LrtAddCompilerOptionsPositionalInputShape(c_options, nullptr, dims, 3),
      kLiteRtStatusOk);

  const char* identifier = nullptr;
  void* payload = nullptr;
  void (*payload_deleter)(void*) = nullptr;
  ASSERT_EQ(LrtGetOpaqueCompilerOptionsData(c_options, &identifier, &payload,
                                            &payload_deleter),
            kLiteRtStatusOk);
  const std::string toml = static_cast<const char*>(payload);
  payload_deleter(payload);
  LrtDestroyCompilerOptions(c_options);

  LiteRtCompilerOptionsT options;
  ASSERT_EQ(ParseLiteRtCompilerOptions(toml.data(), toml.size(), &options),
            kLiteRtStatusOk);
  ASSERT_EQ(options.positional_input_shapes.size(), 1);
  EXPECT_EQ(options.positional_input_shapes[0].signature_key, "");
  EXPECT_EQ(options.positional_input_shapes[0].dims,
            (std::vector<int32_t>{1, -1, 768}));
  ASSERT_EQ(options.signature_input_shapes.size(), 1);
  EXPECT_EQ(options.signature_input_shapes[0].signature_key, "decode");
  EXPECT_EQ(options.signature_input_shapes[0].name, "tokens");
  EXPECT_EQ(options.signature_input_shapes[0].dims,
            (std::vector<int32_t>{1, -1, 768}));
}

}  // namespace
}  // namespace internal
}  // namespace litert
