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

#include "litert/c/options/litert_google_tensor_options.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/test/matchers.h"

namespace litert::google_tensor {
namespace {

void SerializeAndParse(LrtGoogleTensorOptions payload,
                       LrtGoogleTensorOptions* parsed) {
  const char* identifier;
  void* raw_payload = nullptr;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueGoogleTensorOptionsData(
      payload, &identifier, &raw_payload, &payload_deleter));
  EXPECT_STREQ(identifier, "google_tensor");
  const char* toml_str = static_cast<const char*>(raw_payload);

  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptionsFromToml(toml_str, parsed));

  payload_deleter(raw_payload);
}

TEST(LrtGoogleTensorOptionsTest, CreateAndGet) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  const char* identifier;
  void* payload;
  void (*payload_deleter)(void*);
  LITERT_ASSERT_OK(LrtGetOpaqueGoogleTensorOptionsData(
      options, &identifier, &payload, &payload_deleter));

  ASSERT_STREQ(identifier, "google_tensor");
  ASSERT_NE(payload, nullptr);

  payload_deleter(payload);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, FloatTruncationType) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  bool int64_to_int32_truncation;
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
      options, &int64_to_int32_truncation));
  ASSERT_FALSE(int64_to_int32_truncation);

  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsSetInt64ToInt32Truncation(options, true));
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
      options, &int64_to_int32_truncation));
  ASSERT_TRUE(int64_to_int32_truncation);

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_truncation;
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetInt64ToInt32Truncation(
      parsed, &parsed_truncation));
  EXPECT_TRUE(parsed_truncation);

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, Int64ToInt32Truncation) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  const char* output_dir;
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetOutputDir(options, &output_dir));
  ASSERT_STREQ(output_dir, "");

  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsSetOutputDir(options, "/tmp/test_dir"));
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetOutputDir(options, &output_dir));
  ASSERT_STREQ(output_dir, "/tmp/test_dir");

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  const char* parsed_output_dir;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetOutputDir(parsed, &parsed_output_dir));
  EXPECT_STREQ(parsed_output_dir, "/tmp/test_dir");

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, OutputDir) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  bool dump_op_timings;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetDumpOpTimings(options, &dump_op_timings));
  ASSERT_FALSE(dump_op_timings);

  LITERT_ASSERT_OK(LrtGoogleTensorOptionsSetDumpOpTimings(options, true));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetDumpOpTimings(options, &dump_op_timings));
  ASSERT_TRUE(dump_op_timings);

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  bool parsed_dump;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetDumpOpTimings(parsed, &parsed_dump));
  EXPECT_TRUE(parsed_dump);

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, DumpOpTimings) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  LrtGoogleTensorOptionsTruncationType truncation_type;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetFloatTruncationType(options, &truncation_type));
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeAuto);

  LITERT_ASSERT_OK(LrtGoogleTensorOptionsSetFloatTruncationType(
      options, kLiteRtGoogleTensorFloatTruncationTypeBfloat16));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetFloatTruncationType(options, &truncation_type));
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  LrtGoogleTensorOptionsTruncationType parsed_trunc;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetFloatTruncationType(parsed, &parsed_trunc));
  EXPECT_EQ(parsed_trunc, kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(GoogleTensorOptionsTest, CppApi) {
  auto options = GoogleTensorOptions::Create();
  ASSERT_TRUE(options);

  EXPECT_FALSE(options->GetInt64ToInt32Truncation());
  options->SetInt64ToInt32Truncation(true);
  EXPECT_TRUE(options->GetInt64ToInt32Truncation());

  EXPECT_EQ(options->GetFloatTruncationType(),
            kLiteRtGoogleTensorFloatTruncationTypeAuto);
  options->SetFloatTruncationType(
      kLiteRtGoogleTensorFloatTruncationTypeBfloat16);
  EXPECT_EQ(options->GetFloatTruncationType(),
            kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  EXPECT_EQ(options->GetOutputDir(), "");
  options->SetOutputDir("/tmp/test_dir");
  EXPECT_EQ(options->GetOutputDir(), "/tmp/test_dir");

  EXPECT_FALSE(options->GetDumpOpTimings());
  options->SetDumpOpTimings(true);
  EXPECT_TRUE(options->GetDumpOpTimings());

  EXPECT_EQ(options->GetOpFiltersProto(), "");
  options->SetOpFiltersProto("test_proto_string");
  EXPECT_EQ(options->GetOpFiltersProto(), "test_proto_string");
}

TEST(LrtGoogleTensorOptionsTest, OpFiltersProto) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  const char* op_filters_proto;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetOpFiltersProto(options, &op_filters_proto));
  ASSERT_STREQ(op_filters_proto, "");

  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsSetOpFiltersProto(options, "some_\"proto\"path\\"));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetOpFiltersProto(options, &op_filters_proto));
  ASSERT_STREQ(op_filters_proto, "some_\"proto\"path\\");

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  const char* parsed_op_filters_proto;
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetOpFiltersProto(
      parsed, &parsed_op_filters_proto));
  EXPECT_STREQ(parsed_op_filters_proto, "some_\"proto\"path\\");

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, Enable4BitCompilationCAPI) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  // Check default value
  bool enable_4bit = true;  // Initialize to non-default to ensure it's set
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetEnable4BitCompilation(options, &enable_4bit));
  EXPECT_FALSE(enable_4bit);

  // Set to true
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsSetEnable4BitCompilation(options, true));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetEnable4BitCompilation(options, &enable_4bit));
  EXPECT_TRUE(enable_4bit);

  // Set to false
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsSetEnable4BitCompilation(options, false));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetEnable4BitCompilation(options, &enable_4bit));
  EXPECT_FALSE(enable_4bit);

  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, Enable4BitCompilationCAPINullArgs) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  bool enable_4bit;
  EXPECT_EQ(LrtGoogleTensorOptionsSetEnable4BitCompilation(nullptr, true),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LrtGoogleTensorOptionsGetEnable4BitCompilation(nullptr, &enable_4bit),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LrtGoogleTensorOptionsGetEnable4BitCompilation(options, nullptr),
            kLiteRtStatusErrorInvalidArgument);

  LrtDestroyGoogleTensorOptions(options);
}

TEST(LrtGoogleTensorOptionsTest, TestingFlagsCAPI) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  // Default is empty
  std::vector<std::vector<std::string>> flags;
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetTestingFlags(options, &flags));
  EXPECT_TRUE(flags.empty());

  // Set multiple flags
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsSetTestingFlags(
      options, "flag1=val1,flag2=val2,flag3"));
  LITERT_ASSERT_OK(LrtGoogleTensorOptionsGetTestingFlags(options, &flags));
  ASSERT_EQ(flags.size(), 3);
  EXPECT_THAT(flags[0], testing::ElementsAre("flag1", "val1"));
  EXPECT_THAT(flags[1], testing::ElementsAre("flag2", "val2"));
  // Parser behavior: "flag3" becomes {"flag3", ""}
  EXPECT_THAT(flags[2], testing::ElementsAre("flag3", ""));

  // Round trip serialization
  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);
  std::vector<std::vector<std::string>> parsed_flags;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetTestingFlags(parsed, &parsed_flags));
  ASSERT_EQ(parsed_flags.size(), 3);
  EXPECT_THAT(parsed_flags[0], testing::ElementsAre("flag1", "val1"));
  EXPECT_THAT(parsed_flags[1], testing::ElementsAre("flag2", "val2"));
  EXPECT_THAT(parsed_flags[2], testing::ElementsAre("flag3", ""));

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

TEST(GoogleTensorOptionsTest, TestingFlagsCppAPI) {
  auto options = GoogleTensorOptions::Create();
  ASSERT_TRUE(options);

  // Set flags
  options->SetTestingFlags("enable_reference=true");
  auto flags = options->GetTestingFlags();

  ASSERT_EQ(flags.size(), 1);
  EXPECT_THAT(flags[0], testing::ElementsAre("enable_reference", "true"));

  options->SetTestingFlags("key=value");
  flags = options->GetTestingFlags();
  ASSERT_EQ(flags.size(), 1);
  EXPECT_THAT(flags[0], testing::ElementsAre("key", "value"));

  // No = here
  options->SetTestingFlags("another_flag");
  flags = options->GetTestingFlags();
  ASSERT_EQ(flags.size(), 1);
  EXPECT_THAT(flags[0], testing::ElementsAre("another_flag", ""));
}

TEST(LrtGoogleTensorOptionsTest, TestingFlagsEscaping) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  LITERT_ASSERT_OK(LrtGoogleTensorOptionsSetTestingFlags(
      options, "key=\"value\",path=C:\\foo"));

  LrtGoogleTensorOptions parsed;
  SerializeAndParse(options, &parsed);

  std::vector<std::vector<std::string>> parsed_flags;
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetTestingFlags(parsed, &parsed_flags));
  ASSERT_EQ(parsed_flags.size(), 2);
  EXPECT_THAT(parsed_flags[0], testing::ElementsAre("key", "\"value\""));
  EXPECT_THAT(parsed_flags[1], testing::ElementsAre("path", "C:\\foo"));

  LrtDestroyGoogleTensorOptions(parsed);
  LrtDestroyGoogleTensorOptions(options);
}

}  // namespace
}  // namespace litert::google_tensor
