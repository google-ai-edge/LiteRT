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

TEST(LrtGoogleTensorOptionsTest, PerformanceMode) {
  LrtGoogleTensorOptions options;
  LITERT_ASSERT_OK(LrtCreateGoogleTensorOptions(&options));

  LiteRtGoogleTensorOptionsPerformanceMode performance_mode;
  EXPECT_EQ(
      LrtGoogleTensorOptionsGetPerformanceMode(options, &performance_mode),
      kLiteRtStatusErrorNotFound);

  LITERT_ASSERT_OK(LrtGoogleTensorOptionsSetPerformanceMode(
      options, kLiteRtGoogleTensorOptionsPerformanceModeHighPerformance));
  LITERT_ASSERT_OK(
      LrtGoogleTensorOptionsGetPerformanceMode(options, &performance_mode));
  EXPECT_EQ(performance_mode,
            kLiteRtGoogleTensorOptionsPerformanceModeHighPerformance);

  LrtDestroyGoogleTensorOptions(options);
}

}  // namespace
}  // namespace litert::google_tensor
