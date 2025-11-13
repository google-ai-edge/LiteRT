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

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/test/matchers.h"

namespace litert::google_tensor {
namespace {

TEST(LiteRtGoogleTensorOptionsTest, CreateAndGet) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, FloatTruncationType) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  bool int64_to_int32_truncation;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
      options_data, &int64_to_int32_truncation));
  ASSERT_FALSE(int64_to_int32_truncation);

  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(options_data, true));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
      options_data, &int64_to_int32_truncation));
  ASSERT_TRUE(int64_to_int32_truncation);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, Int64ToInt32Truncation) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  const char* output_dir;
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsGetOutputDir(options_data, &output_dir));
  ASSERT_STREQ(output_dir, "");

  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetOutputDir(options_data, "/tmp/test_dir"));
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsGetOutputDir(options_data, &output_dir));
  ASSERT_STREQ(output_dir, "/tmp/test_dir");

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, OutputDir) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  bool dump_op_timings;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetDumpOpTimings(options_data,
                                                             &dump_op_timings));
  ASSERT_FALSE(dump_op_timings);

  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetDumpOpTimings(options_data, true));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetDumpOpTimings(options_data,
                                                             &dump_op_timings));
  ASSERT_TRUE(dump_op_timings);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, DumpOpTimings) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  LiteRtGoogleTensorOptionsTruncationType truncation_type;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetFloatTruncationType(
      options_data, &truncation_type));
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeAuto);

  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsSetFloatTruncationType(
      options_data, kLiteRtGoogleTensorFloatTruncationTypeBfloat16));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetFloatTruncationType(
      options_data, &truncation_type));
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  LiteRtDestroyOpaqueOptions(options);
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

TEST(LiteRtGoogleTensorOptionsTest, Hash) {
  LiteRtOpaqueOptions options1;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options1));
  LiteRtOpaqueOptions options2;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options2));

  uint64_t hash1, hash2;
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options1, &hash1));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options2, &hash2));
  EXPECT_EQ(hash1, hash2);

  LiteRtGoogleTensorOptions options_data1;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options1, &options_data1));
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(options_data1, true));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options1, &hash1));
  EXPECT_NE(hash1, hash2);

  LiteRtGoogleTensorOptions options_data2;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options2, &options_data2));
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(options_data2, true));
  LITERT_ASSERT_OK(LiteRtGetOpaqueOptionsHash(options2, &hash2));
  EXPECT_EQ(hash1, hash2);

  LiteRtDestroyOpaqueOptions(options1);
  LiteRtDestroyOpaqueOptions(options2);
}

TEST(LiteRtGoogleTensorOptionsTest, Enable4BitCompilationCAPI) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data1;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data1));

  // Check default value
  bool enable_4bit = true;  // Initialize to non-default to ensure it's set
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
      options_data1, &enable_4bit));
  EXPECT_FALSE(enable_4bit);

  // Set to true
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetEnable4BitCompilation(options_data1, true));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
      options_data1, &enable_4bit));
  EXPECT_TRUE(enable_4bit);

  // Set to false
  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetEnable4BitCompilation(options_data1, false));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetEnable4BitCompilation(
      options_data1, &enable_4bit));
  EXPECT_FALSE(enable_4bit);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, Enable4BitCompilationCAPINullArgs) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data1;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data1));

  bool enable_4bit;
  EXPECT_EQ(LiteRtGoogleTensorOptionsSetEnable4BitCompilation(nullptr, true),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtGoogleTensorOptionsGetEnable4BitCompilation(nullptr, &enable_4bit),
      kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(
      LiteRtGoogleTensorOptionsGetEnable4BitCompilation(options_data1, nullptr),
      kLiteRtStatusErrorInvalidArgument);

  LiteRtDestroyOpaqueOptions(options);
}

}  // namespace
}  // namespace litert::google_tensor
