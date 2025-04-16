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

#include <gtest/gtest.h>
#include "litert/c/litert_opaque_options.h"
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
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeUnspecified);

  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsSetFloatTruncationType(
      options_data, kLiteRtGoogleTensorFloatTruncationTypeBfloat16));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetFloatTruncationType(
      options_data, &truncation_type));
  ASSERT_EQ(truncation_type, kLiteRtGoogleTensorFloatTruncationTypeBfloat16);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(LiteRtGoogleTensorOptionsTest, EnableReference) {
  LiteRtOpaqueOptions options;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsCreate(&options));

  LiteRtGoogleTensorOptions options_data;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGet(options, &options_data));

  bool enable_reference;
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetEnableReference(
      options_data, &enable_reference));
  ASSERT_FALSE(enable_reference);

  LITERT_ASSERT_OK(
      LiteRtGoogleTensorOptionsSetEnableReference(options_data, true));
  LITERT_ASSERT_OK(LiteRtGoogleTensorOptionsGetEnableReference(
      options_data, &enable_reference));
  ASSERT_TRUE(enable_reference);

  LiteRtDestroyOpaqueOptions(options);
}

TEST(GoogleTensorOptionsTest, CppApi) {
  auto options = GoogleTensorOptions::Create();
  ASSERT_TRUE(options);

  EXPECT_FALSE(options->GetInt64ToInt32Truncation());
  options->SetInt64ToInt32Truncation(true);
  EXPECT_TRUE(options->GetInt64ToInt32Truncation());

  EXPECT_EQ(options->GetFloatTruncationType(),
            kLiteRtGoogleTensorFloatTruncationTypeUnspecified);
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

  EXPECT_FALSE(options->GetEnableReference());
  options->SetEnableReference(true);
  EXPECT_TRUE(options->GetEnableReference());
}

}  // namespace
}  // namespace litert::google_tensor
