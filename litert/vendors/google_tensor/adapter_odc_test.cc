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

#include "litert/vendors/google_tensor/adapter_odc.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/matchers.h"
#include "litert/vendors/google_tensor/compiler/google_tensor_options.pb.h"
#include "litert/vendors/google_tensor/edgetpu_compiler_options.pb.h"

namespace litert::google_tensor {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptions;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptionsShardingIntensity;
using ::third_party::odml::litert::litert::vendors::google_tensor::compiler::
    GoogleTensorOptionsTruncationType;

TEST(CreateEdgeTpuCompilerOptionsTest, EmptyPayloadYieldsDefaults) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options,
      CreateEdgeTpuCompilerOptions(/*options=*/nullptr, /*options_size=*/0));
  EXPECT_EQ(options.float_truncation_type(),
            EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_AUTO);
  EXPECT_FALSE(options.int64_to_int32_truncation());
  EXPECT_EQ(options.sharding_intensity(),
            EdgeTpuCompilerOptions::SHARDING_INTENSITY_UNSPECIFIED);
  EXPECT_FALSE(options.enable_four_bit_compilation());
  EXPECT_FALSE(options.enable_dynamic_range_quantization());
  EXPECT_THAT(options.litert_version(), IsEmpty());
  EXPECT_THAT(options.internal_options(), IsEmpty());
}

TEST(CreateEdgeTpuCompilerOptionsTest, ZeroSizeNonNullPayloadYieldsDefaults) {
  const std::string empty;
  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options,
      CreateEdgeTpuCompilerOptions(empty.data(), empty.size()));
  EXPECT_EQ(options.SerializeAsString(),
            EdgeTpuCompilerOptions().SerializeAsString());
}

TEST(CreateEdgeTpuCompilerOptionsTest, NullPointerWithNonZeroSizeFails) {
  Expected<EdgeTpuCompilerOptions> result =
      CreateEdgeTpuCompilerOptions(/*options=*/nullptr, /*options_size=*/4);
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(CreateEdgeTpuCompilerOptionsTest, UnparsablePayloadFails) {
  const std::string garbage = "\xff\xff\xff not a proto";
  Expected<EdgeTpuCompilerOptions> result =
      CreateEdgeTpuCompilerOptions(garbage.data(), garbage.size());
  ASSERT_FALSE(result.HasValue());
  EXPECT_EQ(result.Error().Status(), kLiteRtStatusErrorInvalidArgument);
}

TEST(CreateEdgeTpuCompilerOptionsTest, CopiesSupportedFields) {
  GoogleTensorOptions google_tensor_options;
  google_tensor_options.set_float_truncation_type(
      GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_BFLOAT16);
  google_tensor_options.set_int64_to_int32_truncation(true);
  google_tensor_options.set_sharding_intensity(
      GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_EXTENSIVE);
  google_tensor_options.set_enable_four_bit_compilation(true);
  google_tensor_options.set_enable_dynamic_range_quantization(true);
  google_tensor_options.mutable_compiler_config()->set_litert_version("1.2.3");
  const std::string serialized = google_tensor_options.SerializeAsString();

  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options,
      CreateEdgeTpuCompilerOptions(serialized.data(), serialized.size()));

  EXPECT_EQ(options.float_truncation_type(),
            EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_BFLOAT16);
  EXPECT_TRUE(options.int64_to_int32_truncation());
  EXPECT_EQ(options.sharding_intensity(),
            EdgeTpuCompilerOptions::SHARDING_INTENSITY_EXTENSIVE);
  EXPECT_TRUE(options.enable_four_bit_compilation());
  EXPECT_TRUE(options.enable_dynamic_range_quantization());
  EXPECT_EQ(options.litert_version(), "1.2.3");
  EXPECT_THAT(options.internal_options(), IsEmpty());
}

TEST(CreateEdgeTpuCompilerOptionsTest, PreservesLiteRtVersionWireTagTwo) {
  GoogleTensorOptions google_tensor_options;
  google_tensor_options.mutable_compiler_config()->set_litert_version("2.1.0");
  const std::string serialized = google_tensor_options.SerializeAsString();

  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options,
      CreateEdgeTpuCompilerOptions(serialized.data(), serialized.size()));
  const std::string wire_bytes = options.SerializeAsString();

  // Tag 2 with length-delimited wire type (2) is (2 << 3) | 2 = 0x12, followed
  // by length 5 ("2.1.0"). Older vendor binaries rely on this exact wire tag.
  EXPECT_EQ(wire_bytes, "\x12\x05" "2.1.0");
}

TEST(CreateEdgeTpuCompilerOptionsTest, MapsEveryTruncationType) {
  struct TruncationCase {
    GoogleTensorOptionsTruncationType input;
    EdgeTpuCompilerOptions::FloatTruncationType expected;
  };
  const TruncationCase cases[] = {
      {GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_AUTO,
       EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_AUTO},
      {GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_NO_TRUNCATION,
       EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_NO_TRUNCATION},
      {GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_BFLOAT16,
       EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_BFLOAT16},
      {GoogleTensorOptionsTruncationType::FLOAT_TRUNCATION_TYPE_HALF,
       EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_HALF},
      {static_cast<GoogleTensorOptionsTruncationType>(99),
       EdgeTpuCompilerOptions::FLOAT_TRUNCATION_TYPE_AUTO},
  };
  for (const TruncationCase& test_case : cases) {
    SCOPED_TRACE(test_case.input);
    GoogleTensorOptions google_tensor_options;
    google_tensor_options.set_float_truncation_type(test_case.input);
    const std::string serialized = google_tensor_options.SerializeAsString();
    LITERT_ASSERT_OK_AND_ASSIGN(
        EdgeTpuCompilerOptions options,
        CreateEdgeTpuCompilerOptions(serialized.data(), serialized.size()));
    EXPECT_EQ(options.float_truncation_type(), test_case.expected);
  }
}

TEST(CreateEdgeTpuCompilerOptionsTest, MapsEveryShardingIntensity) {
  struct ShardingCase {
    GoogleTensorOptionsShardingIntensity input;
    EdgeTpuCompilerOptions::ShardingIntensity expected;
  };
  const ShardingCase cases[] = {
      {GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_UNSPECIFIED,
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_UNSPECIFIED},
      {GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MINIMAL,
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_MINIMAL},
      {GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MODERATE,
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_MODERATE},
      {GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_EXTENSIVE,
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_EXTENSIVE},
      {GoogleTensorOptionsShardingIntensity::SHARDING_INTENSITY_MAXIMUM,
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_MAXIMUM},
      {static_cast<GoogleTensorOptionsShardingIntensity>(99),
       EdgeTpuCompilerOptions::SHARDING_INTENSITY_UNSPECIFIED},
  };
  for (const ShardingCase& test_case : cases) {
    SCOPED_TRACE(test_case.input);
    GoogleTensorOptions google_tensor_options;
    google_tensor_options.set_sharding_intensity(test_case.input);
    const std::string serialized = google_tensor_options.SerializeAsString();
    LITERT_ASSERT_OK_AND_ASSIGN(
        EdgeTpuCompilerOptions options,
        CreateEdgeTpuCompilerOptions(serialized.data(), serialized.size()));
    EXPECT_EQ(options.sharding_intensity(), test_case.expected);
  }
}

TEST(CreateEdgeTpuCompilerOptionsTest,
     LargeModelSupportAndHostOnlyFieldsAreNotForwarded) {
  InterceptLogs intercepted_logs;
  GoogleTensorOptions google_tensor_options;
  google_tensor_options.set_enable_large_model_support(true);
  google_tensor_options.set_output_dir("/tmp/out");
  google_tensor_options.set_dump_op_timings(true);
  google_tensor_options.set_op_filters_proto("/tmp/filters.textproto");
  google_tensor_options.set_extra_options_path("/tmp/extra.pb");
  google_tensor_options.set_extra_options("qos_class: REALTIME");
  const std::string serialized = google_tensor_options.SerializeAsString();

  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options,
      CreateEdgeTpuCompilerOptions(serialized.data(), serialized.size()));
  EXPECT_EQ(options.SerializeAsString(),
            EdgeTpuCompilerOptions().SerializeAsString());
  std::ostringstream warning_logs;
  warning_logs << intercepted_logs;
  EXPECT_THAT(warning_logs.str(),
              HasSubstr("enable_large_model_support is not supported"));

  InterceptLogs::ClearLogs();
  google_tensor_options.set_enable_large_model_support(false);
  const std::string serialized_no_large_model =
      google_tensor_options.SerializeAsString();
  LITERT_ASSERT_OK_AND_ASSIGN(
      EdgeTpuCompilerOptions options_no_large_model,
      CreateEdgeTpuCompilerOptions(serialized_no_large_model.data(),
                                   serialized_no_large_model.size()));
  EXPECT_EQ(options_no_large_model.SerializeAsString(),
            EdgeTpuCompilerOptions().SerializeAsString());
  std::ostringstream no_warning_logs;
  no_warning_logs << intercepted_logs;
  EXPECT_THAT(no_warning_logs.str(),
              Not(HasSubstr("enable_large_model_support is not supported")));
}

}  // namespace
}  // namespace litert::google_tensor
