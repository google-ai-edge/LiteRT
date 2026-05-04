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

#include "litert/tools/flags/vendors/google_tensor_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_google_tensor_options.h"
#include "litert/tools/flags/options_parser_registry.h"

// NOLINTBEGIN(*alien-types*)
// TODO: Move absl parse/unparse function to same file as enum types if
// it becomes an issue.

bool AbslParseFlag(absl::string_view text,
                   LrtGoogleTensorOptionsTruncationType* options,
                   std::string* error) {
  if (text == "auto") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeAuto;
    return true;
  }
  if (text == "no_truncation") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeNoTruncation;
    return true;
  }
  if (text == "bfloat16") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeBfloat16;
    return true;
  }
  if (text == "half") {
    *options = kLiteRtGoogleTensorFloatTruncationTypeHalf;
    return true;
  }
  *error = "Unknown truncation type";
  return false;
}

std::string AbslUnparseFlag(LrtGoogleTensorOptionsTruncationType options) {
  switch (options) {
    case kLiteRtGoogleTensorFloatTruncationTypeAuto:
      return "auto";
    case kLiteRtGoogleTensorFloatTruncationTypeNoTruncation:
      return "no_truncation";
    case kLiteRtGoogleTensorFloatTruncationTypeBfloat16:
      return "bfloat16";
    case kLiteRtGoogleTensorFloatTruncationTypeHalf:
      return "half";
  }
}

bool AbslParseFlag(absl::string_view text,
                   LrtGoogleTensorOptionsShardingIntensity* options,
                   std::string* error) {
  if (text == "minimal") {
    *options = kLiteRtGoogleTensorShardingIntensityMinimal;
    return true;
  }
  if (text == "moderate") {
    *options = kLiteRtGoogleTensorShardingIntensityModerate;
    return true;
  }
  if (text == "extensive") {
    *options = kLiteRtGoogleTensorShardingIntensityExtensive;
    return true;
  }
  if (text == "maximum") {
    *options = kLiteRtGoogleTensorShardingIntensityMaximum;
    return true;
  }
  *error = "Unknown sharding intensity";
  return false;
}

std::string AbslUnparseFlag(LrtGoogleTensorOptionsShardingIntensity options) {
  switch (options) {
    case kLiteRtGoogleTensorShardingIntensityMinimal:
      return "minimal";
    case kLiteRtGoogleTensorShardingIntensityModerate:
      return "moderate";
    case kLiteRtGoogleTensorShardingIntensityExtensive:
      return "extensive";
    case kLiteRtGoogleTensorShardingIntensityMaximum:
      return "maximum";
  }
}

ABSL_FLAG(LrtGoogleTensorOptionsTruncationType, google_tensor_truncation_type,
          kLiteRtGoogleTensorFloatTruncationTypeAuto,
          "Float truncation type for Google Tensor.");

ABSL_FLAG(bool, google_tensor_int64_to_int32, false,
          "Whether to truncate int64 to int32.");

ABSL_FLAG(bool, google_tensor_dump_op_timings, false,
          "Whether to dump op timings.");

ABSL_FLAG(bool, google_tensor_enable_large_model_support, false,
          "Whether to enable large model support.");

ABSL_FLAG(bool, google_tensor_enable_4bit_compilation, false,
          "Whether to enable 4bit compilation.");

ABSL_FLAG(LrtGoogleTensorOptionsShardingIntensity,
          google_tensor_sharding_intensity,
          kLiteRtGoogleTensorShardingIntensityMinimal,
          "Sharding intensity for Google Tensor.");

ABSL_FLAG(bool, google_tensor_enable_dynamic_range_quantization, false,
          "Whether to enable dynamic range quantization.");

ABSL_FLAG(std::string, google_tensor_testing_flags, "",
          "Testing flags for Google Tensor. Flag1=value1,Flag2=value2");

ABSL_FLAG(
    std::string, google_tensor_op_filters_proto, "",
    "A path to a file containing proto text formatted OpFilters for the Google "
    "Tensor plugin.");

ABSL_FLAG(
    litert::google_tensor::GoogleTensorOptions::PerformanceMode,
    google_tensor_performance_mode,
    litert::google_tensor::GoogleTensorOptions::PerformanceMode::kBalanced,
    "Performance mode for Google Tensor.");

namespace litert::google_tensor {

bool AbslParseFlag(::absl::string_view text,
                   GoogleTensorOptions::PerformanceMode* options,
                   ::std::string* error) {
  using PerformanceMode = GoogleTensorOptions::PerformanceMode;
  if (text == "extreme_power_saver") {
    *options = PerformanceMode::kExtremePowerSaver;
    return true;
  }
  if (text == "power_saver") {
    *options = PerformanceMode::kPowerSaver;
    return true;
  }
  if (text == "balanced") {
    *options = PerformanceMode::kBalanced;
    return true;
  }
  if (text == "high_performance") {
    *options = PerformanceMode::kHighPerformance;
    return true;
  }
  if (text == "sustained_performance") {
    *options = PerformanceMode::kSustainedPerformance;
    return true;
  }
  if (text == "burst") {
    *options = PerformanceMode::kBurst;
    return true;
  }
  *error = "Unknown performance mode";
  return false;
}

::std::string AbslUnparseFlag(GoogleTensorOptions::PerformanceMode options) {
  using PerformanceMode = GoogleTensorOptions::PerformanceMode;
  switch (options) {
    case PerformanceMode::kExtremePowerSaver:
      return "extreme_power_saver";
    case PerformanceMode::kPowerSaver:
      return "power_saver";
    case PerformanceMode::kBalanced:
      return "balanced";
    case PerformanceMode::kHighPerformance:
      return "high_performance";
    case PerformanceMode::kSustainedPerformance:
      return "sustained_performance";
    case PerformanceMode::kBurst:
      return "burst";
  }
}

Expected<void> UpdateGoogleTensorOptionsFromFlags(
    GoogleTensorOptions& options) {
  options.SetFloatTruncationType(
      absl::GetFlag(FLAGS_google_tensor_truncation_type));
  options.SetInt64ToInt32Truncation(
      absl::GetFlag(FLAGS_google_tensor_int64_to_int32));
  options.SetDumpOpTimings(absl::GetFlag(FLAGS_google_tensor_dump_op_timings));
  options.SetEnableLargeModelSupport(
      absl::GetFlag(FLAGS_google_tensor_enable_large_model_support));
  options.SetEnable4BitCompilation(
      absl::GetFlag(FLAGS_google_tensor_enable_4bit_compilation));
  options.SetShardingIntensity(
      absl::GetFlag(FLAGS_google_tensor_sharding_intensity));
  options.SetEnableDynamicRangeQuantization(
      absl::GetFlag(FLAGS_google_tensor_enable_dynamic_range_quantization));
  options.SetTestingFlags(absl::GetFlag(FLAGS_google_tensor_testing_flags));
  options.SetOpFiltersProto(
      absl::GetFlag(FLAGS_google_tensor_op_filters_proto));
  options.SetPerformanceMode(
      ::absl::GetFlag(::FLAGS_google_tensor_performance_mode));
  return {};
}

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  LITERT_ASSIGN_OR_RETURN(auto& google_tensor_opts,
                          options.GetGoogleTensorOptions());
  return UpdateGoogleTensorOptionsFromFlags(google_tensor_opts);
});

}  // namespace litert::google_tensor

// NOLINTEND(*alien-types*)
