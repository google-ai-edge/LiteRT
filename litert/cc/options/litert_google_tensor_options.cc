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

#include "litert/cc/options/litert_google_tensor_options.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_macros.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
// TODO(b/448037748): Add unit tests for this file.

namespace litert::google_tensor {

GoogleTensorOptions::GoogleTensorOptions(LrtGoogleTensorOptions options)
    : options_(options) {}

Expected<GoogleTensorOptions> GoogleTensorOptions::Create() {
  LrtGoogleTensorOptions options;
  LITERT_RETURN_IF_ERROR(LrtCreateGoogleTensorOptions(&options));
  return GoogleTensorOptions(options);
}

LiteRtStatus GoogleTensorOptions::GetOpaqueOptionsData(
    const char** identifier, void** payload,
    void (**payload_deleter)(void*)) const {
  return LrtGetOpaqueGoogleTensorOptionsData(Get(), identifier, payload,
                                             payload_deleter);
}

void GoogleTensorOptions::SetFloatTruncationType(
    LrtGoogleTensorOptionsTruncationType truncation_type) {
  internal::AssertOk(LrtGoogleTensorOptionsSetFloatTruncationType, Get(),
                     truncation_type);
}

LrtGoogleTensorOptionsTruncationType
GoogleTensorOptions::GetFloatTruncationType() const {
  LrtGoogleTensorOptions options_data = Get();
  LrtGoogleTensorOptionsTruncationType truncation_type;
  internal::AssertOk(LrtGoogleTensorOptionsGetFloatTruncationType, options_data,
                     &truncation_type);
  return truncation_type;
}

void GoogleTensorOptions::SetInt64ToInt32Truncation(
    bool int64_to_int32_truncation) {
  internal::AssertOk(LrtGoogleTensorOptionsSetInt64ToInt32Truncation, Get(),
                     int64_to_int32_truncation);
}

bool GoogleTensorOptions::GetInt64ToInt32Truncation() const {
  LrtGoogleTensorOptions options_data = Get();
  bool int64_to_int32_truncation;
  internal::AssertOk(LrtGoogleTensorOptionsGetInt64ToInt32Truncation,
                     options_data, &int64_to_int32_truncation);
  return int64_to_int32_truncation;
}

void GoogleTensorOptions::SetOutputDir(absl::string_view output_dir) {
  internal::AssertOk(LrtGoogleTensorOptionsSetOutputDir, Get(),
                     output_dir.data());
}

absl::string_view GoogleTensorOptions::GetOutputDir() const {
  LrtGoogleTensorOptions options_data = Get();
  const char* output_dir;
  internal::AssertOk(LrtGoogleTensorOptionsGetOutputDir, options_data,
                     &output_dir);
  return absl::string_view(output_dir);
}

void GoogleTensorOptions::SetDumpOpTimings(bool dump_op_timings) {
  internal::AssertOk(LrtGoogleTensorOptionsSetDumpOpTimings, Get(),
                     dump_op_timings);
}

bool GoogleTensorOptions::GetDumpOpTimings() const {
  LrtGoogleTensorOptions options_data = Get();
  bool dump_op_timings;
  LrtGoogleTensorOptionsGetDumpOpTimings(options_data, &dump_op_timings);
  return dump_op_timings;
}

void GoogleTensorOptions::SetEnableLargeModelSupport(
    bool enable_large_model_support) {
  internal::AssertOk(LrtGoogleTensorOptionsSetEnableLargeModelSupport, Get(),
                     enable_large_model_support);
}

bool GoogleTensorOptions::GetEnableLargeModelSupport() const {
  LrtGoogleTensorOptions options_data = Get();
  bool enable_large_model_support;
  LrtGoogleTensorOptionsGetEnableLargeModelSupport(options_data,
                                                   &enable_large_model_support);
  return enable_large_model_support;
}

void GoogleTensorOptions::SetEnable4BitCompilation(
    bool enable_4bit_compilation) {
  internal::AssertOk(LrtGoogleTensorOptionsSetEnable4BitCompilation, Get(),
                     enable_4bit_compilation);
}

bool GoogleTensorOptions::GetEnable4BitCompilation() const {
  LrtGoogleTensorOptions options_data = Get();
  bool enable_4bit_compilation;
  LrtGoogleTensorOptionsGetEnable4BitCompilation(options_data,
                                                 &enable_4bit_compilation);
  return enable_4bit_compilation;
}

void GoogleTensorOptions::SetShardingIntensity(
    LrtGoogleTensorOptionsShardingIntensity sharding_intensity) {
  internal::AssertOk(LrtGoogleTensorOptionsSetShardingIntensity, Get(),
                     sharding_intensity);
}

LrtGoogleTensorOptionsShardingIntensity
GoogleTensorOptions::GetShardingIntensity() const {
  LrtGoogleTensorOptions options_data = Get();
  LrtGoogleTensorOptionsShardingIntensity sharding_intensity;
  LrtGoogleTensorOptionsGetShardingIntensity(options_data, &sharding_intensity);
  return sharding_intensity;
}

void GoogleTensorOptions::SetEnableDynamicRangeQuantization(
    bool enable_dynamic_range_quantization) {
  internal::AssertOk(LrtGoogleTensorOptionsSetEnableDynamicRangeQuantization,
                     Get(), enable_dynamic_range_quantization);
}

bool GoogleTensorOptions::GetEnableDynamicRangeQuantization() const {
  LrtGoogleTensorOptions options_data = Get();
  bool enable_dynamic_range_quantization;
  LrtGoogleTensorOptionsGetEnableDynamicRangeQuantization(
      options_data, &enable_dynamic_range_quantization);
  return enable_dynamic_range_quantization;
}

void GoogleTensorOptions::SetTestingFlags(const std::string& testing_flags) {
  internal::AssertOk(LrtGoogleTensorOptionsSetTestingFlags, Get(),
                     testing_flags);
}

std::vector<std::vector<std::string>> GoogleTensorOptions::GetTestingFlags()
    const {
  LrtGoogleTensorOptions options_data = Get();
  std::vector<std::vector<std::string>> testing_flags;
  LrtGoogleTensorOptionsGetTestingFlags(options_data, &testing_flags);
  return testing_flags;
}

void GoogleTensorOptions::SetOpFiltersProto(
    absl::string_view op_filters_proto) {
  internal::AssertOk(LrtGoogleTensorOptionsSetOpFiltersProto, Get(),
                     op_filters_proto.data());
}

absl::string_view GoogleTensorOptions::GetOpFiltersProto() const {
  LrtGoogleTensorOptions options_data = Get();
  const char* op_filters_proto;
  internal::AssertOk(LrtGoogleTensorOptionsGetOpFiltersProto, options_data,
                     &op_filters_proto);
  return absl::string_view(op_filters_proto);
}

}  // namespace litert::google_tensor
