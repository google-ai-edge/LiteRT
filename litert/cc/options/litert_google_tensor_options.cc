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

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////
// TODO(b/448037748): Add unit tests for this file.

namespace litert::google_tensor {

Expected<GoogleTensorOptions> GoogleTensorOptions::Create(
    OpaqueOptions& options) {
  const auto id = options.GetIdentifier();
  if (!id || *id != Discriminator()) {
    return Error(kLiteRtStatusErrorInvalidArgument);
  }
  return GoogleTensorOptions(options.Get(), OwnHandle::kNo);
}

Expected<GoogleTensorOptions> GoogleTensorOptions::Create() {
  LiteRtOpaqueOptions options;
  LITERT_RETURN_IF_ERROR(LiteRtGoogleTensorOptionsCreate(&options));
  return GoogleTensorOptions(options, OwnHandle::kYes);
}

void GoogleTensorOptions::SetFloatTruncationType(
    LiteRtGoogleTensorOptionsTruncationType truncation_type) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetFloatTruncationType, Data(),
                     truncation_type);
}

LiteRtGoogleTensorOptionsTruncationType
GoogleTensorOptions::GetFloatTruncationType() const {
  LiteRtGoogleTensorOptions options_data = Data();
  LiteRtGoogleTensorOptionsTruncationType truncation_type;
  internal::AssertOk(LiteRtGoogleTensorOptionsGetFloatTruncationType,
                     options_data, &truncation_type);
  return truncation_type;
}

void GoogleTensorOptions::SetInt64ToInt32Truncation(
    bool int64_to_int32_truncation) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation, Data(),
                     int64_to_int32_truncation);
}

bool GoogleTensorOptions::GetInt64ToInt32Truncation() const {
  LiteRtGoogleTensorOptions options_data = Data();
  bool int64_to_int32_truncation;
  internal::AssertOk(LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation,
                     options_data, &int64_to_int32_truncation);
  return int64_to_int32_truncation;
}

void GoogleTensorOptions::SetOutputDir(absl::string_view output_dir) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetOutputDir, Data(),
                     output_dir.data());
}

absl::string_view GoogleTensorOptions::GetOutputDir() const {
  LiteRtGoogleTensorOptions options_data = Data();
  const char* output_dir;
  internal::AssertOk(LiteRtGoogleTensorOptionsGetOutputDir, options_data,
                     &output_dir);
  return absl::string_view(output_dir);
}

void GoogleTensorOptions::SetDumpOpTimings(bool dump_op_timings) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetDumpOpTimings, Data(),
                     dump_op_timings);
}

bool GoogleTensorOptions::GetDumpOpTimings() const {
  LiteRtGoogleTensorOptions options_data = Data();
  bool dump_op_timings;
  LiteRtGoogleTensorOptionsGetDumpOpTimings(options_data, &dump_op_timings);
  return dump_op_timings;
}

void GoogleTensorOptions::SetEnableLargeModelSupport(
    bool enable_large_model_support) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetEnableLargeModelSupport,
                     Data(), enable_large_model_support);
}

bool GoogleTensorOptions::GetEnableLargeModelSupport() const {
  LiteRtGoogleTensorOptions options_data = Data();
  bool enable_large_model_support;
  LiteRtGoogleTensorOptionsGetEnableLargeModelSupport(
      options_data, &enable_large_model_support);
  return enable_large_model_support;
}

void GoogleTensorOptions::SetEnable4BitCompilation(
    bool enable_4bit_compilation) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetEnable4BitCompilation, Data(),
                     enable_4bit_compilation);
}

bool GoogleTensorOptions::GetEnable4BitCompilation() const {
  LiteRtGoogleTensorOptions options_data = Data();
  bool enable_4bit_compilation;
  LiteRtGoogleTensorOptionsGetEnable4BitCompilation(options_data,
                                                    &enable_4bit_compilation);
  return enable_4bit_compilation;
}

void GoogleTensorOptions::SetShardingIntensity(
    LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetShardingIntensity, Data(),
                     sharding_intensity);
}

LiteRtGoogleTensorOptionsShardingIntensity
GoogleTensorOptions::GetShardingIntensity() const {
  LiteRtGoogleTensorOptions options_data = Data();
  LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity;
  LiteRtGoogleTensorOptionsGetShardingIntensity(options_data,
                                                &sharding_intensity);
  return sharding_intensity;
}

void GoogleTensorOptions::SetTestingFlags(const std::string& testing_flags) {
  internal::AssertOk(LiteRtGoogleTensorOptionsSetTestingFlags, Data(),
                     testing_flags);
}

std::vector<std::vector<std::string>> GoogleTensorOptions::GetTestingFlags()
    const {
  LiteRtGoogleTensorOptions options_data = Data();
  std::vector<std::vector<std::string>> testing_flags;
  LiteRtGoogleTensorOptionsGetTestingFlags(options_data, &testing_flags);
  return testing_flags;
}

LiteRtGoogleTensorOptions GoogleTensorOptions::Data() const {
  LiteRtGoogleTensorOptions options_data;
  internal::AssertOk(LiteRtGoogleTensorOptionsGet, Get(), &options_data);
  return options_data;
}

}  // namespace litert::google_tensor
