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

#include <memory>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_handle.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

struct LiteRtGoogleTensorOptionsT {
  LiteRtGoogleTensorOptionsTruncationType float_truncation_type =
      kLiteRtGoogleTensorFloatTruncationTypeUnspecified;
  bool int64_to_int32_truncation = false;
  std::string output_dir = "";
  bool dump_op_timings = false;
  bool enable_large_model_support = false;
  LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity =
      kLiteRtGoogleTensorShardingIntensityMinimal;
};

LiteRtStatus LiteRtGoogleTensorOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtGoogleTensorOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGoogleTensorOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtGoogleTensorOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtGoogleTensorOptionsGetIdentifier() { return "google_tensor"; }

LiteRtStatus LiteRtGoogleTensorOptionsGet(
    LiteRtOpaqueOptions options, LiteRtGoogleTensorOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtGoogleTensorOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtGoogleTensorOptions>(payload);

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// float_truncation_type -------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType truncation_type) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->float_truncation_type = truncation_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetFloatTruncationType(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsTruncationType* truncation_type) {
  if (options == nullptr || truncation_type == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *truncation_type = options->float_truncation_type;
  return kLiteRtStatusOk;
}

// int64_to_int32_truncation ---------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool int64_to_int32_truncation) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->int64_to_int32_truncation = int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetInt64ToInt32Truncation(
    LiteRtGoogleTensorOptions options, bool* int64_to_int32_truncation) {
  if (options == nullptr || int64_to_int32_truncation == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *int64_to_int32_truncation = options->int64_to_int32_truncation;
  return kLiteRtStatusOk;
}

// output_dir ------------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetOutputDir(
    LiteRtGoogleTensorOptions options, const char* output_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->output_dir = output_dir;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetOutputDir(
    LiteRtGoogleTensorOptions options, const char** output_dir) {
  if (options == nullptr || output_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *output_dir = options->output_dir.c_str();
  return kLiteRtStatusOk;
}

// dump_op_timings -------------------------------------------------------------

LiteRtStatus LiteRtGoogleTensorOptionsSetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool dump_op_timings) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dump_op_timings = dump_op_timings;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetDumpOpTimings(
    LiteRtGoogleTensorOptions options, bool* dump_op_timings) {
  if (options == nullptr || dump_op_timings == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dump_op_timings = options->dump_op_timings;
  return kLiteRtStatusOk;
}

// enable_large_model_support --------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool enable_large_model_support) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->enable_large_model_support = enable_large_model_support;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetEnableLargeModelSupport(
    LiteRtGoogleTensorOptions options, bool* enable_large_model_support) {
  if (options == nullptr || enable_large_model_support == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *enable_large_model_support = options->enable_large_model_support;
  return kLiteRtStatusOk;
}

// sharding intensity ----------------------------------------------------------
LiteRtStatus LiteRtGoogleTensorOptionsSetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity sharding_intensity) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->sharding_intensity = sharding_intensity;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGoogleTensorOptionsGetShardingIntensity(
    LiteRtGoogleTensorOptions options,
    LiteRtGoogleTensorOptionsShardingIntensity* sharding_intensity) {
  if (options == nullptr || sharding_intensity == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sharding_intensity = options->sharding_intensity;
  return kLiteRtStatusOk;
}

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert::google_tensor {

const char* GoogleTensorOptions::Discriminator() {
  return LiteRtGoogleTensorOptionsGetIdentifier();
}

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

LiteRtGoogleTensorOptions GoogleTensorOptions::Data() const {
  LiteRtGoogleTensorOptions options_data;
  internal::AssertOk(LiteRtGoogleTensorOptionsGet, Get(), &options_data);
  return options_data;
}

}  // namespace litert::google_tensor
