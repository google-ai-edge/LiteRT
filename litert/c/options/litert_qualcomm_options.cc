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
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/c/options/litert_qualcomm_options.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"

struct LiteRtQualcommOptionsT {
  LiteRtQualcommOptionsLogLevel log_level = kLiteRtQualcommLogLevelInfo;
  LiteRtQualcommOptionsProfiling profiling = kLiteRtQualcommProfilingOff;
  bool use_htp_preference = false;
  bool use_qint16_as_quint16 = false;
  bool enable_weight_sharing = false;
  LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode =
      kLiteRtQualcommHtpPerformanceModeDefault;
  std::vector<std::int32_t> dump_tensor_ids;
  std::string qnn_json_path;
};

LiteRtStatus LiteRtQualcommOptionsCreate(LiteRtOpaqueOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto options_data = std::make_unique<LiteRtQualcommOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtQualcommOptionsGetIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtQualcommOptions>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

const char* LiteRtQualcommOptionsGetIdentifier() { return "qualcomm"; }

LiteRtStatus LiteRtQualcommOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtQualcommOptions* options_data) {
  if (options_data == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const char* identifier;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  if (absl::NullSafeStringView(identifier) !=
      LiteRtQualcommOptionsGetIdentifier()) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  void* payload;
  LITERT_RETURN_IF_ERROR(LiteRtGetOpaqueOptionsData(options, &payload));
  *options_data = reinterpret_cast<LiteRtQualcommOptions>(payload);

  return kLiteRtStatusOk;
}

// GLOBAL OPTIONS //////////////////////////////////////////////////////////////

// log_level -------------------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->log_level = log_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel* log_level) {
  if (log_level == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *log_level = options->log_level;

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

LiteRtStatus LiteRtQualcommOptionsSetUseHtpPreference(
    LiteRtQualcommOptions options, bool use_htp_preference) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_htp_preference = use_htp_preference;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetUseHtpPreference(
    LiteRtQualcommOptions options, bool* use_htp_preference) {
  if (use_htp_preference == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_htp_preference = options->use_htp_preference;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetUseQint16AsQuint16(
    LiteRtQualcommOptions options, bool use_qint16_as_quint16) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_qint16_as_quint16 = use_qint16_as_quint16;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetUseQint16AsQuint16(
    LiteRtQualcommOptions options, bool* use_qint16_as_quint16) {
  if (use_qint16_as_quint16 == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_qint16_as_quint16 = options->use_qint16_as_quint16;

  return kLiteRtStatusOk;
}

// enable_weight_sharing -------------------------------------------------------

LiteRtStatus LiteRtQualcommOptionsSetEnableWeightSharing(
    LiteRtQualcommOptions options, bool enable_weight_sharing) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->enable_weight_sharing = enable_weight_sharing;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetEnableWeightSharing(
    LiteRtQualcommOptions options, bool* enable_weight_sharing) {
  if (enable_weight_sharing == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *enable_weight_sharing = options->enable_weight_sharing;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetDumpTensorIds(
    LiteRtQualcommOptions options, const std::int32_t* ids,
    std::uint32_t number_of_ids) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (size_t i = 0; i < number_of_ids; i++) {
    options->dump_tensor_ids.emplace_back(ids[i]);
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetDumpTensorIds(
    LiteRtQualcommOptions options, std::int32_t** ids,
    std::uint32_t* number_of_ids) {
  if (ids == nullptr || number_of_ids == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ids = options->dump_tensor_ids.data();
  *number_of_ids = options->dump_tensor_ids.size();
  return kLiteRtStatusOk;
}

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

LiteRtStatus LiteRtQualcommOptionsSetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->htp_performance_mode = htp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode* htp_performance_mode) {
  if (options == nullptr || htp_performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *htp_performance_mode = options->htp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling profiling) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->profiling = profiling;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling* profiling) {
  if (options == nullptr || profiling == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *profiling = options->profiling;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetQnnJsonPath(LiteRtQualcommOptions options,
                                                 const char* qnn_json_path) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->qnn_json_path = qnn_json_path;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetQnnJsonPath(LiteRtQualcommOptions options,
                                                 const char** qnn_json_path) {
  if (options == nullptr || qnn_json_path == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *qnn_json_path = options->qnn_json_path.data();

  return kLiteRtStatusOk;
}
