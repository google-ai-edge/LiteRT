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
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/cache/hash_util.h"

struct LiteRtQualcommOptionsT {
  LiteRtQualcommOptionsLogLevel log_level = kLiteRtQualcommLogLevelInfo;
  LiteRtQualcommOptionsProfiling profiling = kLiteRtQualcommProfilingOff;
  bool use_htp_preference = false;
  bool use_qint16_as_quint16 = false;
  LiteRtQualcommOptionsBackend qnn_backend = kLiteRtQualcommBackendHtp;
  bool enable_weight_sharing = false;
  bool use_conv_hmx = true;
  bool use_fold_relu = true;
  LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode =
      kLiteRtQualcommHtpPerformanceModeDefault;
  LiteRtQualcommOptionsDspPerformanceMode dsp_performance_mode =
      kLiteRtQualcommDspPerformanceModeDefault;
  std::vector<std::int32_t> dump_tensor_ids;
  std::string ir_json_dir;
  std::string dlc_dir;
  std::uint32_t vtcm_size = 0;
  std::uint32_t num_hvx_threads = 0;
  LiteRtQualcommOptionsOptimizationLevel optimization_level =
      kHtpOptimizeForInferenceO3;
  LiteRtQualcommOptionsGraphPriority graph_priority =
      kLiteRTQualcommGraphPriorityDefault;
  std::string saver_output_dir;
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

  auto qti_hash = [](const void* payload) -> uint64_t {
    const LiteRtQualcommOptionsT* options =
        reinterpret_cast<const LiteRtQualcommOptionsT*>(payload);
    uint64_t ans = 0;
    litert::HashCombine(
        ans, options->log_level, options->profiling,
        options->use_htp_preference, options->use_qint16_as_quint16,
        options->qnn_backend, options->enable_weight_sharing,
        options->htp_performance_mode, options->dsp_performance_mode,
        options->ir_json_dir, options->dlc_dir, options->vtcm_size,
        options->num_hvx_threads, options->optimization_level);
    return ans;
  };
  LITERT_RETURN_IF_ERROR(LiteRtSetOpaqueOptionsHash(*options, qti_hash));

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

// Profiling -------------------------------------------------------------------

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

// Saver -------------------------------------------------------------------
LiteRtStatus LiteRtQualcommOptionsSetSaverOutputDir(
    LiteRtQualcommOptions options, const char* saver_output_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->saver_output_dir = saver_output_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetSaverOutputDir(
    LiteRtQualcommOptions options, const char** saver_output_dir) {
  if (options == nullptr || saver_output_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *saver_output_dir = options->saver_output_dir.c_str();

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
    size_t number_of_ids) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  for (size_t i = 0; i < number_of_ids; i++) {
    options->dump_tensor_ids.emplace_back(ids[i]);
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetDumpTensorIds(
    LiteRtQualcommOptions options, const std::int32_t** ids,
    size_t* number_of_ids) {
  if (ids == nullptr || number_of_ids == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ids = options->dump_tensor_ids.data();
  *number_of_ids = options->dump_tensor_ids.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetUseConvHMX(LiteRtQualcommOptions options,
                                                bool use_conv_hmx) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_conv_hmx = use_conv_hmx;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetUseConvHMX(LiteRtQualcommOptions options,
                                                bool* use_conv_hmx) {
  if (use_conv_hmx == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_conv_hmx = options->use_conv_hmx;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetUseFoldReLU(LiteRtQualcommOptions options,
                                                 bool use_fold_relu) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_fold_relu = use_fold_relu;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetUseFoldReLU(LiteRtQualcommOptions options,
                                                 bool* use_fold_relu) {
  if (use_fold_relu == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_fold_relu = options->use_fold_relu;

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

LiteRtStatus LiteRtQualcommOptionsSetDspPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsDspPerformanceMode dsp_performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->dsp_performance_mode = dsp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetDspPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsDspPerformanceMode* dsp_performance_mode) {
  if (options == nullptr || dsp_performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *dsp_performance_mode = options->dsp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetIrJsonDir(LiteRtQualcommOptions options,
                                               const char* ir_json_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->ir_json_dir = ir_json_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetIrJsonDir(LiteRtQualcommOptions options,
                                               const char** ir_json_dir) {
  if (options == nullptr || ir_json_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *ir_json_dir = options->ir_json_dir.c_str();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetDlcDir(LiteRtQualcommOptions options,
                                            const char* dlc_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->dlc_dir = dlc_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetDlcDir(LiteRtQualcommOptions options,
                                            const char** dlc_dir) {
  if (options == nullptr || dlc_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *dlc_dir = options->dlc_dir.c_str();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetVtcmSize(LiteRtQualcommOptions options,
                                              std::uint32_t vtcm_size) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->vtcm_size = vtcm_size;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetVtcmSize(LiteRtQualcommOptions options,
                                              std::uint32_t* vtcm_size) {
  if (options == nullptr || vtcm_size == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *vtcm_size = options->vtcm_size;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetNumHvxThreads(
    LiteRtQualcommOptions options, std::uint32_t num_hvx_threads) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->num_hvx_threads = num_hvx_threads;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetNumHvxThreads(
    LiteRtQualcommOptions options, std::uint32_t* num_hvx_threads) {
  if (options == nullptr || num_hvx_threads == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *num_hvx_threads = options->num_hvx_threads;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetOptimizationLevel(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsOptimizationLevel optimization_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->optimization_level = optimization_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetOptimizationLevel(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsOptimizationLevel* optimization_level) {
  if (options == nullptr || optimization_level == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *optimization_level = options->optimization_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetGraphPriority(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsGraphPriority graph_priority) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->graph_priority = graph_priority;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetGraphPriority(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsGraphPriority* graph_priority) {
  if (options == nullptr || graph_priority == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *graph_priority = options->graph_priority;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsSetBackend(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsBackend qnn_backend) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->qnn_backend = qnn_backend;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtQualcommOptionsGetBackend(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsBackend* qnn_backend) {
  if (qnn_backend == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *qnn_backend = options->qnn_backend;

  return kLiteRtStatusOk;
}
