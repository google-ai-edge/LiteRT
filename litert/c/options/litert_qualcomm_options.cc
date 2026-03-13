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
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_options_helper.h"
#include "litert/c/litert_common.h"
#include "litert/core/litert_toml_parser.h"

using litert::internal::ParseToml;

struct LrtQualcommOptionsT {
  std::optional<LrtQualcommOptionsLogLevel> log_level;
  std::optional<LrtQualcommOptionsProfiling> profiling;
  std::optional<bool> use_htp_preference;
  std::optional<bool> use_qint16_as_quint16;
  std::optional<bool> use_int64_bias_as_int32;
  std::optional<LrtQualcommOptionsBackend> qnn_backend;
  std::optional<bool> enable_weight_sharing;
  std::optional<bool> use_conv_hmx;
  std::optional<bool> use_fold_relu;
  std::optional<LrtQualcommOptionsHtpPerformanceMode> htp_performance_mode;
  std::optional<LrtQualcommOptionsDspPerformanceMode> dsp_performance_mode;
  std::optional<std::vector<std::int32_t>> dump_tensor_ids;
  std::optional<std::string> ir_json_dir;
  std::optional<std::string> dlc_dir;
  std::optional<std::uint32_t> vtcm_size;
  std::optional<std::uint32_t> num_hvx_threads;
  std::optional<LrtQualcommOptionsOptimizationLevel> optimization_level;
  std::optional<LrtQualcommOptionsGraphPriority> graph_priority;
  std::optional<std::string> saver_output_dir;
};

LiteRtStatus LrtCreateQualcommOptionsFromToml(const char* toml_payload,
                                              LrtQualcommOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LrtQualcommOptions parsed_options = nullptr;
  if (LrtCreateQualcommOptions(&parsed_options) != kLiteRtStatusOk)
    return kLiteRtStatusErrorRuntimeFailure;

  auto status = ParseToml(
      toml_payload,
      [&parsed_options](absl::string_view key,
                        absl::string_view value) -> LiteRtStatus {
        if (key == "log_level") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetLogLevel(
              parsed_options, static_cast<LrtQualcommOptionsLogLevel>(*v));
        } else if (key == "profiling") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetProfiling(
              parsed_options, static_cast<LrtQualcommOptionsProfiling>(*v));
        } else if (key == "use_htp_preference") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetUseHtpPreference(parsed_options, *v);
        } else if (key == "use_qint16_as_quint16") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetUseQint16AsQuint16(parsed_options, *v);
        } else if (key == "use_int64_bias_as_int32") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetUseInt64BiasAsInt32(parsed_options, *v);
        } else if (key == "qnn_backend") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetBackend(
              parsed_options, static_cast<LrtQualcommOptionsBackend>(*v));
        } else if (key == "enable_weight_sharing") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetEnableWeightSharing(parsed_options, *v);
        } else if (key == "use_conv_hmx") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetUseConvHMX(parsed_options, *v);
        } else if (key == "use_fold_relu") {
          auto v = litert::internal::ParseTomlBool(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetUseFoldReLU(parsed_options, *v);
        } else if (key == "htp_performance_mode") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetHtpPerformanceMode(
              parsed_options,
              static_cast<LrtQualcommOptionsHtpPerformanceMode>(*v));
        } else if (key == "dsp_performance_mode") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetDspPerformanceMode(
              parsed_options,
              static_cast<LrtQualcommOptionsDspPerformanceMode>(*v));
        } else if (key == "dump_tensor_ids") {
          auto parts = litert::internal::ParseTomlStringArray(value);
          if (!parts) return parts.Error().Status();
          std::vector<int32_t> ids;
          for (auto& part : parts.Value()) {
            auto v = litert::internal::ParseTomlInt(part);
            if (!v) return v.Error().Status();
            ids.push_back(*v);
          }
          LrtQualcommOptionsSetDumpTensorIds(parsed_options, ids.data(),
                                             ids.size());
        } else if (key == "ir_json_dir") {
          LrtQualcommOptionsSetIrJsonDir(parsed_options,
                                         std::string(value).c_str());
        } else if (key == "dlc_dir") {
          LrtQualcommOptionsSetDlcDir(parsed_options,
                                      std::string(value).c_str());
        } else if (key == "vtcm_size") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetVtcmSize(parsed_options,
                                        static_cast<uint32_t>(*v));
        } else if (key == "num_hvx_threads") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetNumHvxThreads(parsed_options,
                                             static_cast<uint32_t>(*v));
        } else if (key == "optimization_level") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetOptimizationLevel(
              parsed_options,
              static_cast<LrtQualcommOptionsOptimizationLevel>(*v));
        } else if (key == "graph_priority") {
          auto v = litert::internal::ParseTomlInt(value);
          if (!v) return v.Error().Status();
          LrtQualcommOptionsSetGraphPriority(
              parsed_options, static_cast<LrtQualcommOptionsGraphPriority>(*v));
        } else if (key == "saver_output_dir") {
          LrtQualcommOptionsSetSaverOutputDir(parsed_options,
                                              std::string(value).c_str());
        }

        return kLiteRtStatusOk;
      });

  if (status != kLiteRtStatusOk) {
    LrtDestroyQualcommOptions(parsed_options);
    return status;
  }

  *options = parsed_options;
  return kLiteRtStatusOk;
}

LiteRtStatus LrtCreateQualcommOptions(LrtQualcommOptions* options) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = new LrtQualcommOptionsT;
  return kLiteRtStatusOk;
}

void LrtDestroyQualcommOptions(LrtQualcommOptions options) { delete options; }

LiteRtStatus LrtGetOpaqueQualcommOptionsData(LrtQualcommOptions options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*)) {
  if (options == nullptr || identifier == nullptr || payload == nullptr ||
      payload_deleter == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  std::ostringstream toml;
  if (options->log_level.has_value()) {
    toml << "log_level = " << static_cast<int>(*options->log_level) << "\n";
  }
  if (options->profiling.has_value()) {
    toml << "profiling = " << static_cast<int>(*options->profiling) << "\n";
  }
  if (options->use_htp_preference.has_value()) {
    toml << "use_htp_preference = "
         << (*options->use_htp_preference ? "true" : "false") << "\n";
  }
  if (options->use_qint16_as_quint16.has_value()) {
    toml << "use_qint16_as_quint16 = "
         << (*options->use_qint16_as_quint16 ? "true" : "false") << "\n";
  }
  if (options->use_int64_bias_as_int32.has_value()) {
    toml << "use_int64_bias_as_int32 = "
         << (*options->use_int64_bias_as_int32 ? "true" : "false") << "\n";
  }
  if (options->qnn_backend.has_value()) {
    toml << "qnn_backend = " << static_cast<int>(*options->qnn_backend) << "\n";
  }
  if (options->enable_weight_sharing.has_value()) {
    toml << "enable_weight_sharing = "
         << (*options->enable_weight_sharing ? "true" : "false") << "\n";
  }
  if (options->use_conv_hmx.has_value()) {
    toml << "use_conv_hmx = " << (*options->use_conv_hmx ? "true" : "false")
         << "\n";
  }
  if (options->use_fold_relu.has_value()) {
    toml << "use_fold_relu = " << (*options->use_fold_relu ? "true" : "false")
         << "\n";
  }
  if (options->htp_performance_mode.has_value()) {
    toml << "htp_performance_mode = "
         << static_cast<int>(*options->htp_performance_mode) << "\n";
  }
  if (options->dsp_performance_mode.has_value()) {
    toml << "dsp_performance_mode = "
         << static_cast<int>(*options->dsp_performance_mode) << "\n";
  }
  if (options->dump_tensor_ids.has_value()) {
    toml << "dump_tensor_ids = [";
    for (size_t i = 0; i < options->dump_tensor_ids->size(); ++i) {
      if (i > 0) toml << ", ";
      toml << "\"" << (*options->dump_tensor_ids)[i] << "\"";
    }
    toml << "]\n";
  }
  if (options->ir_json_dir.has_value()) {
    toml << "ir_json_dir = \"" << *options->ir_json_dir << "\"\n";
  }
  if (options->dlc_dir.has_value()) {
    toml << "dlc_dir = \"" << *options->dlc_dir << "\"\n";
  }
  if (options->vtcm_size.has_value()) {
    toml << "vtcm_size = " << *options->vtcm_size << "\n";
  }
  if (options->num_hvx_threads.has_value()) {
    toml << "num_hvx_threads = " << *options->num_hvx_threads << "\n";
  }
  if (options->optimization_level.has_value()) {
    toml << "optimization_level = "
         << static_cast<int>(*options->optimization_level) << "\n";
  }
  if (options->graph_priority.has_value()) {
    toml << "graph_priority = " << static_cast<int>(*options->graph_priority)
         << "\n";
  }
  if (options->saver_output_dir.has_value()) {
    toml << "saver_output_dir = \"" << *options->saver_output_dir << "\"\n";
  }

  *identifier = LrtQualcommOptionsGetIdentifier();
  std::string toml_str = toml.str();
  litert::internal::MakeCStringPayload(toml_str, payload, payload_deleter);

  return kLiteRtStatusOk;
}

const char* LrtQualcommOptionsGetIdentifier() { return "qualcomm"; }

// GLOBAL OPTIONS //////////////////////////////////////////////////////////////

// log_level -------------------------------------------------------------------

LiteRtStatus LrtQualcommOptionsSetLogLevel(
    LrtQualcommOptions options, LrtQualcommOptionsLogLevel log_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->log_level = log_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetLogLevel(
    LrtQualcommOptions options, LrtQualcommOptionsLogLevel* log_level) {
  if (log_level == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *log_level = options->log_level.value_or(kLiteRtQualcommLogLevelInfo);

  return kLiteRtStatusOk;
}

// Profiling -------------------------------------------------------------------

LiteRtStatus LrtQualcommOptionsSetProfiling(
    LrtQualcommOptions options, LrtQualcommOptionsProfiling profiling) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->profiling = profiling;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetProfiling(
    LrtQualcommOptions options, LrtQualcommOptionsProfiling* profiling) {
  if (options == nullptr || profiling == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *profiling = options->profiling.value_or(kLiteRtQualcommProfilingOff);

  return kLiteRtStatusOk;
}

// Saver -------------------------------------------------------------------
LiteRtStatus LrtQualcommOptionsSetSaverOutputDir(LrtQualcommOptions options,
                                                 const char* saver_output_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->saver_output_dir = saver_output_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetSaverOutputDir(
    LrtQualcommOptions options, const char** saver_output_dir) {
  if (options == nullptr || saver_output_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *saver_output_dir = options->saver_output_dir.has_value()
                          ? options->saver_output_dir->c_str()
                          : "";

  return kLiteRtStatusOk;
}

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

LiteRtStatus LrtQualcommOptionsSetUseHtpPreference(LrtQualcommOptions options,
                                                   bool use_htp_preference) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_htp_preference = use_htp_preference;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetUseHtpPreference(LrtQualcommOptions options,
                                                   bool* use_htp_preference) {
  if (use_htp_preference == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_htp_preference = options->use_htp_preference.value_or(false);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetUseQint16AsQuint16(
    LrtQualcommOptions options, bool use_qint16_as_quint16) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_qint16_as_quint16 = use_qint16_as_quint16;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetUseQint16AsQuint16(
    LrtQualcommOptions options, bool* use_qint16_as_quint16) {
  if (use_qint16_as_quint16 == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_qint16_as_quint16 = options->use_qint16_as_quint16.value_or(false);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetUseInt64BiasAsInt32(
    LrtQualcommOptions options, bool use_int64_bias_as_int32) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_int64_bias_as_int32 = use_int64_bias_as_int32;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetUseInt64BiasAsInt32(
    LrtQualcommOptions options, bool* use_int64_bias_as_int32) {
  if (use_int64_bias_as_int32 == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_int64_bias_as_int32 = options->use_int64_bias_as_int32.value_or(true);

  return kLiteRtStatusOk;
}

// enable_weight_sharing -------------------------------------------------------

LiteRtStatus LrtQualcommOptionsSetEnableWeightSharing(
    LrtQualcommOptions options, bool enable_weight_sharing) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->enable_weight_sharing = enable_weight_sharing;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetEnableWeightSharing(
    LrtQualcommOptions options, bool* enable_weight_sharing) {
  if (enable_weight_sharing == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *enable_weight_sharing = options->enable_weight_sharing.value_or(false);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetDumpTensorIds(LrtQualcommOptions options,
                                                const std::int32_t* ids,
                                                size_t number_of_ids) {
  if (options == nullptr || ids == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->dump_tensor_ids =
      std::vector<std::int32_t>(ids, ids + number_of_ids);
  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetDumpTensorIds(LrtQualcommOptions options,
                                                const std::int32_t** ids,
                                                size_t* number_of_ids) {
  if (ids == nullptr || number_of_ids == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!options->dump_tensor_ids.has_value()) {
    *ids = nullptr;
    *number_of_ids = 0;
  } else {
    *ids = options->dump_tensor_ids->data();
    *number_of_ids = options->dump_tensor_ids->size();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetUseConvHMX(LrtQualcommOptions options,
                                             bool use_conv_hmx) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_conv_hmx = use_conv_hmx;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetUseConvHMX(LrtQualcommOptions options,
                                             bool* use_conv_hmx) {
  if (use_conv_hmx == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_conv_hmx = options->use_conv_hmx.value_or(true);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetUseFoldReLU(LrtQualcommOptions options,
                                              bool use_fold_relu) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->use_fold_relu = use_fold_relu;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetUseFoldReLU(LrtQualcommOptions options,
                                              bool* use_fold_relu) {
  if (use_fold_relu == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *use_fold_relu = options->use_fold_relu.value_or(true);

  return kLiteRtStatusOk;
}

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

LiteRtStatus LrtQualcommOptionsSetHtpPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsHtpPerformanceMode htp_performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->htp_performance_mode = htp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetHtpPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsHtpPerformanceMode* htp_performance_mode) {
  if (options == nullptr || htp_performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *htp_performance_mode = options->htp_performance_mode.value_or(
      kLiteRtQualcommHtpPerformanceModeDefault);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetDspPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsDspPerformanceMode dsp_performance_mode) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->dsp_performance_mode = dsp_performance_mode;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetDspPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsDspPerformanceMode* dsp_performance_mode) {
  if (options == nullptr || dsp_performance_mode == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *dsp_performance_mode = options->dsp_performance_mode.value_or(
      kLiteRtQualcommDspPerformanceModeDefault);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetIrJsonDir(LrtQualcommOptions options,
                                            const char* ir_json_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->ir_json_dir = ir_json_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetIrJsonDir(LrtQualcommOptions options,
                                            const char** ir_json_dir) {
  if (options == nullptr || ir_json_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *ir_json_dir =
      options->ir_json_dir.has_value() ? options->ir_json_dir->c_str() : "";

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetDlcDir(LrtQualcommOptions options,
                                         const char* dlc_dir) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->dlc_dir = dlc_dir;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetDlcDir(LrtQualcommOptions options,
                                         const char** dlc_dir) {
  if (options == nullptr || dlc_dir == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *dlc_dir = options->dlc_dir.has_value() ? options->dlc_dir->c_str() : "";

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetVtcmSize(LrtQualcommOptions options,
                                           std::uint32_t vtcm_size) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->vtcm_size = vtcm_size;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetVtcmSize(LrtQualcommOptions options,
                                           std::uint32_t* vtcm_size) {
  if (options == nullptr || vtcm_size == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *vtcm_size = options->vtcm_size.value_or(0);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetNumHvxThreads(LrtQualcommOptions options,
                                                std::uint32_t num_hvx_threads) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->num_hvx_threads = num_hvx_threads;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetNumHvxThreads(
    LrtQualcommOptions options, std::uint32_t* num_hvx_threads) {
  if (options == nullptr || num_hvx_threads == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *num_hvx_threads = options->num_hvx_threads.value_or(0);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetOptimizationLevel(
    LrtQualcommOptions options,
    LrtQualcommOptionsOptimizationLevel optimization_level) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->optimization_level = optimization_level;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetOptimizationLevel(
    LrtQualcommOptions options,
    LrtQualcommOptionsOptimizationLevel* optimization_level) {
  if (options == nullptr || optimization_level == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *optimization_level =
      options->optimization_level.value_or(kHtpOptimizeForInferenceO3);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetGraphPriority(
    LrtQualcommOptions options,
    LrtQualcommOptionsGraphPriority graph_priority) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->graph_priority = graph_priority;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetGraphPriority(
    LrtQualcommOptions options,
    LrtQualcommOptionsGraphPriority* graph_priority) {
  if (options == nullptr || graph_priority == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *graph_priority =
      options->graph_priority.value_or(kLiteRTQualcommGraphPriorityDefault);

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsSetBackend(
    LrtQualcommOptions options, LrtQualcommOptionsBackend qnn_backend) {
  if (options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  options->qnn_backend = qnn_backend;

  return kLiteRtStatusOk;
}

LiteRtStatus LrtQualcommOptionsGetBackend(
    LrtQualcommOptions options, LrtQualcommOptionsBackend* qnn_backend) {
  if (qnn_backend == nullptr || options == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *qnn_backend = options->qnn_backend.value_or(kLiteRtQualcommBackendHtp);

  return kLiteRtStatusOk;
}
