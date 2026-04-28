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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_

#include <stddef.h>
#include <stdint.h>

#include "litert/c/litert_common.h"

// User-facing options for Qualcomm. This is not built as part of
// libLiteRt_QualcommXXX.so, and should not include `vendor/` or qnn sdk
// headers.

// C-API for an opaque options type relevant to Qualcomm (both dspatch and
// plugin).
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtQualcommOptions);

// The string identifier that discriminates qualcomm options within
// type erased options.
const char* LrtQualcommOptionsGetIdentifier();

// Create a qualcomm options object.
LiteRtStatus LrtCreateQualcommOptions(LrtQualcommOptions* options);

#ifdef __cplusplus
// Create a qualcomm options object mapped from a TOML payload.
LiteRtStatus LrtCreateQualcommOptionsFromToml(const char* toml_payload,
                                              LrtQualcommOptions* options);
#endif  // __cplusplus

// Destroy a qualcomm options object.
void LrtDestroyQualcommOptions(LrtQualcommOptions options);

LiteRtStatus LrtGetOpaqueQualcommOptionsData(LrtQualcommOptions options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*));

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

// log_level

// This determines the logging level of all underlying qualcomm sdk libraries.
// Does not effect litert logging. Defaults to INFO.

typedef enum LrtQualcommOptionsLogLevel {
  kLiteRtQualcommLogOff = 0,
  kLiteRtQualcommLogLevelError = 1,
  kLiteRtQualcommLogLevelWarn = 2,
  kLiteRtQualcommLogLevelInfo = 3,
  kLiteRtQualcommLogLevelVerbose = 4,
  kLiteRtQualcommLogLevelDebug = 5,
} LrtQualcommOptionsLogLevel;

LiteRtStatus LrtQualcommOptionsSetLogLevel(
    LrtQualcommOptions options, LrtQualcommOptionsLogLevel log_level);

LiteRtStatus LrtQualcommOptionsGetLogLevel(
    LrtQualcommOptions options, LrtQualcommOptionsLogLevel* log_level);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// use_htp_preference
// @deprecated This option is deprecated and will be no-op.

// This option controls whether to convert a LiteRt operation to QNN operations
// which are preferred by the HTP backend. Defaults to false.

LiteRtStatus LrtQualcommOptionsSetUseHtpPreference(LrtQualcommOptions options,
                                                   bool use_htp_preference);

LiteRtStatus LrtQualcommOptionsGetUseHtpPreference(LrtQualcommOptions options,
                                                   bool* use_htp_preference);

// use_qint16_as_quint16
// @deprecated This option is deprecated and will be no-op.

// This option controls whether to convert a quantized int16 model to a
// quantized uint16 model. Defaults to false.

LiteRtStatus LrtQualcommOptionsSetUseQint16AsQuint16(
    LrtQualcommOptions options, bool use_qint16_as_quint16);

LiteRtStatus LrtQualcommOptionsGetUseQint16AsQuint16(
    LrtQualcommOptions options, bool* use_qint16_as_quint16);

// use_int64_bias_as_int32

// This option controls whether to convert bias tensors of FullyConnected
// and Conv2D Ops from int64 to int32 . Defaults to true.

LiteRtStatus LrtQualcommOptionsSetUseInt64BiasAsInt32(
    LrtQualcommOptions options, bool use_int64_bias_as_int32);

LiteRtStatus LrtQualcommOptionsGetUseInt64BiasAsInt32(
    LrtQualcommOptions options, bool* use_int64_bias_as_int32);

// enable_weight_sharing

// Weight sharing indicates whether different subgraphs may share weight
// tensors. This is only supported on x86 AOT. Defaults to false.

LiteRtStatus LrtQualcommOptionsSetEnableWeightSharing(
    LrtQualcommOptions options, bool enable_weight_sharing);

LiteRtStatus LrtQualcommOptionsGetEnableWeightSharing(
    LrtQualcommOptions options, bool* enable_weight_sharing);

LiteRtStatus LrtQualcommOptionsSetDumpTensorIds(LrtQualcommOptions options,
                                                const int32_t* ids,
                                                size_t number_of_ids);

LiteRtStatus LrtQualcommOptionsGetDumpTensorIds(LrtQualcommOptions options,
                                                const int32_t** ids,
                                                size_t* number_of_ids);

// When using short conv hmx, one might have better performance, but convolution
// that have short depth and/or weights that are not symmetric could exhibit
// inaccurate results.

LiteRtStatus LrtQualcommOptionsSetUseConvHMX(LrtQualcommOptions options,
                                             bool use_conv_hmx);

LiteRtStatus LrtQualcommOptionsGetUseConvHMX(LrtQualcommOptions options,
                                             bool* use_conv_hmx);

// When using fold relu, one might have better performance. This optimization is
// correct when quantization ranges for convolution are equal to or are subset
// of the Relu operation.

LiteRtStatus LrtQualcommOptionsSetUseFoldReLU(LrtQualcommOptions options,
                                              bool use_fold_relu);

LiteRtStatus LrtQualcommOptionsGetUseFoldReLU(LrtQualcommOptions options,
                                              bool* use_fold_relu);

// graph_io_tensor_mem_type

// This controls whether graph inputs and outputs use MemHandle or Raw buffers
// during compilation.

typedef enum LrtQualcommOptionsGraphIOTensorMemType {
  kLiteRtQualcommGraphIOTensorMemTypeRaw = 0,
  kLiteRtQualcommGraphIOTensorMemTypeMemHandle,
} LrtQualcommOptionsGraphIOTensorMemType;

LiteRtStatus LrtQualcommOptionsSetGraphIOTensorMemType(
    LrtQualcommOptions options,
    LrtQualcommOptionsGraphIOTensorMemType mem_type);

LiteRtStatus LrtQualcommOptionsGetGraphIOTensorMemType(
    LrtQualcommOptions options,
    LrtQualcommOptionsGraphIOTensorMemType* mem_type);

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

// htp_performance_mode

// Configures the HTP device to optimize for performance or power efficiency.
// See QnnHtpPerfInfrastructure_SetPowerConfigFn_t in qnn_sdk. By default, it
// will be decided by the backend (unknown).

typedef enum LrtQualcommOptionsHtpPerformanceMode {
  kLiteRtQualcommHtpPerformanceModeDefault = 0,
  kLiteRtQualcommHtpPerformanceModeSustainedHighPerformance = 1,
  kLiteRtQualcommHtpPerformanceModeBurst = 2,
  kLiteRtQualcommHtpPerformanceModeHighPerformance = 3,
  kLiteRtQualcommHtpPerformanceModePowerSaver = 4,
  kLiteRtQualcommHtpPerformanceModeLowPowerSaver = 5,
  kLiteRtQualcommHtpPerformanceModeHighPowerSaver = 6,
  kLiteRtQualcommHtpPerformanceModeLowBalanced = 7,
  kLiteRtQualcommHtpPerformanceModeBalanced = 8,
  kLiteRtQualcommHtpPerformanceModeExtremePowerSaver = 9,
} LrtQualcommOptionsHtpPerformanceMode;

LiteRtStatus LrtQualcommOptionsSetHtpPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsHtpPerformanceMode htp_performance_mode);

LiteRtStatus LrtQualcommOptionsGetHtpPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsHtpPerformanceMode* htp_performance_mode);

// dsp_performance_mode

typedef enum LrtQualcommOptionsDspPerformanceMode {
  kLiteRtQualcommDspPerformanceModeDefault = 0,
  kLiteRtQualcommDspPerformanceModeSustainedHighPerformance = 1,
  kLiteRtQualcommDspPerformanceModeBurst = 2,
  kLiteRtQualcommDspPerformanceModeHighPerformance = 3,
  kLiteRtQualcommDspPerformanceModePowerSaver = 4,
  kLiteRtQualcommDspPerformanceModeLowPowerSaver = 5,
  kLiteRtQualcommDspPerformanceModeHighPowerSaver = 6,
  kLiteRtQualcommDspPerformanceModeLowBalanced = 7,
  kLiteRtQualcommDspPerformanceModeBalanced = 8,
} LrtQualcommOptionsDspPerformanceMode;

LiteRtStatus LrtQualcommOptionsSetDspPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsDspPerformanceMode dsp_performance_mode);

LiteRtStatus LrtQualcommOptionsGetDspPerformanceMode(
    LrtQualcommOptions options,
    LrtQualcommOptionsDspPerformanceMode* dsp_performance_mode);

// profiling

// This option controls the profiling level. A higher level results in a more
// detailed report after execution. Defaults to off.

typedef enum LrtQualcommOptionsProfiling {
  kLiteRtQualcommProfilingOff = 0,
  kLiteRtQualcommProfilingBasic,
  kLiteRtQualcommProfilingDetailed,
  kLiteRtQualcommProfilingLinting,
  kLiteRtQualcommProfilingOptrace,
} LrtQualcommOptionsProfiling;

LiteRtStatus LrtQualcommOptionsSetProfiling(
    LrtQualcommOptions options, LrtQualcommOptionsProfiling profiling);

LiteRtStatus LrtQualcommOptionsGetProfiling(
    LrtQualcommOptions options, LrtQualcommOptionsProfiling* profiling);

LiteRtStatus LrtQualcommOptionsSetIrJsonDir(LrtQualcommOptions options,
                                            const char* ir_json_dir);

LiteRtStatus LrtQualcommOptionsGetIrJsonDir(LrtQualcommOptions options,
                                            const char** ir_json_dir);

LiteRtStatus LrtQualcommOptionsSetDlcDir(LrtQualcommOptions options,
                                         const char* dlc_dir);

LiteRtStatus LrtQualcommOptionsGetDlcDir(LrtQualcommOptions options,
                                         const char** dlc_dir);

LiteRtStatus LrtQualcommOptionsSetVtcmSize(LrtQualcommOptions options,
                                           uint32_t vtcm_size);

LiteRtStatus LrtQualcommOptionsGetVtcmSize(LrtQualcommOptions options,
                                           uint32_t* vtcm_size);

LiteRtStatus LrtQualcommOptionsSetNumHvxThreads(LrtQualcommOptions options,
                                                uint32_t num_hvx_threads);

LiteRtStatus LrtQualcommOptionsGetNumHvxThreads(LrtQualcommOptions options,
                                                uint32_t* num_hvx_threads);

typedef enum LrtQualcommOptionsOptimizationLevel {
  kHtpOptimizeForInference = 0,
  kHtpOptimizeForPrepare,
  kHtpOptimizeForInferenceO3,
} LrtQualcommOptionsOptimizationLevel;

LiteRtStatus LrtQualcommOptionsSetOptimizationLevel(
    LrtQualcommOptions options,
    LrtQualcommOptionsOptimizationLevel optimization_level);

LiteRtStatus LrtQualcommOptionsGetOptimizationLevel(
    LrtQualcommOptions options,
    LrtQualcommOptionsOptimizationLevel* optimization_level);

typedef enum LrtQualcommOptionsGraphPriority {
  kLiteRTQualcommGraphPriorityDefault = 0,
  kLiteRTQualcommGraphPriorityLow,
  kLiteRTQualcommGraphPriorityNormal,
  kLiteRTQualcommGraphPriorityNormalHigh,
  kLiteRTQualcommGraphPriorityHigh,
} LrtQualcommOptionsGraphPriority;

LiteRtStatus LrtQualcommOptionsSetGraphPriority(
    LrtQualcommOptions options, LrtQualcommOptionsGraphPriority graph_priority);

LiteRtStatus LrtQualcommOptionsGetGraphPriority(
    LrtQualcommOptions options,
    LrtQualcommOptionsGraphPriority* graph_priority);

typedef enum LrtQualcommOptionsBackend {
  kLiteRtQualcommBackendUndefined = 0,
  kLiteRtQualcommBackendGpu,
  kLiteRtQualcommBackendHtp,
  kLiteRtQualcommBackendDsp,
  kLiteRtQualcommBackendIr,
} LrtQualcommOptionsBackend;

LiteRtStatus LrtQualcommOptionsSetBackend(
    LrtQualcommOptions options, LrtQualcommOptionsBackend qnn_backend);

LiteRtStatus LrtQualcommOptionsGetBackend(
    LrtQualcommOptions options, LrtQualcommOptionsBackend* qnn_backend);

LiteRtStatus LrtQualcommOptionsSetSaverOutputDir(LrtQualcommOptions options,
                                                 const char* saver_output_dir);

LiteRtStatus LrtQualcommOptionsGetSaverOutputDir(LrtQualcommOptions options,
                                                 const char** saver_output_dir);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
