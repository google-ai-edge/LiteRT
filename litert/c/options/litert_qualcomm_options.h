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

LITERT_DEFINE_HANDLE(LiteRtQualcommOptions);

// Create a qualcomm options object that is type erased. The actual option
// data can be accessed from the payload.
LiteRtStatus LiteRtQualcommOptionsCreate(LiteRtOpaqueOptions* options);

// The a string identifier that discriminates qualcomm options within
// type erased options.
const char* LiteRtQualcommOptionsGetIdentifier();

// Attempt to retrieve qualcomm options from the opaque options. Fails unless
// the opaque options are of another type.
LiteRtStatus LiteRtQualcommOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtQualcommOptions* options_data);

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

// log_level

// This determines the logging level of all underlying qualcomm sdk libraries.
// Does not effect litert logging. Defaults to INFO.

typedef enum LiteRtQualcommOptionsLogLevel {
  kLiteRtQualcommLogOff = 0,
  kLiteRtQualcommLogLevelError = 1,
  kLiteRtQualcommLogLevelWarn = 2,
  kLiteRtQualcommLogLevelInfo = 3,
  kLiteRtQualcommLogLevelVerbose = 4,
  kLiteRtQualcommLogLevelDebug = 5,
} LiteRtQualcommOptionsLogLevel;

LiteRtStatus LiteRtQualcommOptionsSetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel log_level);

LiteRtStatus LiteRtQualcommOptionsGetLogLevel(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsLogLevel* log_level);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// use_htp_preference

// This option controls whether to convert a LiteRt operation to QNN operations
// which are preferred by the HTP backend. Defaults to false.

LiteRtStatus LiteRtQualcommOptionsSetUseHtpPreference(
    LiteRtQualcommOptions options, bool use_htp_preference);

LiteRtStatus LiteRtQualcommOptionsGetUseHtpPreference(
    LiteRtQualcommOptions options, bool* use_htp_preference);

// use_qint16_as_quint16

// This option controls whether to convert a quantized int16 model to a
// quantized uint16 model. Defaults to false.

LiteRtStatus LiteRtQualcommOptionsSetUseQint16AsQuint16(
    LiteRtQualcommOptions options, bool use_qint16_as_quint16);

LiteRtStatus LiteRtQualcommOptionsGetUseQint16AsQuint16(
    LiteRtQualcommOptions options, bool* use_qint16_as_quint16);

// enable_weight_sharing

// Weight sharing indicates whether different subgraphs may share weight
// tensors. This is only supported on x86 AOT. Defaults to false.

LiteRtStatus LiteRtQualcommOptionsSetEnableWeightSharing(
    LiteRtQualcommOptions options, bool enable_weight_sharing);

LiteRtStatus LiteRtQualcommOptionsGetEnableWeightSharing(
    LiteRtQualcommOptions options, bool* enable_weight_sharing);

LiteRtStatus LiteRtQualcommOptionsSetDumpTensorIds(
    LiteRtQualcommOptions options, const int32_t* ids, uint32_t number_of_ids);

LiteRtStatus LiteRtQualcommOptionsGetDumpTensorIds(
    LiteRtQualcommOptions options, int32_t** ids, uint32_t* number_of_ids);

// When using short conv hmx, one might have better performance, but convolution
// that have short depth and/or weights that are not symmetric could exhibit
// inaccurate results.

LiteRtStatus LiteRtQualcommOptionsSetUseConvHMX(LiteRtQualcommOptions options,
                                                bool use_conv_hmx);

LiteRtStatus LiteRtQualcommOptionsGetUseConvHMX(LiteRtQualcommOptions options,
                                                bool* use_conv_hmx);

// When using fold relu, one might have better performance. This optimization is
// correct when quantization ranges for convolution are equal to or are subset
// of the Relu operation.

LiteRtStatus LiteRtQualcommOptionsSetUseFoldReLU(LiteRtQualcommOptions options,
                                                 bool use_fold_relu);

LiteRtStatus LiteRtQualcommOptionsGetUseFoldReLU(LiteRtQualcommOptions options,
                                                 bool* use_fold_relu);

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

// htp_performance_mode

// Configures the HTP device to optimize for performance or power efficiency.
// See QnnHtpPerfInfrastructure_SetPowerConfigFn_t in qnn_sdk. By default, it
// will be decided by the backend (unknown).

typedef enum LiteRtQualcommOptionsHtpPerformanceMode {
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
} LiteRtQualcommOptionsHtpPerformanceMode;

LiteRtStatus LiteRtQualcommOptionsSetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode htp_performance_mode);

LiteRtStatus LiteRtQualcommOptionsGetHtpPerformanceMode(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsHtpPerformanceMode* htp_performance_mode);

// profiling

// This option controls the profiling level. A higher level results in a more
// detailed report after execution. Defaults to off.

typedef enum LiteRtQualcommOptionsProfiling {
  kLiteRtQualcommProfilingOff = 0,
  kLiteRtQualcommProfilingBasic,
  kLiteRtQualcommProfilingDetailed,
  kLiteRtQualcommProfilingLinting,
  kLiteRtQualcommProfilingOptrace,
} LiteRtQualcommOptionsProfiling;

LiteRtStatus LiteRtQualcommOptionsSetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling profiling);

LiteRtStatus LiteRtQualcommOptionsGetProfiling(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsProfiling* profiling);

LiteRtStatus LiteRtQualcommOptionsSetIrJsonDir(LiteRtQualcommOptions options,
                                               const char* ir_json_dir);

LiteRtStatus LiteRtQualcommOptionsGetIrJsonDir(LiteRtQualcommOptions options,
                                               const char** ir_json_dir);

LiteRtStatus LiteRtQualcommOptionsSetDlcDir(LiteRtQualcommOptions options,
                                            const char* dlc_dir);

LiteRtStatus LiteRtQualcommOptionsGetDlcDir(LiteRtQualcommOptions options,
                                            const char** dlc_dir);

LiteRtStatus LiteRtQualcommOptionsSetVtcmSize(LiteRtQualcommOptions options,
                                              uint32_t vtcm_size);

LiteRtStatus LiteRtQualcommOptionsGetVtcmSize(LiteRtQualcommOptions options,
                                              uint32_t* vtcm_size);

LiteRtStatus LiteRtQualcommOptionsSetNumHvxThreads(
    LiteRtQualcommOptions options, uint32_t num_hvx_threads);

LiteRtStatus LiteRtQualcommOptionsGetNumHvxThreads(
    LiteRtQualcommOptions options, uint32_t* num_hvx_threads);

typedef enum LiteRtQualcommOptionsOptimizationLevel {
  kHtpOptimizeForInference = 0,
  kHtpOptimizeForPrepare,
  kHtpOptimizeForInferenceO3,
} LiteRtQualcommOptionsOptimizationLevel;

LiteRtStatus LiteRtQualcommOptionsSetOptimizationLevel(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsOptimizationLevel optimization_level);

LiteRtStatus LiteRtQualcommOptionsGetOptimizationLevel(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsOptimizationLevel* optimization_level);

typedef enum LiteRtQualcommOptionsGraphPriority {
  kLiteRTQualcommGraphPriorityDefault = 0,
  kLiteRTQualcommGraphPriorityLow,
  kLiteRTQualcommGraphPriorityNormal,
  kLiteRTQualcommGraphPriorityNormalHigh,
  kLiteRTQualcommGraphPriorityHigh,
} LiteRtQualcommOptionsGraphPriority;

LiteRtStatus LiteRtQualcommOptionsSetGraphPriority(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsGraphPriority graph_priority);

LiteRtStatus LiteRtQualcommOptionsGetGraphPriority(
    LiteRtQualcommOptions options,
    LiteRtQualcommOptionsGraphPriority* graph_priority);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
