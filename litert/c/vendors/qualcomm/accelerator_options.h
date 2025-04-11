// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_
#define ODML_LITERT_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_

#include "litert/c/litert_accelerator_compilation_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQnnLogLevel {
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
} LiteRtQnnLogLevel;

typedef enum LiteRtQnnHtpPerformanceMode {
  kHtpDefault = 0,
  kHtpSustainedHighPerformance = 1,
  kHtpBurst = 2,
  kHtpHighPerformance = 3,
  kHtpPowerSaver = 4,
  kHtpLowPowerSaver = 5,
  kHtpHighPowerSaver = 6,
  kHtpLowBalanced = 7,
  kHtpBalanced = 8,
  kHtpExtremePowerSaver = 9,
} LiteRtQnnHtpPerformanceMode;

typedef enum LiteRtQnnProfilingOptions {
  kProfilingOff = 0,
  kBasicProfiling,
  kPerOpProfiling,
  kLintingProfiling,
} LiteRtQnnProfilingOptions;

LiteRtStatus LiteRtCreateQnnAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions *options);

LiteRtStatus LiteRtDestroyQnnAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options);

// LiteRtStatus LiteRtSetQnnAcceleratorLogLevel(
//     LiteRtAcceleratorCompilationOptions options, LiteRtQnnLogLevel log_level);

// LiteRtStatus LiteRtGetQnnAcceleratorLogLevel(
//     LiteRtAcceleratorCompilationOptions options, LiteRtQnnLogLevel *log_level);

// LiteRtStatus LiteRtSetQnnAcceleratorHtpPerformanceMode(
//     LiteRtAcceleratorCompilationOptions options,
//     TfLiteQnnDelegateHtpPerformanceMode mode);

// LiteRtStatus LiteRtGetQnnAcceleratorHtpPerformanceMode(
//     LiteRtAcceleratorCompilationOptions options,
//     TfLiteQnnDelegateHtpPerformanceMode *mode);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_
