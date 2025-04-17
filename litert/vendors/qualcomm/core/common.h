
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQnnProfilingOptions {  // NOLINT(modernize-use-using)
  kQnnProfilingOff = 0,
  kQnnProfilingBasic = 1,
  kQnnProfilingDetailed = 2
} LiteRtQnnProfilingOptions;

typedef enum LiteRtQnnLogLevel {  // NOLINT(modernize-use-using)
  /// Disable delegate and QNN backend logging messages.
  kQnnLogOff = 0,
  kQnnLogLevelError = 1,
  kQnnLogLevelWarn = 2,
  kQnnLogLevelInfo = 3,
  kQnnLogLevelVerbose = 4,
  kQnnLogLevelDebug = 5,
} LiteRtQnnLogLevel;

typedef struct {  // NOLINT(modernize-use-using)
  /// Apply HTP-friendly op builder.
  bool useHtpPreferencs;
  /// This option will treat quantized int16 tensor as quantized uint16 tensor
  /// for better backend compatibility.
  bool useQInt16AsQUint16;
} LiteRtQnnOptions;
// clang-format off
#define LITERT_QNN_OPTIONS_INIT      \
  {                                  \
    false,    /*useHtpPreferencs*/   \
    true,     /*useQInt16AsQUint16*/ \
  }
// clang-format on
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
