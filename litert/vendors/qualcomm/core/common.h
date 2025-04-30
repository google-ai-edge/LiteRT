
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQnnProfilingOptions {
  kQnnProfilingOff = 0,
  kQnnProfilingBasic = 1,
  kQnnProfilingDetailed = 2
} LiteRtQnnProfilingOptions;

typedef enum LiteRtQnnLogLevel {
  /// Disable delegate and QNN backend logging messages.
  kQnnLogOff = 0,
  kQnnLogLevelError = 1,
  kQnnLogLevelWarn = 2,
  kQnnLogLevelInfo = 3,
  kQnnLogLevelVerbose = 4,
  kQnnLogLevelDebug = 5,
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

typedef struct {
  /// The default performance mode sets no configurations on the HTP.
  LiteRtQnnHtpPerformanceMode performance_mode;
} LiteRtQnnHtpBackendOptions;

// clang-format off
#define LITERT_QNN_HTP_OPTION_INIT {kHtpDefault /*performance_mode*/}
// clang-format on

typedef struct {
  /// Apply HTP-friendly op builder.
  bool use_htp_preferences;
  /// This option will treat quantized int16 tensor as quantized uint16 tensor
  /// for better backend compatibility.
  bool use_qint16_as_quint16;
  /// Optional backend specific options for the HTP backend.
  LiteRtQnnHtpBackendOptions htp_options;
  /// Log level
  LiteRtQnnLogLevel log_level;
} LiteRtQnnOptions;

// This option can be used to specify QNN options.
static const char* kDispatchOptionLiteRtQnnOptions = "litert_qnn_options";

// clang-format off
#define LITERT_QNN_OPTIONS_INIT                                  \
  {                                                              \
      false,                      /*use_htp_preferences*/        \
      true,                       /*use_qint16_as_quint16*/      \
      LITERT_QNN_HTP_OPTION_INIT, /*LiteRtQnnHtpBackendOptions*/ \
      kQnnLogOff,                 /*log_level*/                  \
  }
// clang-format on
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
