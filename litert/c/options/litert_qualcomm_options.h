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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_opaque_options.h"

// User-facing options for Qualcomm. This is not built as part of
// libLiteRt_QualcommXXX.so, and should not include `vendor/` or qnn sdk
// headers.

// C-API for an opaque options type relevant to Qualcomm (both dspatch and
// plugin).
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a qualcomm options object that is type erased. The actual option
// data can be accessed from the payload.
LiteRtStatus LiteRtQualcommOptionsCreate(LiteRtOpaqueOptions* options);

LITERT_DEFINE_HANDLE(LiteRtQualcommOptions);

// The a string identifier that discriminates qualcomm options within
// type erased options.
const char* LiteRtQualcommOptionsGetIdentifier();

// Attempt to retieve qualcomm options from the opaque options. Fails unlesss
// the opaque options are of another type.
LiteRtStatus LiteRtQualcommOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtQualcommOptions* options_data);

// GENERAL SDK SETTINGS ////////////////////////////////////////////////////////

// log_level -------------------------------------------------------------------

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

// enable_weight_sharing -------------------------------------------------------

// Weight sharing indicates whether different subgraphs may share weight
// tensors. This is only supported on x86 AOT. Defaults to true.

LiteRtStatus LiteRtQualcommOptionsSetEnableWeightSharing(
    LiteRtQualcommOptions options, bool enable_weight_sharing);

LiteRtStatus LiteRtQualcommOptionsGetEnableWeightSharing(
    LiteRtQualcommOptions options, bool* enable_weight_sharing);

// DISPATCH OPTIONS ////////////////////////////////////////////////////////////

// power_mode ------------------------------------------------------------------

// Configures the HTP device to optimize for performance or power efficiency.
// See QnnHtpPerfInfrastructure_PowerMode_t in qnn_sdk. By default, it will
// be decided by the backend (unknown).

typedef enum LiteRtQualcommOptionsPowerMode {
  kQualcommPowerModeUknown = 0,
  kQualcommPowerModePerformance = 1,
  kQualcommPowerModePowerSaver = 2,
} LiteRtQualcommOptionsPowerMode;

LiteRtStatus LiteRtQualcommOptionsSetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode power_mode);

LiteRtStatus LiteRtQualcommOptionsGetPowerMode(
    LiteRtQualcommOptions options, LiteRtQualcommOptionsPowerMode* power_mode);

#ifdef __cplusplus
}  // extern "C"

// C++ WRAPPERS ////////////////////////////////////////////////////////////////

namespace litert::qualcomm {

// Wraps a LiteRtQualcommOptions object for convenience.
class QualcommOptions : public OpaqueOptions {
 public:
  using LogLevel = LiteRtQualcommOptionsLogLevel;
  using PowerMode = LiteRtQualcommOptionsPowerMode;

  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() {
    return LiteRtQualcommOptionsGetIdentifier();
  }

  static Expected<QualcommOptions> Create(OpaqueOptions& options);

  static Expected<QualcommOptions> Create();

  void SetLogLevel(LogLevel log_level);
  LogLevel GetLogLevel();

  void SetPowerMode(PowerMode power_mode);
  PowerMode GetPowerMode();

  void SetEnableWeightSharing(bool weight_sharing_enabled);
  bool GetEnableWeightSharing();

 private:
  LiteRtQualcommOptions Data() const;
};

}  // namespace litert::qualcomm

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_QUALCOMM_OPTIONS_H_
