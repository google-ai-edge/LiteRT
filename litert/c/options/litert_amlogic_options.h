// Copyright 2026 Amlogic, Inc.
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

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_

#include "litert/c/litert_common.h"

// User-facing options for Amlogic. Do not include vendor SDK headers.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtAmlogicOptions);

const char* LrtAmlogicOptionsGetIdentifier();

LiteRtStatus LrtCreateAmlogicOptions(LrtAmlogicOptions* options);

#ifdef __cplusplus
LiteRtStatus LrtCreateAmlogicOptionsFromToml(const char* toml_payload,
                                         LrtAmlogicOptions* options);
#endif  // __cplusplus

void LrtDestroyAmlogicOptions(LrtAmlogicOptions options);

LiteRtStatus LrtGetOpaqueAmlogicOptionsData(LrtAmlogicOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*));

// Public log level. The compiler plugin maps this onto ADLA_LogLevel
// (VERBOSE=2 .. FATAL=7). These enumerator values are not ADLA's numbers.
// Unset means the plugin follows LiteRT's own logger severity.
typedef enum LrtAmlogicOptionsLogLevel {
  kLrtAmlogicLogOff = 0,
  kLrtAmlogicLogLevelError = 1,
  kLrtAmlogicLogLevelWarn = 2,
  kLrtAmlogicLogLevelInfo = 3,
  kLrtAmlogicLogLevelVerbose = 4,
  kLrtAmlogicLogLevelDebug = 5,
} LrtAmlogicOptionsLogLevel;

LiteRtStatus LrtAmlogicOptionsSetLogLevel(LrtAmlogicOptions options,
                                      LrtAmlogicOptionsLogLevel log_level);
LiteRtStatus LrtAmlogicOptionsGetLogLevel(LrtAmlogicOptions options,
                                      LrtAmlogicOptionsLogLevel* log_level);

LiteRtStatus LrtAmlogicOptionsSetModelPath(LrtAmlogicOptions options,
                                       const char* model_path);
LiteRtStatus LrtAmlogicOptionsGetModelPath(LrtAmlogicOptions options,
                                       const char** model_path);

LiteRtStatus LrtAmlogicOptionsSetModelName(LrtAmlogicOptions options,
                                       const char* model_name);
LiteRtStatus LrtAmlogicOptionsGetModelName(LrtAmlogicOptions options,
                                       const char** model_name);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_
