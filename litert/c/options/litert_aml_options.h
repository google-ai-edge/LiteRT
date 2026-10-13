// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_

#include "litert/c/litert_common.h"

// User-facing options for Amlogic. Do not include vendor SDK headers.
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtAmlOptions);

const char* LrtAmlOptionsGetIdentifier();

LiteRtStatus LrtCreateAmlOptions(LrtAmlOptions* options);

#ifdef __cplusplus
LiteRtStatus LrtCreateAmlOptionsFromToml(const char* toml_payload,
                                         LrtAmlOptions* options);
#endif  // __cplusplus

void LrtDestroyAmlOptions(LrtAmlOptions options);

LiteRtStatus LrtGetOpaqueAmlOptionsData(LrtAmlOptions options,
                                        const char** identifier, void** payload,
                                        void (**payload_deleter)(void*));

typedef enum LrtAmlOptionsLogLevel {
  kLrtAmlLogOff = 0,
  kLrtAmlLogLevelError = 1,
  kLrtAmlLogLevelWarn = 2,
  kLrtAmlLogLevelInfo = 3,
  kLrtAmlLogLevelVerbose = 4,
  kLrtAmlLogLevelDebug = 5,
} LrtAmlOptionsLogLevel;

LiteRtStatus LrtAmlOptionsSetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel log_level);
LiteRtStatus LrtAmlOptionsGetLogLevel(LrtAmlOptions options,
                                      LrtAmlOptionsLogLevel* log_level);

typedef enum LrtAmlOptionsPerfMode {
  kLrtAmlPerfDefault = 0,
  kLrtAmlPerfHighPerformance,
  kLrtAmlPerfBalanced,
  kLrtAmlPerfPowerSaver,
} LrtAmlOptionsPerfMode;

LiteRtStatus LrtAmlOptionsSetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode mode);
LiteRtStatus LrtAmlOptionsGetPerfMode(LrtAmlOptions options,
                                      LrtAmlOptionsPerfMode* mode);

typedef enum LrtAmlOptionsProfiling {
  kLrtAmlProfilingOff = 0,
  kLrtAmlProfilingBasic,
  kLrtAmlProfilingDetailed,
} LrtAmlOptionsProfiling;

LiteRtStatus LrtAmlOptionsSetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling profiling);
LiteRtStatus LrtAmlOptionsGetProfiling(LrtAmlOptions options,
                                       LrtAmlOptionsProfiling* profiling);

typedef enum LrtAmlOptionsMemoryPolicy {
  kLrtAmlMemoryDefault = 0,
  kLrtAmlMemoryIOMMU,
  kLrtAmlMemoryDMA,
} LrtAmlOptionsMemoryPolicy;

LiteRtStatus LrtAmlOptionsSetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy policy);
LiteRtStatus LrtAmlOptionsGetMemoryPolicy(LrtAmlOptions options,
                                          LrtAmlOptionsMemoryPolicy* policy);

LiteRtStatus LrtAmlOptionsSetModelPath(LrtAmlOptions options,
                                       const char* model_path);
LiteRtStatus LrtAmlOptionsGetModelPath(LrtAmlOptions options,
                                       const char** model_path);

LiteRtStatus LrtAmlOptionsSetModelName(LrtAmlOptions options,
                                       const char* model_name);
LiteRtStatus LrtAmlOptionsGetModelName(LrtAmlOptions options,
                                       const char** model_name);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_AML_OPTIONS_H_
