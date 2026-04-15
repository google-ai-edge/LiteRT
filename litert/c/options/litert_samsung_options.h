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

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_

#include <stdint.h>

#include "litert/c/litert_common.h"

// C-API for an opaque options type relevant to Samsung (both dspatch and
// plugin).
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtSamsungOptions);

// The string identifier that discriminates samsung options.
const char* LrtSamsungOptionsGetIdentifier();

// Create a samsung options object.
LiteRtStatus LrtCreateSamsungOptions(LrtSamsungOptions* options);

#ifdef __cplusplus
// Create a qualcomm options object mapped from a TOML payload.
LiteRtStatus LrtCreateSamsungOptionsFromToml(const char* toml_payload,
                                             LrtSamsungOptions* options);
#endif  // __cplusplus

// Destroy a samsung options object.
void LrtDestroySamsungOptions(LrtSamsungOptions options);

LiteRtStatus LrtGetOpaqueSamsungOptionsData(LrtSamsungOptions options,
                                            const char** identifier,
                                            void** payload,
                                            void (**payload_deleter)(void*));

// COMPILATION OPTIONS ////////////////////////////////////////////////////////

LiteRtStatus LrtSamsungOptionsSetEnableLargeModelSupport(
    LrtSamsungOptions options, bool enable_large_model_support);

LiteRtStatus LrtSamsungOptionsGetEnableLargeModelSupport(
    LrtSamsungOptions options, bool* enable_large_model_support);
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_SAMSUNG_OPTIONS_H_
