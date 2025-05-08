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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a mediatek options object that is type erased. The actual option
// data can be accessed from the payload.
LiteRtStatus LiteRtMediatekOptionsCreate(LiteRtOpaqueOptions* options);
LITERT_DEFINE_HANDLE(LiteRtMediatekOptions);

// The a string identifier that discriminates mediatek options within
// type erased options.
const char* LiteRtMediatekOptionsGetIdentifier();

// Attempt to retrieve mediatek options from the opaque options. Fails
// unless the opaque options are of another type.
LiteRtStatus LiteRtMediatekOptionsGet(LiteRtOpaqueOptions options,
                                      LiteRtMediatekOptions* options_data);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// sdk_version_type -------------------------------------------------------
typedef enum LiteRtMediatekOptionsNeronSDKVersionType {
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion7 = 0,
  kLiteRtMediatekOptionsNeronSDKVersionTypeVersion8 = 1,
} LiteRtMediatekOptionsNeronSDKVersion;
LiteRtStatus LiteRtMediatekOptionsSetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    enum LiteRtMediatekOptionsNeronSDKVersionType sdk_version_type);
LiteRtStatus LiteRtMediatekOptionsGetNeronSDKVersionType(
    LiteRtMediatekOptions options,
    enum LiteRtMediatekOptionsNeronSDKVersionType* sdk_version_type);
#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_MEDIATEK_OPTIONS_H_
