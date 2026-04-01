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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_VERISILICON_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_VERISILICON_OPTIONS_H_
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtVerisiliconOptions);

// Creates a verisilicon options object.
// The caller is responsible for freeing the returned options using
// `LrtDestroyVerisiliconOptions`.
LiteRtStatus LrtCreateVerisiliconOptions(LrtVerisiliconOptions* options);

#ifdef __cplusplus
// Creates a verisilicon options object from a TOML payload.
LiteRtStatus LrtCreateVerisiliconOptionsFromToml(const char* toml_payload,
                                              LrtVerisiliconOptions* options);
#endif  // __cplusplus

// Destroys a verisilicon options object.
void LrtDestroyVerisiliconOptions(LrtVerisiliconOptions options);

// Serializes verisilicon options and returns the components needed to create
// opaque options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions` and freeing the returned payload using
// `payload_deleter`.
LiteRtStatus LrtGetOpaqueVerisiliconOptionsData(const LrtVerisiliconOptions options,
                                             const char** identifier,
                                             void** payload,
                                             void (**payload_deleter)(void*));

// The string identifier that discriminates verisilicon options within
// type erased options.
const char* LrtVerisiliconOptionsGetIdentifier();

// COMPILATION OPTIONS /////////////////////////////////////////////////////////


// DISPATCH OPTIONS /////////////////////////////////////////////////////////
// viplite_adapter options --------------------------------------------

// Specify the index of device for running the compiled model.
LiteRtStatus LrtVerisiliconOptionsSetDeviceIndex(
    LrtVerisiliconOptions options,
    unsigned int device_index);

LiteRtStatus LrtVerisiliconOptionsGetDeviceIndex(
    const LrtVerisiliconOptions options,
    unsigned int* device_index);

//Specify the index of VIP-core for running the compiled model.
LiteRtStatus LrtVerisiliconOptionsSetCoreIndex(
    LrtVerisiliconOptions options,
    unsigned int core_index);

LiteRtStatus LrtVerisiliconOptionsGetCoreIndex(
    const LrtVerisiliconOptions options,
    unsigned int* core_index);

//Specify milliseconds time out of network.
LiteRtStatus LrtVerisiliconOptionsSetTimeOut(
    const LrtVerisiliconOptions options,
    unsigned int time);

LiteRtStatus LrtVerisiliconOptionsGetTimeOut(
    const LrtVerisiliconOptions options,
    unsigned int* time);

/* 0: disable
   1: enable execute operations(commands) one by one
   2: enable execute operations(commands) one by one and show more log
*/
LiteRtStatus LrtVerisiliconOptionsSetProfileLevel(
    const LrtVerisiliconOptions options,
    unsigned int level);

LiteRtStatus LrtVerisiliconOptionsGetProfileLevel(
    const LrtVerisiliconOptions options,
    unsigned int* level);

/*dump NBG resource(nbg, input, output)*/
LiteRtStatus LrtVerisiliconOptionsSetDumpNBG(
    const LrtVerisiliconOptions options,
    bool enable);

LiteRtStatus LrtVerisiliconOptionsGetDumpNBG(
    const LrtVerisiliconOptions options,
    bool* enable);


#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_VERISILICON_OPTIONS_H_
