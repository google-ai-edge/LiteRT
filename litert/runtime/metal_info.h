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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_INFO_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_INFO_H_

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void* MetalInfoHandle;

typedef struct MetalInfo {
  MetalInfoHandle metal_info;
} MetalInfo;

typedef MetalInfo* MetalInfoPtr;

// Creates Metal info struct to be used for holding Metal device.
LiteRtStatus LiteRtCreateMetalInfo(MetalInfoPtr* metal_info);

// Creates Metal info struct to be used for holding Metal device with given
// device.
LiteRtStatus LiteRtCreateWithDevice(void* device, MetalInfoPtr* metal_info);

// Deletes Metal info struct.
void LiteRtDeleteMetalInfo(MetalInfoPtr metal_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_INFO_H_
