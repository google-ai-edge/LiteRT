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

// Metal information handle type. This is used to hold the metal device.
// It's created by LiteRtCreateMetalInfo and LiteRtCreateWithDevice and
// destroyed by LiteRtDeleteMetalInfo.
typedef void* MetalInfoHandle;

// Custom metal information data.
// Gpu environment can use this to keep track of the metal device.
// The information is kept in the child struct of `MetalInfoImpl`.
// It's created by LiteRtCreateMetalInfo and LiteRtCreateWithDevice and
// destroyed by LiteRtDeleteMetalInfo.
typedef struct MetalInfo {
  virtual ~MetalInfo() = default;
  MetalInfoHandle metal_info;
  MetalInfoHandle metal_command_queue;
} MetalInfo;

typedef MetalInfo* MetalInfoPtr;

// Creates Metal info struct to be used for holding Metal device.
// `metal_info` could not be a nullptr.
LiteRtStatus LiteRtCreateMetalInfo(MetalInfoPtr* metal_info);

// Creates Metal info struct to be used for holding Metal device with given
// device.
// metal_info could not be a nullptr. The device should be a valid pointer to a
// MTLDevice. If device is nullptr, underlying id<MTLDevice> will be nullptr.
LiteRtStatus LiteRtCreateWithDevice(void* device, MetalInfoPtr* metal_info);

// Deletes the MetalInfo structure and releases any associated Metal resources.
//
// This function deallocates the memory occupied by the structure pointed to
// by metal_info. It's safe to pass nullptr to this function, in which case
// the function has no effect.
void LiteRtDeleteMetalInfo(MetalInfoPtr metal_info);

LiteRtStatus LiteRtCreateWithCommandQueue(void* command_queue, void* device,
                                          MetalInfoPtr* metal_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_RUNTIME_METAL_INFO_H_
