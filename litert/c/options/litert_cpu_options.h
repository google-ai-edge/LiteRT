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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif

LITERT_DEFINE_HANDLE(LiteRtCpuOptions);

// Creates an opaque options object holding CPU options.
LiteRtStatus LiteRtCreateCpuOptions(LiteRtOpaqueOptions* options);

// Gets the underlying CPU options from an opaque options handle.
LiteRtStatus LiteRtFindCpuOptions(LiteRtOpaqueOptions opaque_options,
                                  LiteRtCpuOptions* cpu_options);

// Gets the identifier for CPU options stored in opaque options.
const char* LiteRtGetCpuOptionsIdentifier();

// Sets the number of CPU threads used by the CPU accelerator.
LiteRtStatus LiteRtSetCpuOptionsNumThread(LiteRtCpuOptions options,
                                          int num_threads);

// Gets the number of CPU threads used by the CPU accelerator.
LiteRtStatus LiteRtGetCpuOptionsNumThread(LiteRtCpuOptionsConst options,
                                          int* num_threads);

// Sets the XNNPack flags used by the CPU accelerator.
LiteRtStatus LiteRtSetCpuOptionsXNNPackFlags(LiteRtCpuOptions options,
                                             int flags);

// Gets the XNNPack flags used by the CPU accelerator.
LiteRtStatus LiteRtGetCpuOptionsXNNPackFlags(LiteRtCpuOptionsConst options,
                                             int* flags);

// Sets the XNNPack weight cache file path used by the CPU accelerator.
LiteRtStatus LiteRtSetCpuOptionsXnnPackWeightCachePath(LiteRtCpuOptions options,
                                                       const char* path);

// Gets the XNNPack weight cache file path used by the CPU accelerator.
LiteRtStatus LiteRtGetCpuOptionsXnnPackWeightCachePath(
    LiteRtCpuOptionsConst options, const char** path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_CPU_OPTIONS_H_
