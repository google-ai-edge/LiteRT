// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_RUNTIME_OPTIONS_H_
#define ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_RUNTIME_OPTIONS_H_

#include <stdbool.h>
#include <stdint.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtDarwinnRuntimeOptions);

// Creates DarwiNN runtime options and adds them to the opaque options list.
LiteRtStatus LiteRtCreateDarwinnRuntimeOptions(LiteRtOpaqueOptions* options);

// Finds DarwiNN runtime options in the opaque options list.
LiteRtStatus LiteRtFindDarwinnRuntimeOptions(
    LiteRtOpaqueOptions opaque_options,
    LiteRtDarwinnRuntimeOptions* runtime_options);

// Gets the identifier for DarwiNN runtime options.
const char* LiteRtGetDarwinnRuntimeOptionsIdentifier();

// Power management setters/getters
LiteRtStatus LiteRtSetDarwinnInferencePowerState(
    LiteRtDarwinnRuntimeOptions options, uint32_t power_state);
LiteRtStatus LiteRtGetDarwinnInferencePowerState(
    LiteRtDarwinnRuntimeOptionsConst options, uint32_t* power_state);

LiteRtStatus LiteRtSetDarwinnInferenceMemoryPowerState(
    LiteRtDarwinnRuntimeOptions options, uint32_t memory_power_state);
LiteRtStatus LiteRtGetDarwinnInferenceMemoryPowerState(
    LiteRtDarwinnRuntimeOptionsConst options, uint32_t* memory_power_state);

// Scheduling setters/getters
LiteRtStatus LiteRtSetDarwinnInferencePriority(
    LiteRtDarwinnRuntimeOptions options, int8_t priority);
LiteRtStatus LiteRtGetDarwinnInferencePriority(
    LiteRtDarwinnRuntimeOptionsConst options, int8_t* priority);

LiteRtStatus LiteRtSetDarwinnAtomicInference(
    LiteRtDarwinnRuntimeOptions options, bool atomic_inference);
LiteRtStatus LiteRtGetDarwinnAtomicInference(
    LiteRtDarwinnRuntimeOptionsConst options, bool* atomic_inference);

// Memory coherency preference setter/getter
LiteRtStatus LiteRtSetDarwinnPreferCoherent(
    LiteRtDarwinnRuntimeOptions options, bool prefer_coherent);
LiteRtStatus LiteRtGetDarwinnPreferCoherent(
    LiteRtDarwinnRuntimeOptionsConst options, bool* prefer_coherent);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_OPTIONS_LITERT_DARWINN_RUNTIME_OPTIONS_H_
