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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_TYPES_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtHookTypeRuntimeStart = 0,
  kLiteRtHookTypeRuntimeStop = 1,
  kLiteRtHookTypeCompilerStart = 2,
  kLiteRtHookTypeCompilerStop = 3,
  kLiteRtHookTypeStopAndProcess = 4,
} LiteRtHookType;

// Callback signature for hooks.
// data: pointer to the trace or debug data buffer.
// size: size of the data buffer.
// user_data: user-provided context.
typedef void (*LiteRtHook)(LiteRtHookType type, const void* data, size_t size,
                           void* user_data);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_TYPES_H_
