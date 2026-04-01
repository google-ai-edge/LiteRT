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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_STATIC_ACCELERATOR_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_STATIC_ACCELERATOR_REGISTRY_H_

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/litert_common.h"

class LiteRtEnvironmentT;

#ifdef __cplusplus
extern "C" {
#endif

// Externally defined CPU (XNNPack) accelerator definition.
#if defined(LITERT_USE_XNNPACK)
extern LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef;
#endif  // defined(LITERT_USE_XNNPACK)

// Define a data pointer to an accelerator definition. This pointer is updated
// by statically linked GPU accelerator.
extern LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorGpuDef;

// Define a function pointer for the WebNN accelerator.
extern LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorWebNn)(
    LiteRtEnvironmentT& environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_STATIC_ACCELERATOR_REGISTRY_H_
