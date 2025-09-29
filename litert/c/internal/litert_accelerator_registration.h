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

#ifndef ODML_LITERT_LITERT_C_LITERT_INTERNAL_ACCELERATOR_REGISTRATION_H_
#define ODML_LITERT_LITERT_C_LITERT_INTERNAL_ACCELERATOR_REGISTRATION_H_

#include <stdbool.h>  // NOLINT

#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an empty accelerator handle.
LiteRtStatus LiteRtCreateAccelerator(LiteRtAccelerator* accelerator);

// Destroys an accelerator handle.
//
// Warning: This SHOULD NOT BE CALLED after a call to
// `LiteRtRegisterAccelerator`.
LiteRtStatus LiteRtDestroyAccelerator(LiteRtAccelerator accelerator);

// Sets the registration data AND clean-up function, then registers the
// accelerator with the LiteRT environment.
//
// - `data` and `ReleaseData` may be null.
//
// Note: After this function returns successfully, `data` is managed by the
// LiteRT environment. `ReleaseData` is called to release its memory.
//
// Warning: In case of failure, `accelerator` is released and `data` is released
// using `ReleaseData`.
LiteRtStatus LiteRtRegisterAccelerator(LiteRtEnvironment environment,
                                       LiteRtAccelerator accelerator,
                                       void* data, void (*ReleaseData)(void*));

// Sets the function used to retrieve the accelerator name.
LiteRtStatus LiteRtSetAcceleratorGetName(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetName)(LiteRtAccelerator accelerator, const char** name));

// Sets the function used to retrieve the accelerator implementation version.
//
// Note: This is NOT the LiteRT version. It's the accelerator specific software
// implementation version.
LiteRtStatus LiteRtSetAcceleratorGetVersion(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetVersion)(LiteRtAccelerator accelerator,
                               LiteRtApiVersion* version));

// Sets the function used to retrieve the accelerator hardware support.
LiteRtStatus LiteRtSetAcceleratorGetHardwareSupport(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetHardwareSupport)(
        LiteRtAccelerator accelerator,
        LiteRtHwAcceleratorSet* supported_hardware));

// Sets the function used to return a Delegate to apply the accelerator by the
// compiled model and its destructor. The returned Delegate object is owned by
// the compiled model.
LiteRtStatus LiteRtSetDelegateFunction(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*CreateDelegate)(LiteRtAccelerator accelerator,
                                   LiteRtOptions options,
                                   LiteRtDelegateWrapper* delegate),
    void (*DestroyDelegate)(LiteRtDelegateWrapper delegate));

// Sets the function used to surface whether the delegate created by the
// accelerator does JIT compilation or not.
//
// This affects whether the compiled model creation will apply the accelerator
// without an explicit request in the JIT compilation options.
//
// If this isn't set, the result will be treated as `false`.
LiteRtStatus LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*IsTfLiteDelegateResponsibleForJitCompilation)(
        LiteRtAccelerator accelerator, bool* does_jit_compilation));

// Sets the function used to start collection of HW-specific metrics at a
// specific level of detail (>= 0).
LiteRtStatus LiteRtSetAcceleratorStartMetricsCollection(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*StartMetricsCollection)(LiteRtDelegateWrapper delegate,
                                           int detail_level));

// Sets the function used to stop collection of HW-specific metrics and report
// the collected metrics.
LiteRtStatus LiteRtSetAcceleratorStopMetricsCollection(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*StopMetricsCollection)(LiteRtDelegateWrapper delegate,
                                          LiteRtMetrics metrics));

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ODML_LITERT_LITERT_C_LITERT_INTERNAL_ACCELERATOR_REGISTRATION_H_
