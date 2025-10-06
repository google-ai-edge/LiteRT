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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_API_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_API_H_

#include <cstddef>

#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

// A collection of Accelerator APIs that are used by LiteRT Accelerators to
// register them to LiteRT Runtime.

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t LiteRtAcceleratorId;

typedef struct LiteRtAcceleratorApi {
  // Creates an empty accelerator handle.
  LiteRtStatus (*LiteRtCreateAccelerator)(  // NOLINT
      LiteRtAccelerator* accelerator);
  // Destroys an accelerator handle.
  //
  // Warning: This SHOULD NOT BE CALLED after a call to
  // `LiteRtRegisterAccelerator`.
  LiteRtStatus (*LiteRtDestroyAccelerator)(  // NOLINT
      LiteRtAccelerator accelerator);

  // Sets the registration data AND clean-up function, then registers the
  // accelerator with the LiteRT environment.
  //
  // - `data` and `ReleaseData` may be null.
  //
  // Note: After this function returns successfully, `data` is managed by the
  // LiteRT environment. `ReleaseData` is called to release its memory.
  //
  // Warning: In case of failure, `accelerator` is released and `data` is
  // released using `ReleaseData`.
  LiteRtStatus (*LiteRtRegisterAccelerator)(  // NOLINT
      LiteRtEnvironment environment, LiteRtAccelerator accelerator, void* data,
      void (*ReleaseData)(void*));

  // Sets the function used to retrieve the accelerator name.
  LiteRtStatus (*LiteRtSetAcceleratorGetName)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*GetName)(LiteRtAccelerator accelerator,
                              const char** name));

  // Sets the function used to retrieve the accelerator implementation version.
  //
  // Note: This is NOT the LiteRT version. It's the accelerator specific
  // software implementation version.
  LiteRtStatus (*LiteRtSetAcceleratorGetVersion)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*GetVersion)(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version));

  // Sets the function used to retrieve the accelerator hardware support.
  LiteRtStatus (*LiteRtSetAcceleratorGetHardwareSupport)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*GetHardwareSupport)(
          LiteRtAccelerator accelerator,
          LiteRtHwAcceleratorSet* supported_hardware));

  // Sets the function used to return a Delegate to apply the accelerator by the
  // compiled model and its destructor. The returned Delegate object is owned by
  // the compiled model.
  LiteRtStatus (*LiteRtSetDelegateFunction)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*CreateDelegate)(LiteRtAccelerator accelerator,
                                     LiteRtOptions options,
                                     LiteRtDelegateWrapper* delegate),
      void (*DestroyDelegate)(LiteRtDelegateWrapper delegate));  // NOLINT

  // Sets the function used to surface whether the delegate created by the
  // accelerator does JIT compilation or not.
  //
  // This affects whether the compiled model creation will apply the accelerator
  // without an explicit request in the JIT compilation options.
  //
  // If this isn't set, the result will be treated as `false`.
  LiteRtStatus (
      *LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*IsTfLiteDelegateResponsibleForJitCompilation)(
          LiteRtAccelerator accelerator, bool* does_jit_compilation));

  // Sets the function used to start collection of HW-specific metrics at a
  // specific level of detail (>= 0).
  LiteRtStatus (*LiteRtSetAcceleratorStartMetricsCollection)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*StartMetricsCollection)(LiteRtDelegateWrapper delegate,
                                             int detail_level));

  // Sets the function used to stop collection of HW-specific metrics and report
  // the collected metrics.
  LiteRtStatus (*LiteRtSetAcceleratorStopMetricsCollection)(  // NOLINT
      LiteRtAccelerator accelerator,
      LiteRtStatus (*StopMetricsCollection)(LiteRtDelegateWrapper delegate,
                                            LiteRtMetrics metrics));

  // Registers custom tensor buffer handlers for the given buffer type.
  LiteRtStatus (*LiteRtRegisterTensorBufferHandlers)(  // NOLINT
      LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
      CreateCustomTensorBuffer create_func,
      DestroyCustomTensorBuffer destroy_func, LockCustomTensorBuffer lock_func,
      UnlockCustomTensorBuffer unlock_func,
      ImportCustomTensorBuffer import_func);
} LiteRtAcceleratorContext;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_INTERNAL_LITERT_ACCELERATOR_API_H_
