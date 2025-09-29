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

#ifndef ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_H_
#define ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_H_

#include "litert/c/litert_common.h"
#include "litert/runtime/metrics.h"

// We need to forward declare this to avoid a dependency loop.
struct LiteRtCompiledModelT;
struct LiteRtEnvironmentT;

struct LiteRtAcceleratorT {
  // Points to the type-erased accelerator state.
  void* data;

  // Points to the environment that owns this accelerator.
  LiteRtEnvironmentT* env;

  // NOLINTBEGIN(*-readability-class-member-naming)

  // Releases the the data.
  //
  // This function is used by the framework to clean up the accelerator. It
  // should not be called by client code.
  void (*ReleaseData)(void*);

  // Retrieves the accelerator name.
  LiteRtStatus (*GetName)(LiteRtAcceleratorT* accelerator, const char** name);

  // Retrieves the accelerator version.
  LiteRtStatus (*GetVersion)(LiteRtAcceleratorT* accelerator,
                             LiteRtApiVersion* version);

  // Retrieves the accelerator hardware support.
  LiteRtStatus (*GetHardwareSupport)(
      LiteRtAcceleratorT* accelerator,
      LiteRtHwAcceleratorSet* supported_hardware);

  // Creates a delegate for the accelerator.
  LiteRtStatus (*CreateDelegate)(LiteRtAcceleratorT* accelerator,
                                 LiteRtOptions compilation_options,
                                 LiteRtDelegateWrapper* delegate);

  // Destroys created delegate for the accelerator.
  void (*DestroyDelegate)(LiteRtDelegateWrapper delegate);

  LiteRtStatus (*IsTfLiteDelegateResponsibleForJitCompilation)(
      LiteRtAcceleratorT* accelerator, bool* does_jit_compilation);

  // Starts collection of HW-specific metrics at a specific level of detail.
  LiteRtStatus (*StartMetricsCollection)(LiteRtDelegateWrapper delegate,
                                         int detail_level);

  // Stops collection of HW-specific metrics and report the collected metrics.
  LiteRtStatus (*StopMetricsCollection)(LiteRtDelegateWrapper delegate,
                                        LiteRtMetricsT* metrics);

  // NOLINTEND(*-readability-class-member-naming)
};

#endif  // ODML_LITERT_LITERT_RUNTIME_ACCELERATOR_H_
