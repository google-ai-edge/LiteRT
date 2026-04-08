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

#include "litert/runtime/accelerators/cpu_registry.h"

#if !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#endif

#include "absl/base/attributes.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/runtime/accelerators/registration_helper.h"

extern "C" {

// Weak declaration: resolved to NULL if not linked (on ELF)
// or to a fallback implementation (on Windows).
ABSL_ATTRIBUTE_WEAK LiteRtStatus LiteRtRegisterStaticLinkedAcceleratorCpu(
    LiteRtEnvironment environment);

#if defined(__APPLE__)
// macOS ld64 does not support undefined weak symbols for data pointers,
// and while these are functions, providing a local weak definition ensures
// the symbol is always defined while still allowing strong override.
ABSL_ATTRIBUTE_WEAK LiteRtStatus LiteRtRegisterStaticLinkedAcceleratorCpu(
    LiteRtEnvironment environment) {
  return kLiteRtStatusErrorUnsupported;
}
#endif

#if defined(LITERT_WINDOWS_OS)
LiteRtStatus LiteRtRegisterStaticLinkedAcceleratorCpuFallback(
    LiteRtEnvironment environment) {
  return kLiteRtStatusErrorUnsupported;
}
// Linker redirection: use Fallback if the primary symbol is missing.
#pragma comment(linker, "/alternatename:LiteRtRegisterStaticLinkedAcceleratorCpu=LiteRtRegisterStaticLinkedAcceleratorCpuFallback")
#endif

}  // extern "C"

namespace litert::internal {

LiteRtStatus LiteRtRegisterCpuAccelerator(LiteRtEnvironment environment) {
  LiteRtStatus (*reg_func)(LiteRtEnvironment) = nullptr;

#if !defined(LITERT_WINDOWS_OS) && !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
  // Layer 1: Dynamic Discovery (Linux/Android)
  // Handles the shadowing problem by searching the entire symbol space.
  reg_func = (LiteRtStatus (*)(LiteRtEnvironment))dlsym(
      RTLD_DEFAULT, "LiteRtRegisterStaticLinkedAcceleratorCpu");
#endif

  // Layer 2: Static Discovery (Weak Symbol Check)
  if (reg_func == nullptr) {
    void* volatile addr = (void*)&LiteRtRegisterStaticLinkedAcceleratorCpu;
    if (addr != nullptr) {
      reg_func = LiteRtRegisterStaticLinkedAcceleratorCpu;
    }
  }

  if (reg_func != nullptr) {
    auto status = reg_func(environment);
    if (status == kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "CPU accelerator registered.");
    } else if (status != kLiteRtStatusErrorUnsupported) {
      LITERT_LOG(LITERT_WARNING,
                 "CPU accelerator could not be loaded and registered: %s.",
                 LiteRtGetStatusString(status));
    }
    return status;
  }

  LITERT_LOG(LITERT_VERBOSE, "CPU accelerator is disabled.");
  return kLiteRtStatusErrorUnsupported;
}

}  // namespace litert::internal
