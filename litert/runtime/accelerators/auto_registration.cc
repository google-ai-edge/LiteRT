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

#include "litert/runtime/accelerators/auto_registration.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/cpu_registry.h"
#include "litert/runtime/accelerators/gpu_registry.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"
#endif  // !defined(LITERT_DISABLE_NPU)

extern "C" {

// Weak declaration for WebNN accelerator.
#if defined(__EMSCRIPTEN__)
ABSL_ATTRIBUTE_WEAK LiteRtStatus LiteRtRegisterStaticLinkedAcceleratorWebNn(
    LiteRtEnvironmentT& environment);
#endif  // defined(__EMSCRIPTEN__)

#if defined(__APPLE__)
// macOS ld64 does not support undefined weak symbols for data pointers,
// and while these are functions, providing a local weak definition ensures
// the symbol is always defined while still allowing strong override.
ABSL_ATTRIBUTE_WEAK LiteRtStatus LiteRtRegisterStaticLinkedAcceleratorWebNn(
    LiteRtEnvironmentT& environment) {
  return kLiteRtStatusErrorUnsupported;
}
#endif

}  // extern "C"

namespace litert {
namespace {

constexpr LiteRtHwAcceleratorSet kDefaultAutoRegisterAccelerators =
    kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorNpu
#if defined(__EMSCRIPTEN__)
    | kLiteRtHwAcceleratorWebNn
#endif  // defined(__EMSCRIPTEN__)
    ;

LiteRtHwAcceleratorSet GetAutoRegisterAccelerators(
    const LiteRtEnvironmentT& environment) {
  auto option =
      environment.GetOption(kLiteRtEnvOptionTagAutoRegisterAccelerators);
  if (!option.has_value()) {
    return kDefaultAutoRegisterAccelerators;
  }
  if (option->type != kLiteRtAnyTypeInt) {
    LITERT_LOG(LITERT_WARNING,
               "Auto-register accelerators option must be an integer bitmask. "
               "Using default accelerator auto-registration.");
    return kDefaultAutoRegisterAccelerators;
  }
  return static_cast<LiteRtHwAcceleratorSet>(option->int_value);
}

}  // namespace

Expected<void> TriggerAcceleratorAutomaticRegistration(
    LiteRtEnvironmentT& environment) {
  const LiteRtHwAcceleratorSet auto_register_accelerators =
      GetAutoRegisterAccelerators(environment);
  // Register the NPU accelerator.
#if !defined(LITERT_DISABLE_NPU)
  if (auto_register_accelerators & kLiteRtHwAcceleratorNpu) {
    if (auto npu_registration = LiteRtRegisterNpuAccelerator(&environment);
        npu_registration == kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "NPU accelerator registered.");
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "NPU accelerator could not be loaded and registered: %s.",
                 LiteRtGetStatusString(npu_registration));
    }
  } else {
    LITERT_LOG(LITERT_VERBOSE,
               "NPU accelerator registration skipped by environment options.");
  }
#else
  LITERT_LOG(LITERT_VERBOSE, "NPU accelerator accelerator is disabled.");
#endif

  // Register the WebNN accelerator if statically linked.
#if defined(__EMSCRIPTEN__)
  if (auto_register_accelerators & kLiteRtHwAcceleratorWebNn) {
    void* volatile addr = (void*)&LiteRtRegisterStaticLinkedAcceleratorWebNn;
    if (addr != nullptr &&
        LiteRtRegisterStaticLinkedAcceleratorWebNn(environment) ==
            kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "Statically linked WebNN accelerator registered.");
    }
  }
#endif  // defined(__EMSCRIPTEN__)

  if (auto_register_accelerators & kLiteRtHwAcceleratorGpu) {
    litert::internal::LiteRtRegisterGpuAccelerator(&environment);
  } else {
    LITERT_LOG(LITERT_VERBOSE,
               "GPU accelerator registration skipped by environment options.");
  }

  // Register the CPU accelerator.
  if (auto_register_accelerators & kLiteRtHwAcceleratorCpu) {
    litert::internal::LiteRtRegisterCpuAccelerator(&environment);
  } else {
    LITERT_LOG(LITERT_VERBOSE,
               "CPU accelerator registration skipped by environment options.");
  }

  return {};
};

}  // namespace litert
