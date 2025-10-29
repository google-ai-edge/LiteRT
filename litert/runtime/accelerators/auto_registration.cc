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

#include <utility>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/xnnpack/xnnpack_accelerator.h"

#if !defined(LITERT_DISABLE_NPU)
#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"
#endif  // !defined(LITERT_DISABLE_NPU)

// Define a function pointer to allow the accelerator registration to be
// overridden by the LiteRT environment. This is to use the GPU accelerator
// statically linked.
extern "C" LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorGpu)(
    LiteRtEnvironmentT& environment) = nullptr;

// Define a function pointer for the WebNN accelerator.
extern "C" LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorWebNn)(
    LiteRtEnvironmentT& environment) = nullptr;

namespace litert {
namespace {

Expected<SharedLibrary> LoadSharedLibrary(absl::string_view shlib_path,
                                          bool try_default_on_failure) {
  auto result = SharedLibrary::Load(shlib_path, RtldFlags::Lazy().Local());
  if (result || !try_default_on_failure) {
    return result;
  }
  return SharedLibrary::Load(RtldFlags::kDefault);
}

}  // namespace

Expected<void> TriggerAcceleratorAutomaticRegistration(
    LiteRtEnvironmentT& environment) {
  // Register the NPU accelerator.
#if !defined(LITERT_DISABLE_NPU)
  if (auto npu_registration = LiteRtRegisterNpuAccelerator(&environment);
      npu_registration == kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "NPU accelerator registered.");
  } else {
    LITERT_LOG(LITERT_WARNING,
               "NPU accelerator could not be loaded and registered: %s.",
               LiteRtGetStatusString(npu_registration));
  }
#else
  LITERT_LOG(LITERT_VERBOSE, "NPU accelerator accelerator is disabled.");
#endif

  // Register the WebNN accelerator if statically linked.
  if (LiteRtRegisterStaticLinkedAcceleratorWebNn != nullptr &&
      LiteRtRegisterStaticLinkedAcceleratorWebNn(environment) ==
          kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Statically linked WebNN accelerator registered.");
  }

  // Register the GPU accelerator.
  // The following is list of plugins that are loaded in the order they are
  // listed. The first plugin that is loaded and registered successfully will
  // be used.
#if defined(LITERT_WINDOWS_OS)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif
#if !defined(LITERT_DISABLE_GPU)
  static constexpr absl::string_view kGpuAcceleratorLibs[] = {
      "libLiteRtGpuAccelerator" SO_EXT,

#ifdef __ANDROID__
#if LITERT_HAS_OPENCL_SUPPORT
      "libLiteRtOpenClAccelerator" SO_EXT,
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_WEBGPU_SUPPORT
      "libLiteRtWebGpuAccelerator" SO_EXT,
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#elif TARGET_OS_IPHONE
#if LITERT_HAS_METAL_SUPPORT
      "libLiteRtMetalAccelerator" SO_EXT,
#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_WEBGPU_SUPPORT
      "libLiteRtWebGpuAccelerator" SO_EXT,
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#else  // !__ANDROID__ && !TARGET_OS_IPHONE
#if LITERT_HAS_WEBGPU_SUPPORT
      "libLiteRtWebGpuAccelerator" SO_EXT,
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_OPENCL_SUPPORT
      "libLiteRtOpenClAccelerator" SO_EXT,
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
      "libLiteRtMetalAccelerator" SO_EXT,
#endif  // LITERT_HAS_METAL_SUPPORT
#endif  // !__ANDROID__ && !TARGET_OS_IPHONE

#if LITERT_HAS_VULKAN_SUPPORT
      "libLiteRtVulkanAccelerator" SO_EXT,
#endif  // LITERT_HAS_VULKAN_SUPPORT
  };
  bool gpu_accelerator_registered = false;
  for (auto plugin_path : kGpuAcceleratorLibs) {
    LITERT_LOG(LITERT_VERBOSE, "Loading GPU accelerator(%s).",
               plugin_path.data());
    auto registration = RegisterSharedObjectAccelerator(
        environment, plugin_path, "LiteRtRegisterGpuAccelerator",
        /*try_symbol_already_loaded=*/false);
    if (registration.HasValue()) {
      LITERT_LOG(LITERT_INFO,
                 "Dynamically loaded GPU accelerator(%s) registered.",
                 plugin_path.data());
      gpu_accelerator_registered = true;
      break;
    }
    const auto& error = registration.Error();
    auto log_level = absl::StrContains(error.Message(), "cannot locate symbol")
                         ? LITERT_WARNING
                         : LITERT_VERBOSE;
    LITERT_LOG(log_level, "Failed to load GPU accelerator(%s): %s, %s.",
               plugin_path.data(), LiteRtGetStatusString(error.Status()),
               registration.Error().Message().data());
  }
  if (!gpu_accelerator_registered) {
    if (LiteRtRegisterStaticLinkedAcceleratorGpu != nullptr &&
        LiteRtRegisterStaticLinkedAcceleratorGpu(environment) ==
            kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "Statically linked GPU accelerator registered.");
      gpu_accelerator_registered = true;
    }
  }
  if (!gpu_accelerator_registered) {
    LITERT_LOG(LITERT_WARNING,
               "GPU accelerator could not be loaded and registered.");
  }
#else
  LITERT_LOG(LITERT_VERBOSE, "GPU accelerator registration disabled.");
#endif

  // Register the CPU accelerator.
  if (auto cpu_registration = LiteRtRegisterCpuAccelerator(&environment);
      cpu_registration == kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "CPU accelerator registered.");
  } else {
    LITERT_LOG(LITERT_WARNING,
               "CPU accelerator could not be loaded and registered: %s.",
               LiteRtGetStatusString(cpu_registration));
  }

  return {};
};

Expected<void> RegisterSharedObjectAccelerator(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view registration_function_name,
    bool try_symbol_already_loaded) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_symbol_already_loaded));
  LITERT_ASSIGN_OR_RETURN(
      auto registration_function,
      shlib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
          registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

}  // namespace litert
