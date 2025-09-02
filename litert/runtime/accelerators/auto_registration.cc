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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_shared_library.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"
#include "litert/runtime/accelerators/xnnpack/xnnpack_accelerator.h"

// Define a function pointer to allow the accelerator registration to be
// overridden by the LiteRT environment. This is to use the GPU accelerator
// statically linked.
extern "C" LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorGpu)(
    LiteRtEnvironmentT& environment) = nullptr;

namespace litert {

Expected<void> TriggerAcceleratorAutomaticRegistration(
    LiteRtEnvironmentT& environment) {
  // Register the NPU accelerator.
  if (auto npu_registration = LiteRtRegisterNpuAccelerator(&environment);
      npu_registration == kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "NPU accelerator registered.");
  } else {
    LITERT_LOG(LITERT_WARNING,
               "NPU accelerator could not be loaded and registered: %s.",
               LiteRtGetStatusString(npu_registration));
  }

  // Register the GPU accelerator.
  // The following is list of plugins that are loaded in the order they are
  // listed. The first plugin that is loaded and registered successfully will
  // be used.
#if defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif
  static constexpr absl::string_view kGpuAcceleratorLibs[] = {
    "libLiteRtGpuAccelerator" SO_EXT,
#if LITERT_HAS_OPENCL_SUPPORT
    "libLiteRtOpenClAccelerator" SO_EXT,
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_METAL_SUPPORT
    "libLiteRtMetalAccelerator" SO_EXT,
#endif  // LITERT_HAS_METAL_SUPPORT
#if LITERT_HAS_WEBGPU_SUPPORT
    "libLiteRtWebGpuAccelerator" SO_EXT,
#endif  // LITERT_HAS_WEBGPU_SUPPORT
#if LITERT_HAS_VULKAN_SUPPORT
    "libLiteRtVulkanAccelerator" SO_EXT,
#endif  // LITERT_HAS_VULKAN_SUPPORT
  };
  bool gpu_accelerator_registered = false;
  for (auto plugin_path : kGpuAcceleratorLibs) {
    LITERT_LOG(LITERT_INFO, "Loading GPU accelerator(%s).", plugin_path.data());
    auto registration = RegisterSharedObjectAccelerator(
        environment, plugin_path, "LiteRtRegisterGpuAccelerator");
    if (registration.HasValue()) {
      LITERT_LOG(LITERT_INFO,
                 "Dynamically loaded GPU accelerator(%s) registered.",
                 plugin_path.data());
      gpu_accelerator_registered = true;
      break;
    }
    LITERT_LOG(LITERT_DEBUG, "Failed to load GPU accelerator(%s): %s, %s.",
               plugin_path.data(),
               LiteRtGetStatusString(registration.Error().Status()),
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
    LiteRtEnvironmentT& environment, absl::string_view plugin_path,
    absl::string_view registration_function_name) {
  auto maybe_lib = SharedLibrary::Load(plugin_path, RtldFlags::Lazy().Local());
  if (!maybe_lib.HasValue()) {
    LITERT_LOG(LITERT_WARNING,
               "Failed to load shared library %s: %s, %s", plugin_path.data(),
               LiteRtGetStatusString(maybe_lib.Error().Status()),
               maybe_lib.Error().Message().data());
    maybe_lib = SharedLibrary::Load(RtldFlags::kDefault);
  }
  // Note: the Load(kDefault) overload always succeeds, so we are sure that
  // maybe_lib contains a value.
  SharedLibrary lib(std::move(maybe_lib.Value()));
  LITERT_ASSIGN_OR_RETURN(auto registration_function,
                          lib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
                              registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(lib));
  return {};
}

}  // namespace litert
