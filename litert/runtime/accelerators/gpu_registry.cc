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

#include "litert/runtime/accelerators/gpu_registry.h"

#include <filesystem>  // NOLINT
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/gpu_static_registry.h"
#include "litert/runtime/accelerators/registration_helper.h"

extern "C" {
LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorGpuDef = nullptr;
}

namespace litert::internal {

#if defined(LITERT_WINDOWS_OS)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif

// On Apple targets the LiteRT accelerator dylibs are bundled as
// ``<X>.framework/<X>`` (iOS) or ``<X>.framework/Versions/A/<X>`` (macOS),
// not as flat ``lib<X>.dylib`` files. Apple App Store Connect rejects
// builds that drop loose ``lib<X>.dylib`` symlinks alongside frameworks
// (ITMS-90432: "Frameworks must be embedded as a .framework bundle"), so
// hardcoding the basename here breaks App Store distribution. Use the
// framework-relative path on Apple and keep the flat basename elsewhere.
#if defined(__APPLE__) && TARGET_OS_OSX
#define LITERT_METAL_ACCEL_LIB \
  "LiteRtMetalAccelerator.framework/Versions/A/LiteRtMetalAccelerator"
#elif defined(__APPLE__) && TARGET_OS_IPHONE
#define LITERT_METAL_ACCEL_LIB \
  "LiteRtMetalAccelerator.framework/LiteRtMetalAccelerator"
#else
#define LITERT_METAL_ACCEL_LIB "libLiteRtMetalAccelerator" SO_EXT
#endif

LiteRtStatus RegisterGpuAccelerator(LiteRtEnvironment environment) {
#if !defined(LITERT_DISABLE_GPU)
  static constexpr absl::string_view kGpuAcceleratorLibs[] = {
      "libLiteRtGpuAccelerator" SO_EXT,
#ifdef __ANDROID__
#if LITERT_HAS_OPENCL_SUPPORT
      "libLiteRtClGlAccelerator" SO_EXT,   "libLiteRtOpenClAccelerator" SO_EXT,
#endif  // LITERT_HAS_OPENCL_SUPPORT
#if LITERT_HAS_WEBGPU_SUPPORT
      "libLiteRtWebGpuAccelerator" SO_EXT,
#endif  // LITERT_HAS_WEBGPU_SUPPORT

#elif TARGET_OS_IPHONE
#if LITERT_HAS_METAL_SUPPORT
      LITERT_METAL_ACCEL_LIB,
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
      LITERT_METAL_ACCEL_LIB,
#endif  // LITERT_HAS_METAL_SUPPORT
#endif  // !__ANDROID__ && !TARGET_OS_IPHONE

#if LITERT_HAS_VULKAN_SUPPORT
      "libLiteRtVulkanAccelerator" SO_EXT,
#endif  // LITERT_HAS_VULKAN_SUPPORT
  };

  bool gpu_accelerator_registered = false;
  if (LiteRtStaticLinkedAcceleratorGpuDef != nullptr &&
      RegisterAcceleratorFromDef(environment,
                                 LiteRtStaticLinkedAcceleratorGpuDef) ==
          kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Statically linked GPU accelerator registered.");
    gpu_accelerator_registered = true;
  }

  if (!gpu_accelerator_registered) {
    std::filesystem::path runtime_lib_path;
    auto option = environment->GetOption(kLiteRtEnvOptionTagRuntimeLibraryDir);
    if (option.has_value() && option->type == kLiteRtAnyTypeString) {
      runtime_lib_path = option->str_value;
    }
    for (auto plugin_path : kGpuAcceleratorLibs) {
      std::filesystem::path full_plugin_path =
          std::filesystem::path(runtime_lib_path) / std::string(plugin_path);
      LITERT_LOG(LITERT_INFO, "Loading GPU accelerator(%s).",
                 full_plugin_path.c_str());
      // Try to load a GPU accelerator using `LiteRtAcceleratorImpl` symbol.
      auto registration = RegisterSharedObjectAcceleratorViaAcceleratorDef(
          *environment, full_plugin_path.string(), "LiteRtAcceleratorImpl",
          /*try_default_on_failure=*/false);
      if (registration.HasValue()) {
        LITERT_LOG(LITERT_INFO,
                   "Dynamically loaded GPU accelerator(%s) registered.",
                   plugin_path.data());
        gpu_accelerator_registered = true;
        break;
      }
      // Try to load a GPU accelerator using `LiteRtRegisterGpuAccelerator`
      // symbol.
      registration = RegisterSharedObjectAcceleratorViaFunctionPointer(
          *environment, full_plugin_path.string(),
          "LiteRtRegisterGpuAccelerator",
          /*try_default_on_failure=*/true);
      if (registration.HasValue()) {
        LITERT_LOG(LITERT_INFO,
                   "Dynamically loaded GPU accelerator(%s) registered.",
                   plugin_path.data());
        gpu_accelerator_registered = true;
        break;
      }
    }
  }

  if (!gpu_accelerator_registered) {
    LITERT_LOG(LITERT_WARNING,
               "GPU accelerator could not be loaded and registered.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
#else
  LITERT_LOG(LITERT_VERBOSE, "GPU accelerator registration disabled.");
  return kLiteRtStatusErrorUnsupported;
#endif
}

}  // namespace litert::internal
