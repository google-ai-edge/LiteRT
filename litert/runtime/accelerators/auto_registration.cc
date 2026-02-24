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

#include <cstddef>
#include <filesystem>  // NOLINT
#include <string>
#include <utility>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/runtime/accelerators/dispatch/dispatch_accelerator.h"
#endif  // !defined(LITERT_DISABLE_NPU)

extern "C" {

// Define a data pointer to an accelerator definition. This pointer is updated
// by statically linked CPU (XNNPack) accelerator.
LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorCpuDef = nullptr;

// Define a data pointer to an accelerator definition. This pointer is updated
// by statically linked GPU accelerator.
LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorGpuDef = nullptr;

// Define a function pointer for the WebNN accelerator.
LiteRtStatus (*LiteRtRegisterStaticLinkedAcceleratorWebNn)(
    LiteRtEnvironmentT& environment) = nullptr;

}  // extern "C"

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

// Registers an accelerator and tensor buffer handlers using the provided
// accelerator definition.
LiteRtStatus RegisterAccelerator(LiteRtEnvironment env,
                                 LiteRtAcceleratorDef* accelerator_def) {
  if (accelerator_def->version != LITERT_ACCELERATOR_DEF_CURRENT_VERSION)
    return kLiteRtStatusErrorWrongVersion;

  if (accelerator_def->get_name == nullptr ||
      accelerator_def->get_version == nullptr ||
      accelerator_def->get_hardware_support == nullptr ||
      accelerator_def->is_tflite_delegate_responsible_for_jit_compilation ==
          nullptr ||
      accelerator_def->create_delegate == nullptr ||
      accelerator_def->destroy_delegate == nullptr ||
      accelerator_def->num_supported_buffer_types >=
          LITERT_ACCELERATOR_DEF_MAX_SUPPORTED_BUFFER_TYPES) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  LiteRtAccelerator accelerator;
  LITERT_RETURN_IF_ERROR(LiteRtCreateAccelerator(&accelerator));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetAcceleratorGetName(accelerator, accelerator_def->get_name));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetVersion(
      accelerator, accelerator_def->get_version));
  LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
      accelerator, accelerator_def->get_hardware_support));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDelegateFunction(accelerator, accelerator_def->create_delegate,
                                accelerator_def->destroy_delegate));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetIsAcceleratorDelegateResponsibleForJitCompilation(
          accelerator,
          accelerator_def->is_tflite_delegate_responsible_for_jit_compilation));

  LITERT_RETURN_IF_ERROR(
      LiteRtRegisterAccelerator(env, accelerator, nullptr, nullptr));

  for (size_t i = 0; i < accelerator_def->num_supported_buffer_types; ++i) {
    LITERT_RETURN_IF_ERROR(LiteRtRegisterTensorBufferHandlers(
        env, accelerator_def->supported_buffer_types[i],
        accelerator_def->create_func, accelerator_def->destroy_func,
        accelerator_def->lock_func, accelerator_def->unlock_func,
        accelerator_def->clear_func, accelerator_def->import_func));
  }

  return kLiteRtStatusOk;
}

// Load a shared library and call the provided registration function.
// This is old method to register an accelerator.
// Warning: This flow should be removed in the future.
Expected<void> RegisterSharedObjectAcceleratorViaFunctionPointer(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view registration_function_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto registration_function,
      shlib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
          registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(&environment));
  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

// Load a shared library and register an accelerator using the provided
// definition object.
Expected<void> RegisterSharedObjectAcceleratorViaAcceleratorDef(
    LiteRtEnvironmentT& environment, absl::string_view shlib_path,
    absl::string_view accelerator_def_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto accelerator_def,
      shlib.LookupSymbol<LiteRtAcceleratorDef*>(accelerator_def_name.data()));

  LITERT_RETURN_IF_ERROR(RegisterAccelerator(&environment, accelerator_def));

  environment.GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
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
      "libLiteRtClGlAccelerator" SO_EXT,
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
  if (LiteRtStaticLinkedAcceleratorGpuDef != nullptr &&
      RegisterAccelerator(&environment, LiteRtStaticLinkedAcceleratorGpuDef) ==
          kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Statically linked GPU accelerator registered.");
    gpu_accelerator_registered = true;
  }
  if (!gpu_accelerator_registered) {
    std::filesystem::path runtime_lib_path;
    auto option = environment.GetOption(kLiteRtEnvOptionTagRuntimeLibraryDir);
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
          environment, full_plugin_path.string(), "LiteRtAcceleratorImpl",
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
          environment, full_plugin_path.string(),
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
  }
#else
  LITERT_LOG(LITERT_VERBOSE, "GPU accelerator registration disabled.");
#endif

  // Register the CPU accelerator.
  if (LiteRtStaticLinkedAcceleratorCpuDef != nullptr) {
    if (auto status = RegisterAccelerator(&environment,
                                          LiteRtStaticLinkedAcceleratorCpuDef);
        status == kLiteRtStatusOk) {
      LITERT_LOG(LITERT_INFO, "CPU accelerator registered.");
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "CPU accelerator could not be loaded and registered: %s.",
                 LiteRtGetStatusString(status));
    }
  }

  return {};
};

}  // namespace litert
