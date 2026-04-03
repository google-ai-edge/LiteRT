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
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/environment.h"
#include "litert/runtime/accelerators/registration_helper.h"

extern "C" {

// Define a data pointer to an accelerator definition. This pointer is updated
// by statically linked GPU accelerator.
LiteRtAcceleratorDef* LiteRtStaticLinkedAcceleratorGpuDef = nullptr;

}  // extern "C"

namespace {

using ::litert::Expected;
using ::litert::RtldFlags;
using ::litert::SharedLibrary;

Expected<SharedLibrary> LoadSharedLibrary(absl::string_view shlib_path,
                                          bool try_default_on_failure) {
  auto result = SharedLibrary::Load(shlib_path, RtldFlags::Lazy().Local());
  if (result || !try_default_on_failure) {
    return result;
  }
  return SharedLibrary::Load(RtldFlags::kDefault);
}

// Load a shared library and call the provided registration function.
// This is old method to register an accelerator.
// Warning: This flow should be removed in the future.
Expected<void> RegisterSharedObjectAcceleratorViaFunctionPointer(
    LiteRtEnvironment environment, absl::string_view shlib_path,
    absl::string_view registration_function_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto registration_function,
      shlib.LookupSymbol<LiteRtStatus (*)(LiteRtEnvironment)>(
          registration_function_name.data()));
  LITERT_RETURN_IF_ERROR(registration_function(environment));
  environment->GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

// Load a shared library and register an accelerator using the provided
// definition object.
Expected<void> RegisterSharedObjectAcceleratorViaAcceleratorDef(
    LiteRtEnvironment environment, absl::string_view shlib_path,
    absl::string_view accelerator_def_name, bool try_default_on_failure) {
  LITERT_ASSIGN_OR_RETURN(
      auto shlib, LoadSharedLibrary(shlib_path, try_default_on_failure));
  LITERT_ASSIGN_OR_RETURN(
      auto accelerator_def,
      shlib.LookupSymbol<LiteRtAcceleratorDef*>(accelerator_def_name.data()));

  LITERT_RETURN_IF_ERROR(::litert::internal::RegisterAcceleratorFromDef(
      environment, accelerator_def));

  environment->GetAcceleratorRegistry().TakeOwnershipOfSharedLibrary(
      std::move(shlib));
  return {};
}

}  // namespace

namespace litert::internal {

#if defined(LITERT_WINDOWS_OS)
#define SO_EXT ".dll"
#elif defined(__APPLE__)
#define SO_EXT ".dylib"
#else
#define SO_EXT ".so"
#endif

LiteRtStatus LiteRtRegisterGpuAccelerator(LiteRtEnvironment environment) {
#if defined(LITERT_DISABLE_GPU)
  LITERT_LOG(LITERT_VERBOSE, "GPU accelerator is disabled.");
  return kLiteRtStatusErrorUnsupported;
#else
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

  if (LiteRtStaticLinkedAcceleratorGpuDef != nullptr &&
      ::litert::internal::RegisterAcceleratorFromDef(
          environment, LiteRtStaticLinkedAcceleratorGpuDef) ==
          kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Statically linked GPU accelerator registered.");
    return kLiteRtStatusOk;
  }

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
        environment, full_plugin_path.string(), "LiteRtAcceleratorImpl",
        /*try_default_on_failure=*/false);
    if (registration.HasValue()) {
      LITERT_LOG(LITERT_INFO,
                 "Dynamically loaded GPU accelerator(%s) registered.",
                 plugin_path.data());
      return kLiteRtStatusOk;
    }
    // Try to load a GPU accelerator using `LiteRtRegisterGpuAccelerator`
    // symbol.
    registration = RegisterSharedObjectAcceleratorViaFunctionPointer(
        environment, full_plugin_path.string(), "LiteRtRegisterGpuAccelerator",
        /*try_default_on_failure=*/true);
    if (registration.HasValue()) {
      LITERT_LOG(LITERT_INFO,
                 "Dynamically loaded GPU accelerator(%s) registered.",
                 plugin_path.data());
      return kLiteRtStatusOk;
    }
  }

  LITERT_LOG(LITERT_WARNING,
             "GPU accelerator could not be loaded and registered.");
  return kLiteRtStatusErrorNotFound;
#endif  // defined(LITERT_DISABLE_GPU)
}

}  // namespace litert::internal
