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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_

#include <any>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

/// Manages configuration options for a LiteRT Environment.
///
/// This class provides methods to access various options related to the LiteRT
/// environment, such as compiler settings, dispatch libraries, and hardware
/// acceleration configurations (e.g., OpenCL, EGL, WebGPU, Metal, Vulkan).
class EnvironmentOptions
    : public internal::NonOwnedHandle<LiteRtEnvironmentOptions> {
 public:
  /// Constructs an EnvironmentOptions instance.
  /// @note EnvironmentOptions are always owned by an environment and this class
  /// only holds a non-owning handle.
  explicit EnvironmentOptions(LiteRtEnvironmentOptions env)
      : NonOwnedHandle(env) {}

  /// Tags for environment options. These tags are used to identify and retrieve
  /// specific configuration settings.
  enum class Tag : int {
    /// Directory for compiler plugin libraries.
    kCompilerPluginLibraryDir = kLiteRtEnvOptionTagCompilerPluginLibraryDir,
    /// Directory for dispatch libraries.
    kDispatchLibraryDir = kLiteRtEnvOptionTagDispatchLibraryDir,
    /// OpenCL device ID.
    kOpenClDeviceId = kLiteRtEnvOptionTagOpenClDeviceId,
    /// OpenCL platform ID.
    kOpenClPlatformId = kLiteRtEnvOptionTagOpenClPlatformId,
    /// OpenCL context.
    kOpenClContext = kLiteRtEnvOptionTagOpenClContext,
    /// OpenCL command queue.
    kOpenClCommandQueue = kLiteRtEnvOptionTagOpenClCommandQueue,
    /// EGL display.
    kEglDisplay = kLiteRtEnvOptionTagEglDisplay,
    /// EGL context.
    kEglContext = kLiteRtEnvOptionTagEglContext,
    /// WebGPU device.
    kWebGpuDevice = kLiteRtEnvOptionTagWebGpuDevice,
    /// WebGPU queue.
    kWebGpuQueue = kLiteRtEnvOptionTagWebGpuQueue,
    /// Metal device.
    kMetalDevice = kLiteRtEnvOptionTagMetalDevice,
    /// Metal command queue.
    kMetalCommandQueue = kLiteRtEnvOptionTagMetalCommandQueue,
    /// Vulkan environment (experimental).
    kVulkanEnvironment = kLiteRtEnvOptionTagVulkanEnvironment,
    /// Callback to be invoked on GPU environment destruction.
    kCallbackOnGpuEnvDestroy = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
    /// User data for the GPU environment destruction callback.
    kCallbackUserDataOnGpuEnvDestroy =
        kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
    /// Magic number configurations.
    kMagicNumberConfigs = kLiteRtEnvOptionTagMagicNumberConfigs,
    /// Magic number verifications.
    kMagicNumberVerifications = kLiteRtEnvOptionTagMagicNumberVerifications,
    /// Directory for the compiler cache.
    kCompilerCacheDir = kLiteRtEnvOptionTagCompilerCacheDir,
    /// Singleton ML Drift WebGPU/Dawn instance. Required for shared libraries
    /// to prevent them from creating their own instances.
    kWebGpuInstance = kLiteRtEnvOptionTagWebGpuInstance,
    /// Dawn procedure table pointer. This allows shared libraries to use the
    /// shared procedures instead of their own.
    kWebGpuProcs = kLiteRtEnvOptionTagWebGpuProcs,
  };

  /// Retrieves the value of an option specified by a tag.
  /// @param tag The tag of the option to retrieve.
  /// @return An `Expected` object containing the option value if successful,
  /// or an error if the option is not found or the handle is null.
  Expected<LiteRtVariant> GetOption(Tag tag) const {
    return GetOption(static_cast<LiteRtEnvOptionTag>(tag));
  }

  /// @deprecated Use `GetOption(Tag)` instead.
  [[deprecated("Use GetOption(Tag) instead.")]]
  Expected<LiteRtVariant> GetOption(LiteRtEnvOptionTag tag) const {
    if (Get() == nullptr) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Environment options are null");
    }
    LiteRtAny option;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetEnvironmentOptionsValue(Get(), tag, &option));
    return ToStdAny(option);
  }
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_
