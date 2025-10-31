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

class EnvironmentOptions
    : public internal::NonOwnedHandle<LiteRtEnvironmentOptions> {
 public:
  // EnvironmentOptions are always owned by some environment, this can never be
  // an owning handle.
  explicit EnvironmentOptions(LiteRtEnvironmentOptions env)
      : NonOwnedHandle(env) {}

  enum class Tag : int {
    kCompilerPluginLibraryDir = kLiteRtEnvOptionTagCompilerPluginLibraryDir,
    kDispatchLibraryDir = kLiteRtEnvOptionTagDispatchLibraryDir,
    kOpenClDeviceId = kLiteRtEnvOptionTagOpenClDeviceId,
    kOpenClPlatformId = kLiteRtEnvOptionTagOpenClPlatformId,
    kOpenClContext = kLiteRtEnvOptionTagOpenClContext,
    kOpenClCommandQueue = kLiteRtEnvOptionTagOpenClCommandQueue,
    kEglDisplay = kLiteRtEnvOptionTagEglDisplay,
    kEglContext = kLiteRtEnvOptionTagEglContext,
    kWebGpuDevice = kLiteRtEnvOptionTagWebGpuDevice,
    kWebGpuQueue = kLiteRtEnvOptionTagWebGpuQueue,
    kMetalDevice = kLiteRtEnvOptionTagMetalDevice,
    kMetalCommandQueue = kLiteRtEnvOptionTagMetalCommandQueue,
    // WARNING: Vulkan support is experimental.
    kVulkanEnvironment = kLiteRtEnvOptionTagVulkanEnvironment,
    kVulkanCommandPool = kLiteRtEnvOptionTagVulkanCommandPool,
    kCallbackOnGpuEnvDestroy = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
    kCallbackUserDataOnGpuEnvDestroy =
        kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
    kMagicNumberConfigs = kLiteRtEnvOptionTagMagicNumberConfigs,
    kMagicNumberVerifications = kLiteRtEnvOptionTagMagicNumberVerifications,
    kCompilerCacheDir = kLiteRtEnvOptionTagCompilerCacheDir,
    // Singleton ML Drift WebGPU/Dawn instance required for shared libraries not
    // to create their own instances.
    kWebGpuInstance = kLiteRtEnvOptionTagWebGpuInstance,
    // Dawn procedure table pointer for shared libraries to populate their
    // tables with the shared procedures instead of their own procedures.
    kWebGpuProcs = kLiteRtEnvOptionTagWebGpuProcs,
  };

  Expected<LiteRtVariant> GetOption(Tag tag) const {
    return GetOption(static_cast<LiteRtEnvOptionTag>(tag));
  }

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
