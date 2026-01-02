// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_H_
#define ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_H_

#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

/// @brief The environment works like a context that holds the runtime states.
///
/// To create a `CompiledModel` or `TensorBuffer`, an Environment is required.
/// In a case of having multiple CompiledModels, it is recommended to share
/// the same Environment.
class Environment
    : public internal::Handle<LiteRtEnvironment, LiteRtDestroyEnvironment> {
 public:
  enum class OptionTag {
    CompilerPluginLibraryDir = kLiteRtEnvOptionTagCompilerPluginLibraryDir,
    DispatchLibraryDir = kLiteRtEnvOptionTagDispatchLibraryDir,
    ClDeviceId = kLiteRtEnvOptionTagOpenClDeviceId,
    ClPlatformId = kLiteRtEnvOptionTagOpenClPlatformId,
    ClContext = kLiteRtEnvOptionTagOpenClContext,
    ClCommandQueue = kLiteRtEnvOptionTagOpenClCommandQueue,
    EglContext = kLiteRtEnvOptionTagEglContext,
    EglDisplay = kLiteRtEnvOptionTagEglDisplay,
    WebGpuDevice = kLiteRtEnvOptionTagWebGpuDevice,
    WebGpuQueue = kLiteRtEnvOptionTagWebGpuQueue,
    MetalDevice = kLiteRtEnvOptionTagMetalDevice,
    MetalCommandQueue = kLiteRtEnvOptionTagMetalCommandQueue,
    /// @warning Vulkan support is experimental.
    VulkanEnvironment = kLiteRtEnvOptionTagVulkanEnvironment,
    CallbackOnGpuEnvDestroy = kLiteRtEnvOptionTagCallbackOnGpuEnvDestroy,
    CallbackUserDataOnGpuEnvDestry =
        kLiteRtEnvOptionTagCallbackUserDataOnGpuEnvDestroy,
    MagicNumberConfigs = kLiteRtEnvOptionTagMagicNumberConfigs,
    MagicNumberVerifications = kLiteRtEnvOptionTagMagicNumberVerifications,
    CompilerCacheDir = kLiteRtEnvOptionTagCompilerCacheDir,
    WebGpuInstance = kLiteRtEnvOptionTagWebGpuInstance,
    WebGpuProcs = kLiteRtEnvOptionTagWebGpuProcs,
  };

  struct Option {
    OptionTag tag;
    LiteRtVariant value;
  };

  Expected<EnvironmentOptions> GetOptions() const {
    LiteRtEnvironmentOptions options;
    LITERT_RETURN_IF_ERROR(LiteRtGetEnvironmentOptions(Get(), &options));
    return EnvironmentOptions(options);
  }

  static Expected<Environment> Create(absl::Span<const Option> options) {
    auto c_options = ConvertOptions(options);
    if (!c_options) {
      return c_options.Error();
    }
    LiteRtEnvironment env;
    if (auto status =
            LiteRtCreateEnvironment(c_options->size(), c_options->data(), &env);
        status != kLiteRtStatusOk) {
      return Error(status);
    } else {
      return Environment(env);
    }
  }

  /// @brief Returns whether the environment supports CL/GL interop.
  bool SupportsClGlInterop() const {
    bool is_supported = false;
    if (auto status =
            LiteRtEnvironmentSupportsClGlInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/CL interop.
  bool SupportsAhwbClInterop() const {
    bool is_supported = false;
    if (auto status =
            LiteRtEnvironmentSupportsAhwbClInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/GL interop.
  bool SupportsAhwbGlInterop() const {
    bool is_supported = false;
    if (auto status =
            LiteRtEnvironmentSupportsAhwbGlInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @internal
  /// @brief Wraps a `LiteRtEnvironment` C object in an `Environment` C++
  /// object.
  /// @warning This is for internal use only.
  static Environment WrapCObject(LiteRtEnvironment env, OwnHandle owned) {
    return Environment(env, owned);
  }

 private:
  explicit Environment(LiteRtEnvironment env, OwnHandle owned = OwnHandle::kYes)
      : Handle(env, owned) {}

  static Expected<std::vector<LiteRtEnvOption>> ConvertOptions(
      absl::Span<const Option> options) {
    std::vector<LiteRtEnvOption> c_options;
    c_options.reserve(options.size());

    for (auto& option : options) {
      auto litert_any = ToLiteRtAny(option.value);
      if (!litert_any) {
        return litert_any.Error();
      }

      LiteRtEnvOption c_option = {
          /*.tag=*/static_cast<LiteRtEnvOptionTag>(option.tag),
          /*.value=*/*litert_any,
      };
      c_options.push_back(c_option);
    }

    return c_options;
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_H_
