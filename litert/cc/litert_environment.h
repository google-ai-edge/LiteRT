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

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
using internal::RuntimeProxy;

/// @brief The environment works like a context that holds the runtime states.
///
/// To create a `CompiledModel` or `TensorBuffer`, an Environment is required.
/// In a case of having multiple CompiledModels, it is recommended to share
/// the same Environment.
class Environment {
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
    RuntimeLibraryDir = kLiteRtEnvOptionTagRuntimeLibraryDir,
  };

  struct Option {
    OptionTag tag;
    LiteRtVariant value;
  };

  Expected<EnvironmentOptions> GetOptions() const {
    LiteRtEnvironmentOptions options;
    LITERT_RETURN_IF_ERROR(runtime_->GetEnvironmentOptions(Get(), &options));
    return EnvironmentOptions(options);
  }

  static Expected<Environment> Create(absl::Span<const Option> options) {
    auto c_options = ConvertOptions(options);
    if (!c_options) {
      return c_options.Error();
    }
    auto runtime = GetBuiltinRuntime();
    LiteRtEnvironment env;
    if (auto status = runtime->CreateEnvironment(c_options->size(),
                                                 c_options->data(), &env);
        status != kLiteRtStatusOk) {
      return Error(status);
    } else {
      return Environment(env, std::move(runtime));
    }
  }

  /// @brief Returns whether the environment supports CL/GL interop.
  bool SupportsClGlInterop() const {
    bool is_supported = false;
    if (auto status =
            runtime_->EnvironmentSupportsClGlInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/CL interop.
  bool SupportsAhwbClInterop() const {
    bool is_supported = false;
    if (auto status =
            runtime_->EnvironmentSupportsAhwbClInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/GL interop.
  bool SupportsAhwbGlInterop() const {
    bool is_supported = false;
    if (auto status =
            runtime_->EnvironmentSupportsAhwbGlInterop(Get(), &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns the underlying environment handle.
  LiteRtEnvironment Get() const noexcept { return handle_.get(); }

  /// @brief Releases ownership of the environment handle.
  ///
  /// After this call, `Get()` returns a null handle.
  LiteRtEnvironment Release() noexcept { return handle_.release(); }

  /// @brief Returns `true` if the underlying LiteRT handle is valid.
  explicit operator bool() const noexcept { return static_cast<bool>(handle_); }

  bool operator==(const Environment& other) const noexcept {
    return Get() == other.Get();
  }

  bool operator!=(const Environment& other) const noexcept {
    return Get() != other.Get();
  }

  /// @internal
  /// @brief Wraps a `LiteRtEnvironment` C object in an `Environment` C++
  /// object.
  /// @warning This is for internal use only.
  static Environment WrapCObject(LiteRtEnvironment env, OwnHandle owned) {
    auto runtime = GetBuiltinRuntime();
    return Environment(env, std::move(runtime), owned);
  }

 private:
  explicit Environment(LiteRtEnvironment env,
                       std::unique_ptr<RuntimeProxy> runtime,
                       OwnHandle owned = OwnHandle::kYes)
      : runtime_(std::move(runtime)) {
    std::function<void(LiteRtEnvironment)> deleter;
    if (owned == OwnHandle::kYes) {
      deleter = [runtime_ptr = runtime_.get()](LiteRtEnvironment env_) {
        runtime_ptr->DestroyEnvironment(env_);
      };
    } else {
      deleter = [](LiteRtEnvironment) {};
    }
    handle_ =
        std::unique_ptr<std::remove_pointer_t<LiteRtEnvironment>,
                        std::function<void(LiteRtEnvironment)>>(env, deleter);
  }

  std::unique_ptr<RuntimeProxy> runtime_;
  // handle_ needs to be declared after runtime_ to ensure that they will be
  // destroyed in the reverse order when the Environment is destroyed. This is
  // because the deleter function may access the member runtime_.
  std::unique_ptr<std::remove_pointer_t<LiteRtEnvironment>,
                  std::function<void(LiteRtEnvironment)>>
      handle_;

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

  static std::unique_ptr<RuntimeProxy> GetBuiltinRuntime() {
    return std::make_unique<RuntimeProxy>(&kLiteRtRuntimeBuiltin);
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_H_
