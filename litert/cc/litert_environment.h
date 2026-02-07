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
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/internal/litert_runtime_builtin.h"
#include "litert/cc/internal/litert_runtime_proxy.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {
using internal::RuntimeProxy;

namespace internal {
/// @internal
/// @brief A holder for an environment and its associated runtime.
/// TODO(b/478306820): Mark fields as const after the bug is fixed.
struct EnvironmentHolder {
  /// @brief The runtime that the environment is associated with.
  RuntimeProxy* runtime;
  /// @brief The environment handle.
  LiteRtEnvironment handle;

  bool operator==(const EnvironmentHolder& other) const noexcept {
    return runtime == other.runtime && handle == other.handle;
  }

  bool operator!=(const EnvironmentHolder& other) const noexcept {
    return !(*this == other);
  }
};
}  // namespace internal

/// @brief The environment works like a context that holds the runtime states.
///
/// To create a `CompiledModel` or `TensorBuffer`, an Environment is required.
/// In a case of having multiple CompiledModels, it is recommended to share
/// the same Environment.
class Environment {
 public:
  enum class [[deprecated("Use EnvironmentOptions::Tag instead.")]] OptionTag {
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

  struct [[deprecated("Use EnvironmentOptions::Option instead.")]] Option {
    OptionTag tag;
    LiteRtVariant value;
  };

  Expected<EnvironmentOptions> GetOptions() const {
    LiteRtEnvironmentOptions options;
    LITERT_RETURN_IF_ERROR(
        runtime_->GetEnvironmentOptions(handle_.get(), &options));
    return FromCOptions(options);
  }

  static Expected<Environment> Create(absl::Span<const Option> options) {
    std::vector<EnvironmentOptions::Option> env_options;
    env_options.reserve(options.size());
    for (const auto& option : options) {
      env_options.push_back(
          {static_cast<EnvironmentOptions::Tag>(option.tag), option.value});
    }
    auto env_options_obj = EnvironmentOptions(env_options);
    return Create(env_options_obj);
  }

  static Expected<Environment> Create(const EnvironmentOptions& options) {
    auto c_options = ToCOptions(options.GetOptions());
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
    if (auto status = runtime_->EnvironmentSupportsClGlInterop(handle_.get(),
                                                               &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/CL interop.
  bool SupportsAhwbClInterop() const {
    bool is_supported = false;
    if (auto status = runtime_->EnvironmentSupportsAhwbClInterop(handle_.get(),
                                                                 &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @brief Returns whether the environment supports AHWB/GL interop.
  bool SupportsAhwbGlInterop() const {
    bool is_supported = false;
    if (auto status = runtime_->EnvironmentSupportsAhwbGlInterop(handle_.get(),
                                                                 &is_supported);
        status != kLiteRtStatusOk) {
      return false;
    }
    return is_supported;
  }

  /// @internal
  /// @brief Returns the underlying environment handle.
  [[deprecated("Use GetHolder() instead.")]]
  LiteRtEnvironment Get() const noexcept {
    return handle_.get();
  }

  /// @internal
  /// @brief Returns the underlying environment handle and runtime.
  internal::EnvironmentHolder GetHolder() const noexcept {
    return {runtime_.get(), handle_.get()};
  }

  /// @internal
  /// @brief Releases ownership of the environment handle.
  ///
  /// After this call, `GetHolder()` returns a null handle.
  internal::EnvironmentHolder Release() noexcept {
    return {runtime_.release(), handle_.release()};
  }

  /// @brief Returns `true` if the underlying LiteRT handle is valid.
  explicit operator bool() const noexcept { return static_cast<bool>(handle_); }

  bool operator==(const Environment& other) const noexcept {
    return GetHolder() == other.GetHolder();
  }

  bool operator!=(const Environment& other) const noexcept {
    return GetHolder() != other.GetHolder();
  }

  /// @internal
  /// @brief Wraps a `LiteRtEnvironment` C object in an `Environment` C++
  /// object, with the builtin runtime.
  /// @warning This is for internal use only.
  static Environment WrapCObject(LiteRtEnvironment env, OwnHandle owned) {
    auto runtime = GetBuiltinRuntime();
    return Environment(env, std::move(runtime), owned);
  }

  /// @internal
  /// @brief Wraps a `EnvironmentHolder` in an `Environment` C++ object.
  /// @warning This is for internal use only.
  static Environment WrapCObject(const internal::EnvironmentHolder& env,
                                 OwnHandle owned) {
    auto runtime =
        std::unique_ptr<RuntimeProxy, std::function<void(RuntimeProxy*)>>(
            env.runtime, [](RuntimeProxy*) {});
    return Environment(env.handle, std::move(runtime), owned);
  }

 private:
  explicit Environment(
      LiteRtEnvironment env,
      std::unique_ptr<RuntimeProxy, std::function<void(RuntimeProxy*)>> runtime,
      OwnHandle owned = OwnHandle::kYes)
      : runtime_(std::move(runtime)) {
    std::function<void(LiteRtEnvironment)> handle_deleter;
    auto runtime_ptr = runtime_.get();
    if (owned == OwnHandle::kYes && runtime_ptr != nullptr) {
      handle_deleter = [runtime_ptr](LiteRtEnvironment env_) {
        runtime_ptr->DestroyEnvironment(env_);
      };
    } else {
      handle_deleter = [](LiteRtEnvironment) {};
    }
    handle_ = std::unique_ptr<std::remove_pointer_t<LiteRtEnvironment>,
                              std::function<void(LiteRtEnvironment)>>(
        env, handle_deleter);
  }

  std::unique_ptr<RuntimeProxy, std::function<void(RuntimeProxy*)>> runtime_;
  // handle_ needs to be declared after runtime_ to ensure that they will be
  // destroyed in the reverse order when the Environment is destroyed. This is
  // because the deleter function may access the member runtime_.
  std::unique_ptr<std::remove_pointer_t<LiteRtEnvironment>,
                  std::function<void(LiteRtEnvironment)>>
      handle_;

  static Expected<std::vector<LiteRtEnvOption>> ToCOptions(
      absl::Span<const EnvironmentOptions::Option> options) {
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

  Expected<EnvironmentOptions> FromCOptions(
      LiteRtEnvironmentOptions options) const {
    std::vector<EnvironmentOptions::Option> env_options;
    for (int i = 0; i <= kLiteRtEnvOptionTagRuntimeLibraryDir; ++i) {
      LiteRtAny value;
      if (runtime_->GetEnvironmentOptionsValue(
              options, static_cast<LiteRtEnvOptionTag>(i), &value) ==
          kLiteRtStatusOk) {
        EnvironmentOptions::Option option(
            static_cast<EnvironmentOptions::Tag>(i), ToStdAny(value));
        env_options.push_back(std::move(option));
      }
    }
    return EnvironmentOptions(env_options);
  }

  static std::unique_ptr<RuntimeProxy> GetBuiltinRuntime() {
    return std::make_unique<RuntimeProxy>(&kLiteRtRuntimeBuiltin);
  }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_H_
