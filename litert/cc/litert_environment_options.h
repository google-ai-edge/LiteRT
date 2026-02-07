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

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert {

/// Manages configuration options for a LiteRT Environment.
///
/// This class provides methods to access various options related to the LiteRT
/// environment, such as compiler settings, dispatch libraries, and hardware
/// acceleration configurations (e.g., OpenCL, EGL, WebGPU, Metal, Vulkan).
class EnvironmentOptions {
 public:
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

  struct Option {
    Tag tag;
    LiteRtVariant value;
    // To keep the ownership of the string value.
    std::string str_value;

    Option(Tag param_tag, LiteRtVariant param_value) : tag(param_tag) {
      if (std::holds_alternative<absl::string_view>(param_value)) {
        str_value = std::get<absl::string_view>(param_value);
        value = str_value.c_str();
      } else if (std::holds_alternative<const char*>(param_value)) {
        str_value = std::get<const char*>(param_value);
        value = str_value.c_str();
      } else {
        value = param_value;
      }
    }

    Option(const Option& other) : tag(other.tag), str_value(other.str_value) {
      if (std::holds_alternative<const char*>(other.value)) {
        value = str_value.c_str();
      } else {
        value = other.value;
      }
    }

    Option& operator=(const Option& other) {
      if (this != &other) {
        tag = other.tag;
        str_value = other.str_value;
        if (std::holds_alternative<const char*>(other.value)) {
          value = str_value.c_str();
        } else {
          value = other.value;
        }
      }
      return *this;
    }

    Option(Option&& other) noexcept
        : tag(other.tag), str_value(std::move(other.str_value)) {
      if (std::holds_alternative<const char*>(other.value)) {
        value = str_value.c_str();
      } else {
        value = std::move(other.value);
      }
    }

    Option& operator=(Option&& other) noexcept {
      if (this != &other) {
        tag = other.tag;
        str_value = std::move(other.str_value);
        if (std::holds_alternative<const char*>(other.value)) {
          value = str_value.c_str();
        } else {
          value = std::move(other.value);
        }
      }
      return *this;
    }
  };

  /// Constructs an `EnvironmentOptions` object from a span of options.
  /// @param options A span of `Option` objects to initialize the environment
  /// options with.
  explicit EnvironmentOptions(absl::Span<const Option> options)
      : options_(options.begin(), options.end()) {}

  /// Retrieves all options.
  /// @return A span of all options.
  absl::Span<const Option> GetOptions() const {
    return absl::MakeConstSpan(options_);
  }

  /// Retrieves the value of an option specified by a tag.
  /// @param tag The tag of the option to retrieve.
  /// @return An `Expected` object containing the option value if successful,
  /// or an error if the option is not found or the handle is null.
  Expected<const LiteRtVariant> GetOption(Tag tag) const {
    for (const auto& option : options_) {
      if (option.tag == tag) {
        return option.value;
      }
    }
    return Error(Status::kErrorNotFound,
                 "Option was not set for this environment.");
  }

  /// @deprecated Use `GetOption(Tag)` instead.
  [[deprecated("Use GetOption(Tag) instead.")]]
  Expected<const LiteRtVariant> GetOption(LiteRtEnvOptionTag tag) const {
    return GetOption(static_cast<Tag>(tag));
  }

 private:
  std::vector<Option> options_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_LITERT_ENVIRONMENT_OPTIONS_H_
