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

#ifndef ODML_LITERT_LITERT_CORE_ENVIRONMENT_H_
#define ODML_LITERT_LITERT_CORE_ENVIRONMENT_H_

#include <memory>
#include <optional>
#include <utility>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/environment_options.h"
#include "litert/runtime/accelerator_registry.h"
#include "litert/runtime/gpu_environment.h"
#include "litert/runtime/tensor_buffer_registry.h"

// A singleton class that contains global LiteRT environment options.
class LiteRtEnvironmentT {
 public:
  using Ptr = std::unique_ptr<LiteRtEnvironmentT>;

  LiteRtEnvironmentT() = default;
  // Create an environment instance with options.
  static litert::Expected<Ptr> CreateWithOptions(
      absl::Span<const LiteRtEnvOption> options);

  ~LiteRtEnvironmentT() = default;

  std::optional<LiteRtAny> GetOption(LiteRtEnvOptionTag tag) const {
    auto opt = options_.GetOption(tag);
    return opt.HasValue() ? std::optional<LiteRtAny>(opt.Value())
                          : std::nullopt;
  }

  LiteRtEnvironmentOptionsT& GetOptions() { return options_; }
  const LiteRtEnvironmentOptionsT& GetOptions() const { return options_; }

  // Adds options to the existing environment.
  litert::Expected<void> AddOptions(absl::Span<const LiteRtEnvOption> options,
                                    bool overwrite = false);

  // Returns the accelerator registry.
  litert::internal::AcceleratorRegistry& GetAcceleratorRegistry() {
    return accelerators_;
  }

  // Returns the tensor buffer registry.
  litert::internal::TensorBufferRegistry& GetTensorBufferRegistry() {
    return tensor_buffer_registry_;
  }

  // Sets the GPU environment. The owner of the GPU environment is transferred
  // to the environment.
  litert::Expected<void> SetGpuEnvironment(
      std::unique_ptr<litert::internal::GpuEnvironment> gpu_env) {
    if (gpu_env_) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "GPU environment is already set.");
    }
    gpu_env_ = std::move(gpu_env);
    return {};
  }

  // Returns the GPU environment object.
  litert::Expected<litert::internal::GpuEnvironment*> GetGpuEnvironment() {
    if (!HasGpuEnvironment()) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "GPU environment is not set.");
    }
    return gpu_env_.get();
  }

  // Returns true if the GPU environment is set.
  bool HasGpuEnvironment() { return gpu_env_ != nullptr; }

  bool SupportsClGlInterop() {
    return gpu_env_ != nullptr && gpu_env_->SupportsClGlInterop();
  }

  bool SupportsAhwbClInterop() {
    return gpu_env_ != nullptr && gpu_env_->SupportsAhwbClInterop();
  }

  bool SupportsAhwbGlInterop() {
    return gpu_env_ != nullptr && gpu_env_->SupportsAhwbGlInterop();
  }

 private:
  litert::internal::AcceleratorRegistry accelerators_;
  litert::internal::TensorBufferRegistry tensor_buffer_registry_;
  LiteRtEnvironmentOptionsT options_;
  std::unique_ptr<litert::internal::GpuEnvironment> gpu_env_;
};

#endif  // ODML_LITERT_LITERT_CORE_ENVIRONMENT_H_
