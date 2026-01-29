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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_COMPILED_MODEL_NEXT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_COMPILED_MODEL_NEXT_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"

/// @file
/// @brief Defines an advanced `CompiledModel` with new and experimental
/// features.

namespace litert {

/// @brief An advanced `CompiledModel` with new and experimental features.
class CompiledModelNext : public CompiledModel {
 public:
  /// @brief Hardware-specific metrics collected by the `CompiledModel`.
  struct Metrics {
    struct Metric {
      std::string name;
      LiteRtAny value;
    };
    std::vector<Metric> metrics;
  };

  /// @brief Creates a `CompiledModelNext` with the given model and non-const
  /// compilation options. The passed options will be built during creation.
  static Expected<CompiledModelNext> Create(litert::Environment& env,
                                            const litert::Model& model,
                                            Options& compilation_options);

  /// @brief Creates a `CompiledModelNext` with const compilation options that
  /// are already built.
  static Expected<CompiledModelNext> Create(litert::Environment& env,
                                            const litert::Model& model,
                                            const Options& compilation_options);

  /// @brief A simplified version that only takes hardware accelerators.
  static Expected<CompiledModelNext> Create(
      litert::Environment& env, const litert::Model& model,
      litert::HwAccelerators hardware_accelerators);

  /// @brief Starts the collection of hardware-specific metrics at a given
  /// level of detail.
  Expected<void> StartMetricsCollection(int detail_level);

  /// @brief Stops the collection of hardware-specific metrics and reports the
  /// collected data.
  Expected<Metrics> StopMetricsCollection();

  /// @brief Sets a dispatch annotation on the compiled model.
  ///
  /// These annotations are propagated to dispatch graphs during model
  /// execution and provide runtime hints and metadata for hardware accelerator
  /// optimization.
  ///
  /// @param signature_index The zero-based index of the signature.
  /// @param key The annotation key.
  /// @param value The annotation value.
  ///
  /// @par Example Annotations:
  /// - `"priority"`: `"high|medium|low"` - Execution priority hints.
  /// - `"memory_type"`: `"shared|dedicated"` - Memory allocation preferences.
  /// - `"accelerator"`: `"npu|gpu|dsp"` - Preferred hardware accelerator.
  /// - `"precision"`: `"fp32|fp16|int8"` - Computation precision requirements.
  Expected<void> SetDispatchAnnotation(size_t signature_index,
                                       absl::string_view key,
                                       absl::string_view value) {
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelSetDispatchAnnotation(
        Get(), signature_index, key.data(), value.data()));
    return {};
  }

  /// @brief Gets a dispatch annotation from the compiled model.
  ///
  /// @param signature_index The zero-based index of the signature.
  /// @param key The annotation key to look up.
  /// @return The annotation value if found, or `std::nullopt` if the key does
  /// not exist.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      size_t signature_index, absl::string_view key) {
    const char* value = nullptr;
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelGetDispatchAnnotation(
        Get(), signature_index, key.data(), &value));
    if (value == nullptr) {
      return Expected<std::optional<std::string>>(std::nullopt);
    }
    return Expected<std::optional<std::string>>(std::string(value));
  }

  /// @brief Removes a dispatch annotation from the compiled model.
  ///
  /// @param signature_index The zero-based index of the signature.
  /// @param key The annotation key to remove.
  /// @note This function succeeds even if the key does not exist.
  Expected<void> RemoveDispatchAnnotation(size_t signature_index,
                                          absl::string_view key) {
    LITERT_RETURN_IF_ERROR(env_.runtime->CompiledModelRemoveDispatchAnnotation(
        Get(), signature_index, key.data()));
    return {};
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<void> SetDispatchAnnotation(absl::string_view key,
                                       absl::string_view value) {
    return SetDispatchAnnotation(0, key, value);
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view key) {
    return GetDispatchAnnotation(0, key);
  }

  /// @brief Overloaded version for the default signature (index 0).
  Expected<void> RemoveDispatchAnnotation(absl::string_view key) {
    return RemoveDispatchAnnotation(0, key);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<void> SetDispatchAnnotation(absl::string_view signature_name,
                                       absl::string_view key,
                                       absl::string_view value) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return SetDispatchAnnotation(signature_index, key, value);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view signature_name, absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return GetDispatchAnnotation(signature_index, key);
  }

  /// @brief Overloaded version that takes a signature name instead of an
  /// index.
  Expected<void> RemoveDispatchAnnotation(absl::string_view signature_name,
                                          absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return RemoveDispatchAnnotation(signature_index, key);
  }

 private:
  explicit CompiledModelNext(internal::EnvironmentHolder& env,
                             LiteRtModel litert_model,
                             LiteRtCompiledModel compiled_model,
                             OwnHandle owned)
      : CompiledModel(env, litert_model,
                      /*model_owned=*/OwnHandle::kNo, compiled_model, owned) {}
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_COMPILED_MODEL_NEXT_H_
