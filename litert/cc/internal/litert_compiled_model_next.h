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
#include "litert/c/litert_compiled_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"

namespace litert {

//  Advanced CompiledModel with new / experimental features.
class CompiledModelNext : public CompiledModel {
 public:
  // Hardware specific metrics collected by the CompiledModel.
  struct Metrics {
    struct Metric {
      std::string name;
      LiteRtAny value;
    };
    std::vector<Metric> metrics;
  };

  // Creates a CompiledModelNext with the given model and non-const compilation
  // options. The passed options will be built during the creation.
  static Expected<CompiledModelNext> Create(litert::Environment& env,
                                            const litert::Model& model,
                                            Options& compilation_options);

  // Similar to above, but takes const compilation options that are already
  // built.
  static Expected<CompiledModelNext> Create(litert::Environment& env,
                                            const litert::Model& model,
                                            const Options& compilation_options);

  // Simple version that only takes hardware accelerators.
  static Expected<CompiledModelNext> Create(
      litert::Environment& env, const litert::Model& model,
      litert::HwAccelerators hardware_accelerators);

  // Starts collection of HW-specific metrics at a specific level of detail.
  Expected<void> StartMetricsCollection(int detail_level);

  // Stops collection of HW-specific metrics and report the collected metrics.
  Expected<Metrics> StopMetricsCollection();

  // Sets a dispatch annotation on the compiled model. These annotations will be
  // propagated to dispatch graphs when they are created during model execution.
  // The annotations provide runtime hints and metadata that can be used by
  // hardware accelerators for optimization.
  //
  // Parameters:
  // - signature_index: the index of the signature (zero-based).
  // - key: the annotation key.
  // - value: the annotation value.
  //
  // Example annotations:
  // - "priority": "high|medium|low" - execution priority hints
  // - "memory_type": "shared|dedicated" - memory allocation preferences
  // - "accelerator": "npu|gpu|dsp" - preferred hardware accelerator
  // - "precision": "fp32|fp16|int8" - computation precision requirements
  Expected<void> SetDispatchAnnotation(size_t signature_index,
                                       absl::string_view key,
                                       absl::string_view value) {
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelSetDispatchAnnotation(
        Get(), signature_index, key.data(), value.data()));
    return {};
  }

  // Gets a dispatch annotation from the compiled model.
  //
  // Parameters:
  // - signature_index: the index of the signature (zero-based).
  // - key: the annotation key to look up.
  //
  // Returns:
  // - The annotation value if found, or nullopt if the key doesn't exist.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      size_t signature_index, absl::string_view key) {
    const char* value = nullptr;
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelGetDispatchAnnotation(
        Get(), signature_index, key.data(), &value));
    if (value == nullptr) {
      return Expected<std::optional<std::string>>(std::nullopt);
    }
    return Expected<std::optional<std::string>>(std::string(value));
  }

  // Removes a dispatch annotation from the compiled model.
  //
  // Parameters:
  // - signature_index: the index of the signature (zero-based).
  // - key: the annotation key to remove.
  //
  // Note: This function succeeds even if the key doesn't exist.
  Expected<void> RemoveDispatchAnnotation(size_t signature_index,
                                          absl::string_view key) {
    LITERT_RETURN_IF_ERROR(LiteRtCompiledModelRemoveDispatchAnnotation(
        Get(), signature_index, key.data()));
    return {};
  }

  // Overloaded version for the default signature (index 0).
  Expected<void> SetDispatchAnnotation(absl::string_view key,
                                       absl::string_view value) {
    return SetDispatchAnnotation(0, key, value);
  }

  // Overloaded version for the default signature (index 0).
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view key) {
    return GetDispatchAnnotation(0, key);
  }

  // Overloaded version for the default signature (index 0).
  Expected<void> RemoveDispatchAnnotation(absl::string_view key) {
    return RemoveDispatchAnnotation(0, key);
  }

  // Overloaded version that takes a signature name instead of index.
  Expected<void> SetDispatchAnnotation(absl::string_view signature_name,
                                       absl::string_view key,
                                       absl::string_view value) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return SetDispatchAnnotation(signature_index, key, value);
  }

  // Overloaded version that takes a signature name instead of index.
  Expected<std::optional<std::string>> GetDispatchAnnotation(
      absl::string_view signature_name, absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return GetDispatchAnnotation(signature_index, key);
  }

  // Overloaded version that takes a signature name instead of index.
  Expected<void> RemoveDispatchAnnotation(absl::string_view signature_name,
                                          absl::string_view key) {
    LITERT_ASSIGN_OR_RETURN(size_t signature_index,
                            model_.GetSignatureIndex(signature_name));
    return RemoveDispatchAnnotation(signature_index, key);
  }

 private:
  explicit CompiledModelNext(LiteRtModel litert_model,
                             LiteRtCompiledModel compiled_model,
                             OwnHandle owned)
      : CompiledModel(litert_model, /*model_owned=*/OwnHandle::kNo,
                      compiled_model, owned) {}
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CC_INTERNAL_LITERT_COMPILED_MODEL_NEXT_H_
