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

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DARWINN_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DARWINN_OPTIONS_H_

#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/options/litert_darwinn_runtime_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

/// @brief A C++ wrapper for DarwiNN runtime options.
class DarwinnRuntimeOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;

  static const char* Discriminator() {
    return LiteRtGetDarwinnRuntimeOptionsIdentifier();
  }

  static absl::string_view Identifier();

  /// @brief Creates a new DarwiNN runtime options instance.
  static Expected<DarwinnRuntimeOptions> Create();

  /// @brief Finds DarwiNN runtime options in an existing opaque options list.
  static Expected<DarwinnRuntimeOptions> Create(OpaqueOptions& options);

  /// @brief Sets/gets the power management settings.
  Expected<void> SetInferencePowerState(uint32_t power_state);
  Expected<uint32_t> GetInferencePowerState() const;

  Expected<void> SetInferenceMemoryPowerState(uint32_t memory_power_state);
  Expected<uint32_t> GetInferenceMemoryPowerState() const;

  /// @brief Sets/gets the scheduling priority.
  Expected<void> SetInferencePriority(int8_t priority);
  Expected<int8_t> GetInferencePriority() const;

  /// @brief Sets/gets atomic inference settings to disable TPU firmware
  /// concurrency.
  Expected<void> SetAtomicInference(bool atomic_inference);
  Expected<bool> GetAtomicInference() const;

  /// @brief Sets/gets the memory coherency preference.
  Expected<void> SetPreferCoherent(bool prefer_coherent);
  Expected<bool> GetPreferCoherent() const;

  /// @brief Sets/gets the runtime internal options.
  Expected<void> SetInternalOptions(absl::string_view internal_options);
  Expected<absl::string_view> GetInternalOptions() const;
};

/// @note `FindOpaqueOptions` template specializations are not needed.
/// The generic template in `litert_opaque_options.h` will use `Discriminator()`
/// to find the options in the linked list and then call
/// `Create(found_options)`.

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_LITERT_DARWINN_OPTIONS_H_
