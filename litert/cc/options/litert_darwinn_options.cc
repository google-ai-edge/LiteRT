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

#include "litert/cc/options/litert_darwinn_options.h"

#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_darwinn_runtime_options.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert {

// DarwinnRuntimeOptions implementation

absl::string_view DarwinnRuntimeOptions::Identifier() {
  return LiteRtGetDarwinnRuntimeOptionsIdentifier();
}

Expected<DarwinnRuntimeOptions> DarwinnRuntimeOptions::Create() {
  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtCreateDarwinnRuntimeOptions(&opaque_options));

  DarwinnRuntimeOptions options(opaque_options, OwnHandle::kYes);
  return options;
}

Expected<DarwinnRuntimeOptions> DarwinnRuntimeOptions::Create(
    OpaqueOptions& options) {
  LITERT_ASSIGN_OR_RETURN(absl::string_view original_identifier,
                          options.GetIdentifier());
  LITERT_RETURN_IF_ERROR(original_identifier == Identifier(),
                         ErrorStatusBuilder::InvalidArgument())
      << "Cannot create DarwiNN runtime options from an opaque options object "
         "that doesn't "
         "already hold DarwiNN runtime options.";
  LiteRtOpaqueOptions opaque_options = options.Get();
  return DarwinnRuntimeOptions(opaque_options, OwnHandle::kNo);
}

// Power management setters/getters
Expected<void> DarwinnRuntimeOptions::SetInferencePowerState(
    uint32_t power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInferencePowerState(darwinn_options, power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInferencePowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t power_state = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInferencePowerState(darwinn_options, &power_state));
  return power_state;
}

Expected<void> DarwinnRuntimeOptions::SetInferenceMemoryPowerState(
    uint32_t memory_power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(LiteRtSetDarwinnInferenceMemoryPowerState(
      darwinn_options, memory_power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInferenceMemoryPowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t memory_power_state = 0;
  LITERT_RETURN_IF_ERROR(LiteRtGetDarwinnInferenceMemoryPowerState(
      darwinn_options, &memory_power_state));
  return memory_power_state;
}

// Scheduling setters/getters
Expected<void> DarwinnRuntimeOptions::SetInferencePriority(int8_t priority) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInferencePriority(darwinn_options, priority));
  return {};
}

Expected<int8_t> DarwinnRuntimeOptions::GetInferencePriority() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  int8_t priority = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInferencePriority(darwinn_options, &priority));
  return priority;
}

Expected<void> DarwinnRuntimeOptions::SetAtomicInference(
    bool atomic_inference) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnAtomicInference(darwinn_options, atomic_inference));
  return {};
}

Expected<bool> DarwinnRuntimeOptions::GetAtomicInference() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  bool atomic_inference = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnAtomicInference(darwinn_options, &atomic_inference));
  return atomic_inference;
}

Expected<void> DarwinnRuntimeOptions::SetPreferCoherent(bool prefer_coherent) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnPreferCoherent(darwinn_options, prefer_coherent));
  return {};
}

Expected<bool> DarwinnRuntimeOptions::GetPreferCoherent() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  bool prefer_coherent = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnPreferCoherent(darwinn_options, &prefer_coherent));
  return prefer_coherent;
}
}  // namespace litert
