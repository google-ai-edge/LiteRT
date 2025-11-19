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

#include "litert/c/options/litert_darwinn_runtime_options.h"

#include <cstdint>
#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/litert_darwinn_options.h"

LiteRtStatus LiteRtCreateDarwinnRuntimeOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto options_data = std::make_unique<litert::LiteRtDarwinnRuntimeOptionsT>();

  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGetDarwinnRuntimeOptionsIdentifier(), options_data.get(),
      [](void* payload) {
        delete reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(payload);
      },
      options));

  options_data.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindDarwinnRuntimeOptions(
    LiteRtOpaqueOptions opaque_options,
    LiteRtDarwinnRuntimeOptions* runtime_options) {
  LITERT_RETURN_IF_ERROR(runtime_options,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "runtime_options is null.";

  void* options_data = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      opaque_options, LiteRtGetDarwinnRuntimeOptionsIdentifier(),
      &options_data));

  *runtime_options =
      reinterpret_cast<LiteRtDarwinnRuntimeOptions>(options_data);
  return kLiteRtStatusOk;
}

const char* LiteRtGetDarwinnRuntimeOptionsIdentifier() {
  return litert::LiteRtDarwinnRuntimeOptionsT::Identifier();
}

// Power management setters/getters
LiteRtStatus LiteRtSetDarwinnInferencePowerState(
    LiteRtDarwinnRuntimeOptions options, uint32_t power_state) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto* opts = reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  opts->inference_power_state = power_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnInferencePowerState(
    LiteRtDarwinnRuntimeOptionsConst options, uint32_t* power_state) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(power_state,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "power_state is null.";

  auto* opts =
      reinterpret_cast<const litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  *power_state = opts->inference_power_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnInferenceMemoryPowerState(
    LiteRtDarwinnRuntimeOptions options, uint32_t memory_power_state) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto* opts = reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  opts->inference_memory_power_state = memory_power_state;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnInferenceMemoryPowerState(
    LiteRtDarwinnRuntimeOptionsConst options, uint32_t* memory_power_state) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(memory_power_state,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "memory_power_state is null.";

  auto* opts =
      reinterpret_cast<const litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  *memory_power_state = opts->inference_memory_power_state;
  return kLiteRtStatusOk;
}

// Scheduling setters/getters
LiteRtStatus LiteRtSetDarwinnInferencePriority(
    LiteRtDarwinnRuntimeOptions options, int8_t priority) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto* opts = reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  opts->inference_priority = priority;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnInferencePriority(
    LiteRtDarwinnRuntimeOptionsConst options, int8_t* priority) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(priority,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "priority is null.";

  auto* opts =
      reinterpret_cast<const litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  *priority = opts->inference_priority;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnAtomicInference(
    LiteRtDarwinnRuntimeOptions options, bool atomic_inference) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto* opts = reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  opts->atomic_inference = atomic_inference;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnAtomicInference(
    LiteRtDarwinnRuntimeOptionsConst options, bool* atomic_inference) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(atomic_inference,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "atomic_inference is null.";

  auto* opts =
      reinterpret_cast<const litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  *atomic_inference = opts->atomic_inference;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnPreferCoherent(
    LiteRtDarwinnRuntimeOptions options, bool prefer_coherent) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";

  auto* opts = reinterpret_cast<litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  opts->prefer_coherent = prefer_coherent;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnPreferCoherent(
    LiteRtDarwinnRuntimeOptionsConst options, bool* prefer_coherent) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(prefer_coherent,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "prefer_coherent is null.";

  auto* opts =
      reinterpret_cast<const litert::LiteRtDarwinnRuntimeOptionsT*>(options);
  *prefer_coherent = opts->prefer_coherent;
  return kLiteRtStatusOk;
}
