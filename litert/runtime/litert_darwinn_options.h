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

#ifndef ODML_LITERT_LITERT_RUNTIME_LITERT_DARWINN_OPTIONS_H_
#define ODML_LITERT_LITERT_RUNTIME_LITERT_DARWINN_OPTIONS_H_

#include <cstdint>

namespace litert {

constexpr uint32_t kDefaultInferencePowerState = 0;
constexpr uint32_t kDefaultInferenceMemoryPowerState = 0;

// Runtime/per-inference options for DarwiNN delegate
struct LiteRtDarwinnRuntimeOptionsT {
  // Power management - frequently changed
  uint32_t inference_power_state = kDefaultInferencePowerState;
  uint32_t inference_memory_power_state = kDefaultInferenceMemoryPowerState;

  // Scheduling - may change per workload
  int8_t inference_priority = -1;  // -1 means default

  // Whether to run inference atomically
  bool atomic_inference = false;

  // Memory coherency preference
  bool prefer_coherent = false;  // Whether to prefer coherent memory allocation

  static const char* Identifier() { return "darwinn_runtime_options"; }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_RUNTIME_LITERT_DARWINN_OPTIONS_H_
