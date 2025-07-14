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
#include <string>

namespace litert {

// Default power states - these would normally come from DarwiNN headers
// For now, using placeholder values
constexpr uint32_t kDefaultInferencePowerState = 0;
constexpr uint32_t kDefaultInferenceMemoryPowerState = 0;
constexpr uint32_t kDefaultInactivePowerState = 0;
constexpr uint32_t kDefaultInactiveMemoryPowerState = 0;
constexpr int64_t kDefaultInactiveTimeoutUs = 1000000;  // 1 second

// Device/compilation-time options for DarwiNN delegate
struct LiteRtDarwinnDeviceOptionsT {
  // Device selection
  std::string device_type;  // "usb", "pci", etc.
  std::string device_path;  // Optional specific device path
  
  // Compilation options  
  bool enable_multiple_subgraphs = false;
  bool compile_if_resize = false;
  bool allow_cpu_fallback = false;
  bool skip_op_filter = false;
  int num_interpreters = 1;
  
  // Memory configuration
  bool avoid_bounce_buffer = false;
  bool register_graph_during_modify_graph_with_delegate = false;
  bool in_kernel_fence = false;
  bool skip_intermediate_buffer_allocation = false;
  bool graph_buffers_donatable = false;
  
  // Async/Tachyon configuration
  bool use_async_api = false;
  bool use_tachyon = false;
  
  // Logging
  bool disable_log_info = false;
  
  static const char* Identifier() { return "darwinn_device_options"; }
};

// Runtime/per-inference options for DarwiNN delegate
struct LiteRtDarwinnRuntimeOptionsT {
  // Power management - frequently changed
  uint32_t inference_power_state = kDefaultInferencePowerState;
  uint32_t inference_memory_power_state = kDefaultInferenceMemoryPowerState;
  
  // Scheduling - may change per workload
  int8_t inference_priority = -1;  // -1 means default
  bool atomic_inference = false;
  
  // Inactive power configuration
  uint32_t inactive_power_state = kDefaultInactivePowerState;
  uint32_t inactive_memory_power_state = kDefaultInactiveMemoryPowerState;
  int64_t inactive_timeout_us = kDefaultInactiveTimeoutUs;
  
  static const char* Identifier() { return "darwinn_runtime_options"; }
};

// Compiler service options for on-device compilation
struct LiteRtDarwinnCompilerOptionsT {
  // Serialized CompilerServiceOptions proto
  std::string serialized_compiler_options;
  
  static const char* Identifier() { return "darwinn_compiler_options"; }
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_RUNTIME_LITERT_DARWINN_OPTIONS_H_