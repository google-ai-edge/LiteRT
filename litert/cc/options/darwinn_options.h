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

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_DARWINN_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_DARWINN_OPTIONS_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "litert/c/options/litert_darwinn_device_options.h"
#include "litert/c/options/litert_darwinn_runtime_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/runtime/litert_darwinn_options.h"

namespace litert {

// C++ wrapper for DarwiNN device options
class DarwinnDeviceOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;
  
  static const char* Discriminator() { 
    return LiteRtDarwinnDeviceOptionsT::Identifier(); 
  }
  
  static absl::string_view Identifier();
  
  // Create a new DarwiNN device options instance
  static Expected<DarwinnDeviceOptions> Create();
  
  // Find DarwiNN device options in an existing opaque options list
  static Expected<DarwinnDeviceOptions> Create(OpaqueOptions& options);
  
  // Device selection setters/getters
  Expected<void> SetDeviceType(const std::string& device_type);
  Expected<std::string> GetDeviceType() const;
  
  Expected<void> SetDevicePath(const std::string& device_path);
  Expected<std::string> GetDevicePath() const;
  
  // Compilation options setters/getters
  Expected<void> SetEnableMultipleSubgraphs(bool enable);
  Expected<bool> GetEnableMultipleSubgraphs() const;
  
  Expected<void> SetCompileIfResize(bool compile_if_resize);
  Expected<bool> GetCompileIfResize() const;
  
  Expected<void> SetAllowCpuFallback(bool allow_cpu_fallback);
  Expected<bool> GetAllowCpuFallback() const;
  
  Expected<void> SetSkipOpFilter(bool skip_op_filter);
  Expected<bool> GetSkipOpFilter() const;
  
  Expected<void> SetNumInterpreters(int num_interpreters);
  Expected<int> GetNumInterpreters() const;
  
  // Memory configuration setters/getters
  Expected<void> SetAvoidBounceBuffer(bool avoid_bounce_buffer);
  Expected<bool> GetAvoidBounceBuffer() const;
  
  Expected<void> SetRegisterGraphDuringModify(bool register_graph);
  Expected<bool> GetRegisterGraphDuringModify() const;
  
  Expected<void> SetInKernelFence(bool in_kernel_fence);
  Expected<bool> GetInKernelFence() const;
  
  Expected<void> SetSkipIntermediateBufferAllocation(bool skip_allocation);
  Expected<bool> GetSkipIntermediateBufferAllocation() const;
  
  Expected<void> SetGraphBuffersDonatable(bool donatable);
  Expected<bool> GetGraphBuffersDonatable() const;
  
  // Async/Tachyon configuration setters/getters
  Expected<void> SetUseAsyncApi(bool use_async_api);
  Expected<bool> GetUseAsyncApi() const;
  
  Expected<void> SetUseTachyon(bool use_tachyon);
  Expected<bool> GetUseTachyon() const;
  
  // Logging setters/getters
  Expected<void> SetDisableLogInfo(bool disable_log_info);
  Expected<bool> GetDisableLogInfo() const;
};

// C++ wrapper for DarwiNN runtime options
class DarwinnRuntimeOptions : public OpaqueOptions {
 public:
  using OpaqueOptions::OpaqueOptions;
  
  static const char* Discriminator() { 
    return LiteRtDarwinnRuntimeOptionsT::Identifier(); 
  }
  
  static absl::string_view Identifier();
  
  // Create a new DarwiNN runtime options instance
  static Expected<DarwinnRuntimeOptions> Create();
  
  // Find DarwiNN runtime options in an existing opaque options list
  static Expected<DarwinnRuntimeOptions> Create(OpaqueOptions& options);
  
  // Power management setters/getters
  Expected<void> SetInferencePowerState(uint32_t power_state);
  Expected<uint32_t> GetInferencePowerState() const;
  
  Expected<void> SetInferenceMemoryPowerState(uint32_t memory_power_state);
  Expected<uint32_t> GetInferenceMemoryPowerState() const;
  
  // Scheduling setters/getters
  Expected<void> SetInferencePriority(int8_t priority);
  Expected<int8_t> GetInferencePriority() const;
  
  Expected<void> SetAtomicInference(bool atomic_inference);
  Expected<bool> GetAtomicInference() const;
  
  // Inactive power configuration setters/getters
  Expected<void> SetInactivePowerState(uint32_t power_state);
  Expected<uint32_t> GetInactivePowerState() const;
  
  Expected<void> SetInactiveMemoryPowerState(uint32_t memory_power_state);
  Expected<uint32_t> GetInactiveMemoryPowerState() const;
  
  Expected<void> SetInactiveTimeoutUs(int64_t timeout_us);
  Expected<int64_t> GetInactiveTimeoutUs() const;
};

// Note: FindOpaqueOptions template specializations are not needed.
// The generic template in litert_opaque_options.h will use Discriminator()
// to find the options in the linked list and then call Create(found_options).

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_DARWINN_OPTIONS_H_