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

#include "litert/cc/options/darwinn_options.h"

#include <cstring>

#include "absl/strings/string_view.h"

namespace litert {

// DarwinnDeviceOptions implementation

absl::string_view DarwinnDeviceOptions::Identifier() {
  return LiteRtGetDarwinnDeviceOptionsIdentifier();
}

Expected<DarwinnDeviceOptions> DarwinnDeviceOptions::Create() {
  LiteRtOpaqueOptions opaque_options = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtCreateDarwinnDeviceOptions(&opaque_options));
  
  DarwinnDeviceOptions options(opaque_options, OwnHandle::kYes);
  return options;
}

Expected<DarwinnDeviceOptions> DarwinnDeviceOptions::Create(OpaqueOptions& options) {
  LITERT_ASSIGN_OR_RETURN(absl::string_view original_identifier,
                          options.GetIdentifier());
  LITERT_RETURN_IF_ERROR(original_identifier == Identifier(),
                         ErrorStatusBuilder::InvalidArgument())
      << "Cannot create DarwiNN device options from an opaque options object that doesn't "
         "already hold DarwiNN device options.";
  LiteRtOpaqueOptions opaque_options = options.Get();
  return DarwinnDeviceOptions(opaque_options, OwnHandle::kNo);
}

// Device selection setters/getters
Expected<void> DarwinnDeviceOptions::SetDeviceType(const std::string& device_type) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnDeviceType(darwinn_options, device_type.c_str()));
  return {};
}

Expected<std::string> DarwinnDeviceOptions::GetDeviceType() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  const char* device_type = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnDeviceType(darwinn_options, &device_type));
  return std::string(device_type);
}

Expected<void> DarwinnDeviceOptions::SetDevicePath(const std::string& device_path) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnDevicePath(
          darwinn_options, 
          device_path.c_str()));
  return {};
}

Expected<std::string> DarwinnDeviceOptions::GetDevicePath() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  const char* device_path = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnDevicePath(darwinn_options, &device_path));
  return std::string(device_path);
}

// Compilation options setters/getters
Expected<void> DarwinnDeviceOptions::SetEnableMultipleSubgraphs(bool enable) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnEnableMultipleSubgraphs(
          darwinn_options, enable));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetEnableMultipleSubgraphs() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool enable = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnEnableMultipleSubgraphs(
          darwinn_options, &enable));
  return enable;
}

Expected<void> DarwinnDeviceOptions::SetCompileIfResize(bool compile_if_resize) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnCompileIfResize(
          darwinn_options, 
          compile_if_resize));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetCompileIfResize() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool compile_if_resize = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnCompileIfResize(
          darwinn_options, 
          &compile_if_resize));
  return compile_if_resize;
}

Expected<void> DarwinnDeviceOptions::SetAllowCpuFallback(bool allow_cpu_fallback) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnAllowCpuFallback(
          darwinn_options, 
          allow_cpu_fallback));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetAllowCpuFallback() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool allow_cpu_fallback = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnAllowCpuFallback(
          darwinn_options, 
          &allow_cpu_fallback));
  return allow_cpu_fallback;
}

Expected<void> DarwinnDeviceOptions::SetSkipOpFilter(bool skip_op_filter) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnSkipOpFilter(
          darwinn_options, 
          skip_op_filter));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetSkipOpFilter() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool skip_op_filter = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnSkipOpFilter(
          darwinn_options, 
          &skip_op_filter));
  return skip_op_filter;
}

Expected<void> DarwinnDeviceOptions::SetNumInterpreters(int num_interpreters) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnNumInterpreters(
          darwinn_options, 
          num_interpreters));
  return {};
}

Expected<int> DarwinnDeviceOptions::GetNumInterpreters() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  int num_interpreters = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnNumInterpreters(
          darwinn_options, 
          &num_interpreters));
  return num_interpreters;
}

// Memory configuration setters/getters
Expected<void> DarwinnDeviceOptions::SetAvoidBounceBuffer(bool avoid_bounce_buffer) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnAvoidBounceBuffer(
          darwinn_options, 
          avoid_bounce_buffer));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetAvoidBounceBuffer() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool avoid_bounce_buffer = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnAvoidBounceBuffer(
          darwinn_options, 
          &avoid_bounce_buffer));
  return avoid_bounce_buffer;
}

Expected<void> DarwinnDeviceOptions::SetRegisterGraphDuringModify(bool register_graph) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnRegisterGraphDuringModify(
          darwinn_options, 
          register_graph));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetRegisterGraphDuringModify() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool register_graph = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnRegisterGraphDuringModify(
          darwinn_options, 
          &register_graph));
  return register_graph;
}

Expected<void> DarwinnDeviceOptions::SetInKernelFence(bool in_kernel_fence) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInKernelFence(
          darwinn_options, 
          in_kernel_fence));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetInKernelFence() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool in_kernel_fence = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInKernelFence(
          darwinn_options, 
          &in_kernel_fence));
  return in_kernel_fence;
}

Expected<void> DarwinnDeviceOptions::SetSkipIntermediateBufferAllocation(bool skip_allocation) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnSkipIntermediateBufferAllocation(
          darwinn_options, 
          skip_allocation));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetSkipIntermediateBufferAllocation() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool skip_allocation = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnSkipIntermediateBufferAllocation(
          darwinn_options, 
          &skip_allocation));
  return skip_allocation;
}

Expected<void> DarwinnDeviceOptions::SetGraphBuffersDonatable(bool donatable) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnGraphBuffersDonatable(
          darwinn_options, 
          donatable));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetGraphBuffersDonatable() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool donatable = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnGraphBuffersDonatable(
          darwinn_options, 
          &donatable));
  return donatable;
}

// Async/Tachyon configuration setters/getters
Expected<void> DarwinnDeviceOptions::SetUseAsyncApi(bool use_async_api) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnUseAsyncApi(
          darwinn_options, 
          use_async_api));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetUseAsyncApi() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool use_async_api = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnUseAsyncApi(
          darwinn_options, 
          &use_async_api));
  return use_async_api;
}

Expected<void> DarwinnDeviceOptions::SetUseTachyon(bool use_tachyon) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnUseTachyon(
          darwinn_options, 
          use_tachyon));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetUseTachyon() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool use_tachyon = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnUseTachyon(
          darwinn_options, 
          &use_tachyon));
  return use_tachyon;
}

// Logging setters/getters
Expected<void> DarwinnDeviceOptions::SetDisableLogInfo(bool disable_log_info) {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnDisableLogInfo(
          darwinn_options, 
          disable_log_info));
  return {};
}

Expected<bool> DarwinnDeviceOptions::GetDisableLogInfo() const {
  LiteRtDarwinnDeviceOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnDeviceOptions(Get(), &darwinn_options));
  bool disable_log_info = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnDisableLogInfo(
          darwinn_options, 
          &disable_log_info));
  return disable_log_info;
}

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

Expected<DarwinnRuntimeOptions> DarwinnRuntimeOptions::Create(OpaqueOptions& options) {
  LITERT_ASSIGN_OR_RETURN(absl::string_view original_identifier,
                          options.GetIdentifier());
  LITERT_RETURN_IF_ERROR(original_identifier == Identifier(),
                         ErrorStatusBuilder::InvalidArgument())
      << "Cannot create DarwiNN runtime options from an opaque options object that doesn't "
         "already hold DarwiNN runtime options.";
  LiteRtOpaqueOptions opaque_options = options.Get();
  return DarwinnRuntimeOptions(opaque_options, OwnHandle::kNo);
}

// Power management setters/getters
Expected<void> DarwinnRuntimeOptions::SetInferencePowerState(uint32_t power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInferencePowerState(
          darwinn_options, 
          power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInferencePowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t power_state = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInferencePowerState(
          darwinn_options, 
          &power_state));
  return power_state;
}

Expected<void> DarwinnRuntimeOptions::SetInferenceMemoryPowerState(uint32_t memory_power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInferenceMemoryPowerState(
          darwinn_options, 
          memory_power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInferenceMemoryPowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t memory_power_state = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInferenceMemoryPowerState(
          darwinn_options, 
          &memory_power_state));
  return memory_power_state;
}

// Scheduling setters/getters
Expected<void> DarwinnRuntimeOptions::SetInferencePriority(int8_t priority) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInferencePriority(
          darwinn_options, 
          priority));
  return {};
}

Expected<int8_t> DarwinnRuntimeOptions::GetInferencePriority() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  int8_t priority = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInferencePriority(
          darwinn_options, 
          &priority));
  return priority;
}

Expected<void> DarwinnRuntimeOptions::SetAtomicInference(bool atomic_inference) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnAtomicInference(
          darwinn_options, 
          atomic_inference));
  return {};
}

Expected<bool> DarwinnRuntimeOptions::GetAtomicInference() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  bool atomic_inference = false;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnAtomicInference(
          darwinn_options, 
          &atomic_inference));
  return atomic_inference;
}

// Inactive power configuration setters/getters
Expected<void> DarwinnRuntimeOptions::SetInactivePowerState(uint32_t power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInactivePowerState(
          darwinn_options, 
          power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInactivePowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t power_state = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInactivePowerState(
          darwinn_options, 
          &power_state));
  return power_state;
}

Expected<void> DarwinnRuntimeOptions::SetInactiveMemoryPowerState(uint32_t memory_power_state) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInactiveMemoryPowerState(
          darwinn_options, 
          memory_power_state));
  return {};
}

Expected<uint32_t> DarwinnRuntimeOptions::GetInactiveMemoryPowerState() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  uint32_t memory_power_state = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInactiveMemoryPowerState(
          darwinn_options, 
          &memory_power_state));
  return memory_power_state;
}

Expected<void> DarwinnRuntimeOptions::SetInactiveTimeoutUs(int64_t timeout_us) {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  LITERT_RETURN_IF_ERROR(
      LiteRtSetDarwinnInactiveTimeoutUs(
          darwinn_options, 
          timeout_us));
  return {};
}

Expected<int64_t> DarwinnRuntimeOptions::GetInactiveTimeoutUs() const {
  LiteRtDarwinnRuntimeOptions darwinn_options;
  LITERT_RETURN_IF_ERROR(LiteRtFindDarwinnRuntimeOptions(Get(), &darwinn_options));
  int64_t timeout_us = 0;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetDarwinnInactiveTimeoutUs(
          darwinn_options, 
          &timeout_us));
  return timeout_us;
}

}  // namespace litert