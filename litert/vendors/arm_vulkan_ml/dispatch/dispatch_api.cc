// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/arm_vulkan_ml/dispatch/vulkan_invocation.h"

namespace {
struct RegisteredBuffer {
  LiteRtTensorBuffer buffer;
  LiteRtRankedTensorType type;
  size_t size;
  size_t offset;
  LiteRtTensorBufferHandle handle;
};
struct DeviceState {
  const LiteRtRuntimeContext* runtime;
  absl::Mutex mutex;
  uint64_t next_handle = 1;
  std::unordered_map<LiteRtTensorBufferHandle, std::shared_ptr<RegisteredBuffer>> buffers;
};
using litert::arm_vulkan_ml::TensorDescriptor;
using litert::arm_vulkan_ml::TensorElement;
bool Matches(const TensorDescriptor& desc, const LiteRtRankedTensorType& type) {
  const auto element = desc.element == TensorElement::kInt8 ? kLiteRtElementTypeInt8 :
      desc.element == TensorElement::kFloat32 ? kLiteRtElementTypeFloat32 : kLiteRtElementTypeInt32;
  if (type.element_type != element || type.layout.has_strides ||
      type.layout.rank != desc.shape.size()) return false;
  for (size_t i = 0; i < desc.shape.size(); ++i)
    if (type.layout.dimensions[i] <= 0 || static_cast<uint32_t>(type.layout.dimensions[i]) != desc.shape[i]) return false;
  return true;
}
size_t Bytes(const TensorDescriptor& tensor) {
  size_t size = tensor.element == TensorElement::kInt8 ? 1 : 4;
  for (auto dim : tensor.shape) {
    if (!dim || size > SIZE_MAX / dim) return 0;
    size *= dim;
  }
  return size;
}
}  // namespace

class LiteRtDispatchDeviceContextT {
 public:
  std::shared_ptr<DeviceState> state;
};
class LiteRtDispatchInvocationContextT {
 public:
  std::shared_ptr<DeviceState> state;
  std::shared_ptr<const litert::arm_vulkan_ml::CompiledPartition> artifact;
  std::unique_ptr<litert::arm_vulkan_ml::VulkanInvocation> invocation;
  std::vector<std::shared_ptr<RegisteredBuffer>> inputs, outputs;
};

namespace litert::arm_vulkan_ml {
LiteRtStatus Initialize(const LiteRtRuntimeContext*, LiteRtEnvironment, LiteRtOptions) {
  return kLiteRtStatusOk;
}
LiteRtStatus GetVendorId(const char** value) {
  if (!value) return kLiteRtStatusErrorInvalidArgument;
  *value = "ArmVulkanML"; return kLiteRtStatusOk;
}
LiteRtStatus GetBuildId(const char** build_id) {
  if (build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = "1.0";
  return kLiteRtStatusOk;
}
LiteRtStatus GetCapabilities(int* value) {
  if (!value) return kLiteRtStatusErrorInvalidArgument;
  *value = kLiteRtDispatchCapabilitiesBasic; return kLiteRtStatusOk;
}
LiteRtStatus DeviceContextCreate(const LiteRtRuntimeContext* runtime, LiteRtOptions,
                                 LiteRtDispatchDeviceContext* output) {
  if (!output) return kLiteRtStatusErrorInvalidArgument;
  *output = nullptr;
  if (!runtime || !runtime->create_tensor_buffer_requirements ||
      !runtime->get_tensor_buffer_type || !runtime->get_tensor_buffer_tensor_type ||
      !runtime->get_tensor_buffer_size || !runtime->get_tensor_buffer_offset ||
      !runtime->lock_tensor_buffer || !runtime->unlock_tensor_buffer)
    return kLiteRtStatusErrorInvalidArgument;
  auto context = std::make_unique<LiteRtDispatchDeviceContextT>();
  context->state = std::make_shared<DeviceState>();
  context->state->runtime = runtime;
  *output = context.release(); return kLiteRtStatusOk;
}
LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext context) {
  if (!context) return kLiteRtStatusErrorInvalidArgument;
  delete context; return kLiteRtStatusOk;
}
LiteRtStatus Requirements(LiteRtDispatchInvocationContext context, int index,
                          const LiteRtRankedTensorType* type,
                          LiteRtTensorBufferRequirements* output, bool input) {
  if (!context || !type || !output || index < 0) return kLiteRtStatusErrorInvalidArgument;
  *output = nullptr;
  const auto& descriptors = input ? context->artifact->inputs : context->artifact->outputs;
  if (static_cast<size_t>(index) >= descriptors.size() || !Matches(descriptors[index], *type))
    return kLiteRtStatusErrorInvalidArgument;
  auto bytes = Bytes(descriptors[index]);
  if (!bytes) return kLiteRtStatusErrorInvalidArgument;
  const LiteRtTensorBufferType host = kLiteRtTensorBufferTypeHostMemory;
  return context->state->runtime->create_tensor_buffer_requirements(1, &host, bytes, 0, nullptr, output);
}
LiteRtStatus GetInputRequirements(LiteRtDispatchInvocationContext c, int i,
    const LiteRtRankedTensorType* t, LiteRtTensorBufferRequirements* r) { return Requirements(c,i,t,r,true); }
LiteRtStatus GetOutputRequirements(LiteRtDispatchInvocationContext c, int i,
    const LiteRtRankedTensorType* t, LiteRtTensorBufferRequirements* r) { return Requirements(c,i,t,r,false); }

LiteRtStatus RegisterTensorBuffer(LiteRtDispatchDeviceContext context,
    LiteRtTensorBuffer buffer, LiteRtTensorBufferHandle* output) {
  if (!context || !buffer || !output) return kLiteRtStatusErrorInvalidArgument;
  auto& state = *context->state;
  absl::MutexLock lock(&state.mutex);
  const auto* api = state.runtime;
  LiteRtTensorBufferType type;
  if (api->get_tensor_buffer_type(buffer, &type) != kLiteRtStatusOk ||
      type != kLiteRtTensorBufferTypeHostMemory) return kLiteRtStatusErrorUnsupported;
  auto registration = std::make_shared<RegisteredBuffer>();
  if (api->get_tensor_buffer_tensor_type(buffer, &registration->type) != kLiteRtStatusOk ||
      api->get_tensor_buffer_size(buffer, &registration->size) != kLiteRtStatusOk ||
      api->get_tensor_buffer_offset(buffer, &registration->offset) != kLiteRtStatusOk ||
      registration->offset > registration->size || !state.next_handle)
    return kLiteRtStatusErrorInvalidArgument;
  registration->buffer = buffer;
  registration->handle = state.next_handle++;
  *output = registration->handle;
  state.buffers.emplace(*output, std::move(registration));
  return kLiteRtStatusOk;
}
LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext context, LiteRtTensorBufferHandle handle) {
  if (!context) return kLiteRtStatusErrorInvalidArgument;
  auto& state = *context->state;
  absl::MutexLock lock(&state.mutex);
  auto found = state.buffers.find(handle);
  if (found == state.buffers.end() || found->second.use_count() != 1)
    return kLiteRtStatusErrorInvalidArgument;
  state.buffers.erase(found); return kLiteRtStatusOk;
}
LiteRtStatus InvocationContextCreate(const LiteRtRuntimeContext*,
    LiteRtDispatchDeviceContext device, LiteRtDispatchExecutableType type,
    const LiteRtMemBuffer* buffer, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* output) {
  if (!output) return kLiteRtStatusErrorInvalidArgument;
  *output = nullptr;
  if (!device || !buffer || !buffer->base_addr || !function_name ||
      num_inputs < 0 || num_outputs < 0) return kLiteRtStatusErrorInvalidArgument;
  if (type != kLiteRtDispatchExecutableTypeJitHandle) return kLiteRtStatusErrorUnsupported;
  if (buffer->offset != 0 || buffer->size != 0) return kLiteRtStatusErrorInvalidArgument;
  const auto* executable = static_cast<const ArmVulkanMLJitExecutable*>(buffer->base_addr);
  auto artifact = executable->partition;
  if (!artifact || artifact->inputs.size() != static_cast<size_t>(num_inputs) ||
      artifact->outputs.size() != static_cast<size_t>(num_outputs)) return kLiteRtStatusErrorInvalidArgument;
  auto context = std::make_unique<LiteRtDispatchInvocationContextT>();
  context->state = device->state; context->artifact = std::move(artifact);
  context->inputs.resize(num_inputs); context->outputs.resize(num_outputs);
  const char* option = std::getenv("LITERT_ARM_VULKAN_ML_ALLOW_EMULATION");
  bool allow_emulation = option && !std::strcmp(option, "1");
  std::string error;
  context->invocation = VulkanInvocation::Create(context->artifact, allow_emulation, error);
  if (!context->invocation) {
    std::fprintf(stderr, "ArmVulkanML: %s\n", error.c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *output = context.release(); return kLiteRtStatusOk;
}
LiteRtStatus InvocationContextDestroy(LiteRtDispatchInvocationContext context) {
  if (!context) return kLiteRtStatusErrorInvalidArgument;
  // Destruction must not race an invocation on the same context (API contract).
  delete context; return kLiteRtStatusOk;
}
LiteRtStatus InvocationContextSetSchedulingInfo(LiteRtDispatchInvocationContext c,
    const LiteRtSchedulingInfo*) { return c ? kLiteRtStatusOk : kLiteRtStatusErrorInvalidArgument; }
LiteRtStatus InvocationContextSetOptions(LiteRtDispatchInvocationContext c,
    LiteRtOptions) { return c ? kLiteRtStatusOk : kLiteRtStatusErrorInvalidArgument; }
LiteRtStatus Attach(LiteRtDispatchInvocationContext context, int index,
                    LiteRtTensorBufferHandle handle, bool input, bool detach) {
  if (!context || index < 0) return kLiteRtStatusErrorInvalidArgument;
  absl::MutexLock lock(&context->state->mutex);
  auto& slots = input ? context->inputs : context->outputs;
  const auto& descriptors = input ? context->artifact->inputs : context->artifact->outputs;
  if (static_cast<size_t>(index) >= slots.size()) return kLiteRtStatusErrorInvalidArgument;
  if (detach) {
    if (!slots[index] || slots[index]->handle != handle) return kLiteRtStatusErrorInvalidArgument;
    slots[index].reset(); return kLiteRtStatusOk;
  }
  auto found = context->state->buffers.find(handle);
  if (found == context->state->buffers.end()) return kLiteRtStatusErrorInvalidArgument;
  const auto& registration = found->second;
  if (!Matches(descriptors[index], registration->type) ||
      Bytes(descriptors[index]) > registration->size - registration->offset)
    return kLiteRtStatusErrorInvalidArgument;
  slots[index] = registration; return kLiteRtStatusOk;
}
LiteRtStatus AttachInput(LiteRtDispatchInvocationContext c, int i, LiteRtTensorBufferHandle h) { return Attach(c,i,h,true,false); }
LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext c, int i, LiteRtTensorBufferHandle h) { return Attach(c,i,h,false,false); }
LiteRtStatus DetachInput(LiteRtDispatchInvocationContext c, int i, LiteRtTensorBufferHandle h) { return Attach(c,i,h,true,true); }
LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext c, int i, LiteRtTensorBufferHandle h) { return Attach(c,i,h,false,true); }
LiteRtStatus Invoke(LiteRtDispatchInvocationContext context) {
  if (!context) return kLiteRtStatusErrorInvalidArgument;
  absl::MutexLock lock(&context->state->mutex);
  const auto* api = context->state->runtime;
  struct Locks {
    const LiteRtRuntimeContext* api;
    std::unordered_map<LiteRtTensorBuffer, void*> addresses;
    ~Locks() { for (auto& entry : addresses) api->unlock_tensor_buffer(entry.first); }
  } locks{api, {}};
  auto address = [&](const std::shared_ptr<RegisteredBuffer>& r) -> void* {
    if (!r) return nullptr;
    auto found = locks.addresses.find(r->buffer);
    if (found == locks.addresses.end()) {
      void* base = nullptr;
      if (api->lock_tensor_buffer(r->buffer, &base, kLiteRtTensorBufferLockModeReadWrite) != kLiteRtStatusOk)
        return nullptr;
      found = locks.addresses.emplace(r->buffer, base).first;
    }
    return found->second ? static_cast<uint8_t*>(found->second) + r->offset : nullptr;
  };
  std::vector<const void*> inputs;
  std::vector<void*> outputs;
  for (const auto& r : context->inputs) { auto p = address(r); if (!p) return kLiteRtStatusErrorRuntimeFailure; inputs.push_back(p); }
  for (const auto& r : context->outputs) { auto p = address(r); if (!p) return kLiteRtStatusErrorRuntimeFailure; outputs.push_back(p); }
  std::string error;
  if (!context->invocation->Run(inputs, outputs, error)) {
    std::fprintf(stderr, "ArmVulkanML: %s\n", error.c_str());
    return kLiteRtStatusErrorRuntimeFailure;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  static_cast<void>(env);
  static_cast<void>(options);
  if (api_version.major != LITERT_API_VERSION_MAJOR ||
      api_version.minor < LITERT_API_VERSION_MINOR) {
    return kLiteRtStatusErrorWrongVersion;
  }
  return kLiteRtStatusOk;
}

namespace {

static const LiteRtDispatchInterface_V1 TheBasicInterface = {
    .abi_header =
        {
            .struct_size = sizeof(LiteRtDispatchInterface_V1),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .initialize = litert::arm_vulkan_ml::Initialize,
    .get_vendor_id = litert::arm_vulkan_ml::GetVendorId,
    .get_build_id = litert::arm_vulkan_ml::GetBuildId,
    .get_capabilities = litert::arm_vulkan_ml::GetCapabilities,
    .device_context_create = litert::arm_vulkan_ml::DeviceContextCreate,
    .device_context_destroy = litert::arm_vulkan_ml::DeviceContextDestroy,
    .get_input_requirements = litert::arm_vulkan_ml::GetInputRequirements,
    .get_output_requirements = litert::arm_vulkan_ml::GetOutputRequirements,
    .register_tensor_buffer = litert::arm_vulkan_ml::RegisterTensorBuffer,
    .unregister_tensor_buffer = litert::arm_vulkan_ml::UnregisterTensorBuffer,
    .invocation_context_create = litert::arm_vulkan_ml::InvocationContextCreate,
    .invocation_context_destroy =
        litert::arm_vulkan_ml::InvocationContextDestroy,
    .invocation_context_set_scheduling_info =
        litert::arm_vulkan_ml::InvocationContextSetSchedulingInfo,
    .attach_input = litert::arm_vulkan_ml::AttachInput,
    .attach_output = litert::arm_vulkan_ml::AttachOutput,
    .detach_input = litert::arm_vulkan_ml::DetachInput,
    .detach_output = litert::arm_vulkan_ml::DetachOutput,
    .invoke = litert::arm_vulkan_ml::Invoke,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .get_num_metrics = nullptr,
    .get_metric = nullptr,
    .destroy_metrics = nullptr,
    .check_runtime_compatibility =
        litert::arm_vulkan_ml::CheckRuntimeCompatibility,
    .invocation_context_set_options =
        litert::arm_vulkan_ml::InvocationContextSetOptions,
};

}  // namespace

}  // namespace litert::arm_vulkan_ml

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major >= 1) {
    if (interface_id == kLiteRtInterfaceBasic) {
      *out_interface = &litert::arm_vulkan_ml::TheBasicInterface;
      return kLiteRtStatusOk;
    }
  }
  return kLiteRtStatusErrorUnsupported;
}
