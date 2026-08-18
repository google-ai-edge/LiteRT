/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

class LiteRtDispatchDeviceContextT {};
class LiteRtDispatchInvocationContextT {};

namespace litert::arm {

LiteRtStatus Initialize(const LiteRtRuntimeContext* runtime_context,
                        LiteRtEnvironment environment, LiteRtOptions options) {
  (void)runtime_context;
  (void)environment;
  (void)options;
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  if (vendor_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = "Arm(R)";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  if (build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = "1.0";
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  if (capabilities == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(const LiteRtRuntimeContext* runtime_context,
                                 LiteRtOptions options,
                                 LiteRtDispatchDeviceContext* device_context) {
  (void)runtime_context;
  (void)options;
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_context = new LiteRtDispatchDeviceContextT;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  (void)tensor_type;
  (void)tensor_buffer_requirements;
  if (invocation_context == nullptr || input_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  (void)tensor_type;
  (void)tensor_buffer_requirements;
  if (invocation_context == nullptr || output_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  (void)tensor_buffer;
  (void)tensor_buffer_handle;
  if (device_context == nullptr || tensor_buffer_handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus UnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus InvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  (void)runtime_context;
  (void)exec_type;
  (void)exec_bytecode_buffer;
  (void)function_name;
  if (device_context == nullptr || invocation_context == nullptr ||
      num_inputs < 0 || num_outputs < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *invocation_context = nullptr;
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  (void)scheduling_info;
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  (void)options;
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (invocation_context == nullptr || graph_input_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (invocation_context == nullptr || graph_output_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (invocation_context == nullptr || graph_input_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  (void)tensor_buffer_handle;
  if (invocation_context == nullptr || graph_output_index < 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  (void)api_version;
  (void)env;
  (void)options;
  return kLiteRtStatusOk;
}

}  // namespace litert::arm

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::arm::Initialize,
    .get_vendor_id = litert::arm::GetVendorId,
    .get_build_id = litert::arm::GetBuildId,
    .get_capabilities = litert::arm::GetCapabilities,
    .device_context_create = litert::arm::DeviceContextCreate,
    .device_context_destroy = litert::arm::DeviceContextDestroy,
    .get_input_requirements = litert::arm::GetInputRequirements,
    .get_output_requirements = litert::arm::GetOutputRequirements,
    .register_tensor_buffer = litert::arm::RegisterTensorBuffer,
    .unregister_tensor_buffer = litert::arm::UnregisterTensorBuffer,
    .invocation_context_create = litert::arm::InvocationContextCreate,
    .invocation_context_destroy = litert::arm::InvocationContextDestroy,
    .invocation_context_set_scheduling_info =
        litert::arm::InvocationContextSetSchedulingInfo,
    .attach_input = litert::arm::AttachInput,
    .attach_output = litert::arm::AttachOutput,
    .detach_input = litert::arm::DetachInput,
    .detach_output = litert::arm::DetachOutput,
    .invoke = litert::arm::Invoke,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .get_num_metrics = nullptr,
    .get_metric = nullptr,
    .destroy_metrics = nullptr,
    .check_runtime_compatibility = litert::arm::CheckRuntimeCompatibility,
    .invocation_context_set_options = litert::arm::InvocationContextSetOptions,
};

LiteRtDispatchApi TheApi = {
    .version =
        {
            .major = LITERT_API_VERSION_MAJOR,
            .minor = LITERT_API_VERSION_MINOR,
            .patch = LITERT_API_VERSION_PATCH,
        },
    .interface = &TheInterface,
    .async_interface = nullptr,
    .graph_interface = nullptr,
};

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  if (api == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *api = TheApi;
  return kLiteRtStatusOk;
}
