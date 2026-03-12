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
//
// Copyright (C) 2026 Samsung Electronics Co. LTD. 
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/samsung/dispatch/enn_manager.h"
#include "litert/vendors/samsung/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/samsung/dispatch/litert_dispatch_invocation_context.h"

namespace {

static std::unique_ptr<::litert::samsung::EnnManager> static_enn_manager;

char BuildId[256];

LiteRtEnvironmentOptions static_environment_options;

LiteRtOptions static_options;

} // namespace

// Initialize the Dispatch API runtime.
//
// This function should be called before calling any other Dispatch API
// functions.
LiteRtStatus
LiteRtSamsungInitialize(LiteRtEnvironmentOptions environment_options,
                        LiteRtOptions options) {
  if (auto enn_manager = ::litert::samsung::EnnManager::Create(); enn_manager) {
    LITERT_LOG(LITERT_INFO, "Initialize done");
    static_enn_manager.reset(enn_manager->release());
    LITERT_LOG(LITERT_INFO, "Static prepared.");
  } else {
    LITERT_LOG(LITERT_INFO, "Failed to initialize: %s",
               enn_manager.Error().Message().c_str());
    return enn_manager.Error().Status();
  }

  snprintf(BuildId, sizeof(BuildId),
           "Samsung Dispatch API version %d.%d.%d, ENN API version %d.%d.%d",
           LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
           LITERT_API_VERSION_PATCH, 0, 1, 0);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

// Return the version of the Dispatch API runtime.
LiteRtStatus LiteRtSamsungGetApiVersion(LiteRtApiVersion *api_version) {}

// Return the vendor id, the Samsung Dispatch runtime.
LiteRtStatus LiteRtSamsungGetVendorId(const char **vendor_id) {
  *vendor_id = "Samsung";
  return kLiteRtStatusOk;
}

// Return the build ID of the Dispatch API runtime.
LiteRtStatus LiteRtSamsungGetBuildId(const char **build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

// Return the capabilities supported by Samsung dispatch API runtime.
// Only Support basic now.
LiteRtStatus LiteRtSamsungGetCapabilities(int *capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungDeviceContextCreate(LiteRtDispatchDeviceContext *device_context) {
  if (auto context =
          LiteRtDispatchDeviceContextT::Create(static_enn_manager.get());
      context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
}

LiteRtStatus
LiteRtSamsungDeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSamsungGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType *tensor_type,
    LiteRtTensorBufferRequirements *tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    return requirements.Error().Status();
  }
}

LiteRtStatus LiteRtSamsungGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType *tensor_type,
    LiteRtTensorBufferRequirements *tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    return requirements.Error().Status();
  }
}

// Registers a buffer with the given device context.
LiteRtStatus LiteRtSamsungRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle *tensor_buffer_handle) {
  if (auto result = device_context->RegisterTensorBuffer(tensor_buffer);
      result) {
    *tensor_buffer_handle = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to register tensor buffer: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus LiteRtSamsungUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status =
          device_context->UnregisterTensorBuffer(tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister tensor buffer %ll",
               tensor_buffer_handle);
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

// Create an invocation context to run a given function from a given
// executable. Parameter `function_name` is required if the provided executable
// includes multiple functions.
LiteRtStatus LiteRtSamsungInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext *invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      static_enn_manager.get(), device_context, exec_type, exec_bytecode_buffer,
      function_name, num_inputs, num_outputs);

  if (!context) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to create context from context binary: %s for function "
               "%s, base address: %p, size: %zu",
               context.Error().Message().c_str(), function_name,
               exec_bytecode_buffer->base_addr, exec_bytecode_buffer->size);
    return context.Error().Status();
  }
  *invocation_context = context->release();

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSamsungInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungAttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungAttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach output buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungDetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_RETURN_IF_ERROR(
      invocation_context->DetachInput(graph_input_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungDetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_RETURN_IF_ERROR(invocation_context->DetachOutput(
      graph_output_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSamsungInvoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Invoke(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to execute invocation context: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////

LiteRtDispatchInterface TheInterface = {
    /*.initialize=*/LiteRtSamsungInitialize,
    /*.get_vendor_id=*/LiteRtSamsungGetVendorId,
    /*.get_build_id=*/LiteRtSamsungGetBuildId,
    /*.get_capabilities=*/LiteRtSamsungGetCapabilities,
    /*.device_context_create=*/LiteRtSamsungDeviceContextCreate,
    /*.device_context_destroy=*/LiteRtSamsungDeviceContextDestroy,
    /*.get_input_requirements=*/LiteRtSamsungGetInputRequirements,
    /*.get_output_requirements=*/LiteRtSamsungGetOutputRequirements,
    /*.register_tensor_buffer=*/LiteRtSamsungRegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/LiteRtSamsungUnregisterTensorBuffer,
    /*.invocation_context_create=*/LiteRtSamsungInvocationContextCreate,
    /*.invocation_context_destroy=*/LiteRtSamsungInvocationContextDestroy,
    /*.attach_input=*/LiteRtSamsungAttachInput,
    /*.attach_output=*/LiteRtSamsungAttachOutput,
    /*.detach_input=*/LiteRtSamsungDetachInput,
    /*.detach_output=*/LiteRtSamsungDetachOutput,
    /*.invoke=*/LiteRtSamsungInvoke,
};

LiteRtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                  /*.minor=*/LITERT_API_VERSION_MINOR,
                  /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&TheInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi *api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
