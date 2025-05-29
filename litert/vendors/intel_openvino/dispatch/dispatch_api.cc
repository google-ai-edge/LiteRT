// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino/runtime/core.hpp>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/intel_openvino/dispatch/device_context.h"
#include "litert/vendors/intel_openvino/dispatch/invocation_context.h"

namespace {
static std::unique_ptr<ov::Core> core;
}  // namespace

namespace litert {
namespace openvino {

// Initialize the Dispatch API runtime.
// This function should be called before calling any other Dispatch API
// functions.
LiteRtStatus DispatchInitialize(LiteRtEnvironmentOptions environment_options,
                                LiteRtOptions options) {
  core = std::make_unique<ov::Core>();
  std::vector<std::string> availableDevices = core->get_available_devices();
  for (auto&& device : availableDevices)
    LITERT_LOG(LITERT_INFO, "[Openvino]Found device plugin for: %s",
               device.c_str());

  return kLiteRtStatusOk;
}

// Return the vendor id of the Dispatch API runtime.
// This function returns a pointer to a statically allocated string that is the
// ID of vendor providing the Dispatch API runtime.
LiteRtStatus DispatchGetVendorId(const char** vendor_id) {
  *vendor_id = "Intel_Openvino";
  return kLiteRtStatusOk;
}

// Return the build ID of the Dispatch API runtime.
// This function returns a pointer to a statically allocated string that is the
// ID of the Dispatch API runtime build.
LiteRtStatus DispatchGetBuildId(const char** build_id) {
  *build_id = "1.0";
  return kLiteRtStatusOk;
}

// Return the capabilities supported by the Dispatch API runtime as a set of the
// values specified in LiteRtDispatchCapabilities.
LiteRtStatus DispatchGetCapabilities(int* capabilities) {
  // TODO: add support for async later
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

// Create a `LiteRtDispatchDeviceContext` object.
// The returned object is used to talk with the underlying HW. The caller owns
// the memory associated with the context and should call
// LiteRtDispatchDeviceContextDestroy() to release it. Return NULL in case of
// error.
LiteRtStatus DispatchDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  if (auto context = LiteRtDispatchDeviceContextT::Create(*core); context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
}

// Release a `LiteRtDispatchDeviceContext` object.
// The given context should be release only after releasing all associated
// objects.
LiteRtStatus DispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

// Given a tensor type for an invocation context input, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus DispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto result =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      result) {
    *tensor_buffer_requirements = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get input requirements: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

// Given a tensor type for an invocation context output, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus DispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto result =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      result) {
    *tensor_buffer_requirements = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get output requirements: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

// Registers a buffer with the given device context.
// Note: The memory backing the buffer should be valid until
// `LiteRtDispatchUnregisterTensorBuffer` is called.
LiteRtStatus DispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (auto status = device_context->RegisterTensorBuffer(tensor_buffer);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to register buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  } else {
    *tensor_buffer_handle = *status;
    return kLiteRtStatusOk;
  }
}

// Unregisters the registered buffer associated with the given
// `LiteRtTensorBufferHandle`.
// Note: The registered `LiteRtTensorBufferHandle` is supposed to be
// unregistered with this function before the associated `ThrContext` is deleted
// by calling `LiteRtDispatchDeviceContextDestroy`.
LiteRtStatus DispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status =
          device_context->UnregisterTensorBuffer(tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  } else {
    return kLiteRtStatusOk;
  }
}

// Create an invocation context to run a given function from a given
// executable. Parameter `function_name` is required if the provided executable
// includes multiple functions.
LiteRtStatus DispatchInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      *core, *device_context, exec_type, exec_bytecode_buffer, function_name,
      num_inputs, num_outputs);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create context from context binary: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
  *invocation_context = context->release();
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to attach output: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->DetachInput(graph_input_index,
                                                    tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to detach input: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status = invocation_context->DetachOutput(graph_output_index,
                                                     tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to detach output: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Invoke(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke context: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

}  // namespace openvino
}  // namespace litert

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::openvino::DispatchInitialize,
    .get_vendor_id = litert::openvino::DispatchGetVendorId,
    .get_build_id = litert::openvino::DispatchGetBuildId,
    .get_capabilities = litert::openvino::DispatchGetCapabilities,
    .device_context_create = litert::openvino::DispatchDeviceContextCreate,
    .device_context_destroy = litert::openvino::DispatchDeviceContextDestroy,
    .get_input_requirements = litert::openvino::DispatchGetInputRequirements,
    .get_output_requirements = litert::openvino::DispatchGetOutputRequirements,
    .register_tensor_buffer = litert::openvino::DispatchRegisterTensorBuffer,
    .unregister_tensor_buffer =
        litert::openvino::DispatchUnregisterTensorBuffer,
    .invocation_context_create =
        litert::openvino::DispatchInvocationContextCreate,
    .invocation_context_destroy =
        litert::openvino::DispatchInvocationContextDestroy,
    .attach_input = litert::openvino::DispatchAttachInput,
    .attach_output = litert::openvino::DispatchAttachOutput,
    .detach_input = litert::openvino::DispatchDetachInput,
    .detach_output = litert::openvino::DispatchDetachOutput,
    .invoke = litert::openvino::DispatchInvoke,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = nullptr,
    .graph_interface = nullptr,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
