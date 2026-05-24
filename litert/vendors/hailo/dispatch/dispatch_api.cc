#include <cstddef>

#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/hailo/dispatch/device_context.h"
#include "litert/vendors/hailo/dispatch/invocation_context.h"

namespace litert {
namespace hailo {

LiteRtStatus DispatchInitialize(const LiteRtRuntimeContext* runtime_context,
                                LiteRtEnvironment env, LiteRtOptions options) {
  LiteRtEnvironmentOptions environment_options;
  if (runtime_context->get_environment_options(env, &environment_options) == kLiteRtStatusOk) {
    LiteRtPropagateMinLoggerSeverityWithRuntimeContext(runtime_context, environment_options);
  }
  LITERT_LOG(LITERT_INFO, "Initializing Hailo NPU Dispatch API runtime.");
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetVendorId(const char** vendor_id) {
  if (vendor_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = "Hailo";
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetBuildId(const char** build_id) {
  if (build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = "1.0";
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetCapabilities(int* capabilities) {
  if (capabilities == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDeviceContextCreate(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
    LiteRtDispatchDeviceContext* device_context) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto context = LiteRtDispatchDeviceContextT::Create(runtime_context);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: status = %d", context.Error().Status());
    return context.Error().Status();
  }
  *device_context = context->release();
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr || tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->GetInputRequirements(input_index, *tensor_type);
  if (!result) {
    return result.Error().Status();
  }
  *tensor_buffer_requirements = *result;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr || tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->GetOutputRequirements(output_index, *tensor_type);
  if (!result) {
    return result.Error().Status();
  }
  *tensor_buffer_requirements = *result;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (device_context == nullptr || tensor_buffer == nullptr || tensor_buffer_handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = device_context->RegisterTensorBuffer(tensor_buffer);
  if (!result) {
    return result.Error().Status();
  }
  *tensor_buffer_handle = *result;
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = device_context->UnregisterTensorBuffer(tensor_buffer_handle);
  if (!result) {
    return result.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchInvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (device_context == nullptr || exec_bytecode_buffer == nullptr || invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto context = LiteRtDispatchInvocationContextT::Create(
      *device_context, exec_type, exec_bytecode_buffer, function_name, num_inputs, num_outputs);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: status = %d", context.Error().Status());
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

LiteRtStatus DispatchInvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->AttachInput(graph_input_index, tensor_buffer_handle);
  return result ? kLiteRtStatusOk : result.Error().Status();
}

LiteRtStatus DispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->AttachOutput(graph_output_index, tensor_buffer_handle);
  return result ? kLiteRtStatusOk : result.Error().Status();
}

LiteRtStatus DispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->DetachInput(graph_input_index, tensor_buffer_handle);
  return result ? kLiteRtStatusOk : result.Error().Status();
}

LiteRtStatus DispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->DetachOutput(graph_output_index, tensor_buffer_handle);
  return result ? kLiteRtStatusOk : result.Error().Status();
}

LiteRtStatus DispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto result = invocation_context->Invoke();
  return result ? kLiteRtStatusOk : result.Error().Status();
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  return kLiteRtStatusOk;
}

}  // namespace hailo
}  // namespace litert

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::hailo::DispatchInitialize,
    .get_vendor_id = litert::hailo::DispatchGetVendorId,
    .get_build_id = litert::hailo::DispatchGetBuildId,
    .get_capabilities = litert::hailo::DispatchGetCapabilities,
    .device_context_create = litert::hailo::DispatchDeviceContextCreate,
    .device_context_destroy = litert::hailo::DispatchDeviceContextDestroy,
    .get_input_requirements = litert::hailo::DispatchGetInputRequirements,
    .get_output_requirements = litert::hailo::DispatchGetOutputRequirements,
    .register_tensor_buffer = litert::hailo::DispatchRegisterTensorBuffer,
    .unregister_tensor_buffer = litert::hailo::DispatchUnregisterTensorBuffer,
    .invocation_context_create = litert::hailo::DispatchInvocationContextCreate,
    .invocation_context_destroy = litert::hailo::DispatchInvocationContextDestroy,
    .invocation_context_set_scheduling_info = litert::hailo::DispatchInvocationContextSetSchedulingInfo,
    .attach_input = litert::hailo::DispatchAttachInput,
    .attach_output = litert::hailo::DispatchAttachOutput,
    .detach_input = litert::hailo::DispatchDetachInput,
    .detach_output = litert::hailo::DispatchDetachOutput,
    .invoke = litert::hailo::DispatchInvoke,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .get_num_metrics = nullptr,
    .get_metric = nullptr,
    .destroy_metrics = nullptr,
    .check_runtime_compatibility = litert::hailo::CheckRuntimeCompatibility,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = nullptr,
    .graph_interface = nullptr,
    .tensor_buffer_handlers_def = nullptr,
};

}  // namespace

extern "C" {
LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  if (api == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *api = TheApi;
  return kLiteRtStatusOk;
}
}
