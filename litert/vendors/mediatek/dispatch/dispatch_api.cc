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

#include <cstdio>
#include <optional>
#include <string>

#include "litert/c/litert_any.h"
#include "litert/vendors/cc/options_helper.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/mediatek/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace {

litert::mediatek::NeuronAdapterApi* static_neuron_adapter;
char BuildId[256];

LiteRtEnvironmentOptions static_environment_options;

LiteRtOptions static_options;

}  // namespace

namespace litert {
namespace mediatek {

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

std::optional<std::string> GetSharedLibraryDir(
    LiteRtEnvironmentOptions environment_options) {
  LiteRtAny dispatch_lib_dir_any;
  auto status = LiteRtGetEnvironmentOptionsValue(
      environment_options, kLiteRtEnvOptionTagDispatchLibraryDir,
      &dispatch_lib_dir_any);
  if (status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_INFO, "Failed to get dispatch library dir option: %s",
               LiteRtGetStatusString(status));
    return std::nullopt;
  }
  return std::string(dispatch_lib_dir_any.str_value);
}

LiteRtStatus LiteRtInitialize(LiteRtEnvironment environment,
                              LiteRtOptions options) {
  LiteRtEnvironmentOptions environment_options;
  LiteRtGetEnvironmentOptions(environment, &environment_options);
  static_environment_options = environment_options;
  static_options = options;

  auto [opts, opq_opts, mediatek_opts] =
      litert::ParseOptions<litert::mediatek::MediatekOptions>(options);

  if (!mediatek_opts) {
    LITERT_ASSIGN_OR_RETURN(mediatek_opts,
                            ::litert::mediatek::MediatekOptions::Create());
  }

  auto shared_library_dir_opt = GetSharedLibraryDir(environment_options);

  if (auto neuron_adapter_api = litert::mediatek::NeuronAdapterApi::Create(
          shared_library_dir_opt, mediatek_opts);
      neuron_adapter_api) {
    static_neuron_adapter = neuron_adapter_api->release();
  } else {
    LITERT_LOG(LITERT_INFO, "Initialization failure: %s",
               neuron_adapter_api.Error().Message().c_str());
    return neuron_adapter_api.Error().Status();
  }

  auto get_version = static_neuron_adapter->api().get_version;
  if (!get_version) {
    LITERT_LOG(LITERT_ERROR, "get_version not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  NeuronRuntimeVersion version;
  if (get_version(&version) != NEURON_NO_ERROR) {
    LITERT_LOG(LITERT_ERROR, "Failed to get version");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  LITERT_LOG(LITERT_INFO, "Neuron SDK version: %d.%d.%d", version.major,
             version.minor, version.patch);

  snprintf(BuildId, sizeof(BuildId),
           "MediaTek Dispatch API version %d.%d.%d, NeuronAdaptor API version "
           "%d.%d.%d",
           LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
           LITERT_API_VERSION_PATCH, version.major, version.minor,
           version.patch);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetVendorId(const char** vendor_id) {
  *vendor_id = "MediaTek";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  if (auto context =
          LiteRtDispatchDeviceContextT::Create(*static_neuron_adapter);
      context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
}

LiteRtStatus LiteRtDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetInputRequirements(input_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
               requirements.Error().Message().c_str());
    return requirements.Error().Status();
  }
}

LiteRtStatus LiteRtGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (auto requirements =
          invocation_context->GetOutputRequirements(output_index, *tensor_type);
      requirements) {
    *tensor_buffer_requirements = *requirements;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
               requirements.Error().Message().c_str());
    return requirements.Error().Status();
  }
}

LiteRtStatus LiteRtRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
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

LiteRtStatus LiteRtUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto status =
          device_context->UnregisterTensorBuffer(tensor_buffer_handle);
      !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister tensor buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      *static_neuron_adapter, device_context, exec_type, exec_bytecode_buffer,
      function_name, num_inputs, num_outputs);
  if (!context) {
    LITERT_LOG(LITERT_ERROR, "Failed to create context from context binary: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
  *invocation_context = context->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAttachInput(
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

LiteRtStatus LiteRtAttachOutput(
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

LiteRtStatus LiteRtDetachInput(
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

LiteRtStatus LiteRtDetachOutput(
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

LiteRtStatus LiteRtInvoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Invoke(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke context: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  return kLiteRtStatusOk;
}

}  // namespace mediatek
}  // namespace litert

// /////////////////////////////////////////////////////////////////////////////

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::mediatek::LiteRtInitialize,
    .get_vendor_id = litert::mediatek::LiteRtGetVendorId,
    .get_build_id = litert::mediatek::LiteRtGetBuildId,
    .get_capabilities = litert::mediatek::LiteRtGetCapabilities,
    .device_context_create = litert::mediatek::LiteRtDeviceContextCreate,
    .device_context_destroy = litert::mediatek::LiteRtDeviceContextDestroy,
    .get_input_requirements = litert::mediatek::LiteRtGetInputRequirements,
    .get_output_requirements = litert::mediatek::LiteRtGetOutputRequirements,
    .register_tensor_buffer = litert::mediatek::LiteRtRegisterTensorBuffer,
    .unregister_tensor_buffer = litert::mediatek::LiteRtUnregisterTensorBuffer,
    .invocation_context_create =
        litert::mediatek::LiteRtInvocationContextCreate,
    .invocation_context_destroy =
        litert::mediatek::LiteRtInvocationContextDestroy,
    .attach_input = litert::mediatek::LiteRtAttachInput,
    .attach_output = litert::mediatek::LiteRtAttachOutput,
    .detach_input = litert::mediatek::LiteRtDetachInput,
    .detach_output = litert::mediatek::LiteRtDetachOutput,
    .invoke = litert::mediatek::LiteRtInvoke,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .get_num_metrics = nullptr,
    .get_metric = nullptr,
    .destroy_metrics = nullptr,
    .check_runtime_compatibility = litert::mediatek::CheckRuntimeCompatibility,
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
