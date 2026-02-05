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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/cc/options_helper.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace {

using ::litert::qnn::QnnManager;

static std::unique_ptr<QnnManager>& QnnManagerStorage() {
  static absl::NoDestructor<std::unique_ptr<QnnManager>> storage;
  return *storage;
}

QnnManager& Qnn() { return *QnnManagerStorage(); }

LiteRtEnvironmentOptions TheEnvironmentOptions = nullptr;

LiteRtOptions TheOptions = nullptr;

char BuildId[256];

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus Initialize(LiteRtEnvironment environment, LiteRtOptions options) {
  LiteRtEnvironmentOptions environment_options;
  LiteRtGetEnvironmentOptions(environment, &environment_options);
  TheEnvironmentOptions = environment_options;
  TheOptions = options;

  const char* dispatch_lib_dir = nullptr;
  if (environment_options) {
    LiteRtAny dispatch_lib_dir_any;
    auto status = LiteRtGetEnvironmentOptionsValue(
        environment_options, kLiteRtEnvOptionTagDispatchLibraryDir,
        &dispatch_lib_dir_any);
    if (status == kLiteRtStatusOk && dispatch_lib_dir_any.str_value) {
      dispatch_lib_dir = dispatch_lib_dir_any.str_value;
    }
  }

  // TODO LUKE confirm where the lib dir is coming from, the
  // "dispatch_library_dir" thing makes no sense Since this should be shared lib
  // for libqnn.so.
  auto [opts, opq_opts, qnn_opts] =
      litert::ParseOptions<litert::qualcomm::QualcommOptions>(TheOptions);

  std::optional<std::string> shared_library_dir_opt =
      dispatch_lib_dir != nullptr
          ? std::make_optional(std::string(dispatch_lib_dir))
          : std::nullopt;

  // TODO(Alen): initialize qnn_options from LiteRtOptions
  ::qnn::Options qnn_options;
  if (qnn_opts) {
    InitQnnOptions(qnn_options, qnn_opts.Value());
  } else {
    LITERT_LOG(LITERT_ERROR,
               "Failed to parse qnn options, using default settings. %s",
               qnn_opts.Error().Message().c_str());
  }
  if (auto qnn_manager = QnnManager::Create(
          /*options=*/qnn_options,
          /*shared_library_dir=*/shared_library_dir_opt,
          /*soc_model*/ std::nullopt);
      !qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().c_str());
    return qnn_manager.Error().Status();
  } else {
    std::swap(QnnManagerStorage(), *qnn_manager);
  }

  Qnn_ApiVersion_t qnn_api_version;
  if (auto status = Qnn().Api()->backendGetApiVersion(&qnn_api_version);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN API version: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  const char* build_id;
  if (auto status = Qnn().Api()->backendGetBuildId(&build_id);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN build ID: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  snprintf(BuildId, sizeof(BuildId),
           "Qualcomm Dispatch API version %d.%d.%d, QNN API version %d.%d.%d, "
           "build id: %s",
           LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
           LITERT_API_VERSION_PATCH, qnn_api_version.coreApiVersion.major,
           qnn_api_version.coreApiVersion.minor,
           qnn_api_version.coreApiVersion.patch, build_id);
  BuildId[sizeof(BuildId) - 1] = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Qualcomm";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = BuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  if (auto context = LiteRtDispatchDeviceContextT::Create(Qnn()); context) {
    *device_context = context->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
               context.Error().Message().c_str());
    return context.Error().Status();
  }
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus GetInputRequirements(
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

LiteRtStatus GetOutputRequirements(
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

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (auto status = device_context->RegisterTensorBuffer(buffer); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to register buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  } else {
    *tensor_buffer_handle = *status;
    return kLiteRtStatusOk;
  }
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  if (auto status = device_context->UnregisterTensorBuffer(handle); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to unregister buffer: %s",
               status.Error().Message().c_str());
    return status.Error().Status();
  } else {
    return kLiteRtStatusOk;
  }
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  auto context = LiteRtDispatchInvocationContextT::Create(
      Qnn(), *device_context, exec_bytecode_buffer, function_name);
  if (!context) {
    LITERT_LOG(LITERT_ERROR,
               "Failed to create context from context binary: %s for function "
               "%s, base address: %p, size: %zu",
               context.Error().Message().c_str(), function_name,
               exec_bytecode_buffer->base_addr, exec_bytecode_buffer->size);
    return context.Error().Status();
  }
  *invocation_context = context->release();
  device_context->SetInvocationContext(*invocation_context);
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
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

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
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

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_RETURN_IF_ERROR(
      invocation_context->DetachInput(graph_input_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_RETURN_IF_ERROR(invocation_context->DetachOutput(
      graph_output_index, tensor_buffer_handle));
  return kLiteRtStatusOk;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (auto status = invocation_context->Execute(); !status) {
    LITERT_LOG(LITERT_ERROR, "Failed to execute invocation context: %s",
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

// /////////////////////////////////////////////////////////////////////////////

LiteRtDispatchInterface TheInterface = {
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/GetInputRequirements,
    /*.get_output_requirements=*/GetOutputRequirements,
    /*.register_tensor_buffer=*/RegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/UnregisterTensorBuffer,
    /*.invocation_context_create=*/InvocationContextCreate,
    /*.invocation_context_destroy=*/InvocationContextDestroy,
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
    /*.start_metrics_collection=*/nullptr,
    /*.stop_metrics_collection=*/nullptr,
    /*.get_num_metrics=*/nullptr,
    /*.get_metric=*/nullptr,
    /*.destroy_metrics=*/nullptr,
    /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
};

LiteRtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                  /*.minor=*/LITERT_API_VERSION_MINOR,
                  /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&TheInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
