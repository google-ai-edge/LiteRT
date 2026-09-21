// Copyright (C) 2023 Amlogic, Inc. All rights reserved.
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

#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

#include "litert/vendors/aml/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/aml/dispatch/litert_dispatch_invocation_context.h"

namespace
{

  LiteRtEnvironmentOptions TheEnvironmentOptions = nullptr;

  LiteRtOptions TheOptions = nullptr;

  char BuildId[256];

  // /////////////////////////////////////////////////////////////////////////////
  // Basic Execution API
  // /////////////////////////////////////////////////////////////////////////////

  LiteRtStatus Initialize(LiteRtEnvironment environment, LiteRtOptions options)
  {
    LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtDispatchInitialize enter");
    LiteRtEnvironmentOptions environment_options = nullptr;
    if (environment != nullptr)
    {
      LiteRtGetEnvironmentOptions(environment, &environment_options);
    }
    TheEnvironmentOptions = environment_options;
    TheOptions = options;

    if (environment_options != nullptr)
    {
      LiteRtAny dispatch_lib_dir_any;
      auto status = LiteRtGetEnvironmentOptionsValue(
          environment_options, kLiteRtEnvOptionTagDispatchLibraryDir,
          &dispatch_lib_dir_any);
      if (status == kLiteRtStatusOk &&
          dispatch_lib_dir_any.type == kLiteRtAnyTypeString &&
          dispatch_lib_dir_any.str_value != nullptr)
      {
        (void)dispatch_lib_dir_any.str_value;
      }
    }

    snprintf(BuildId, sizeof(BuildId), "AML Dispatch API %d.%d.%d",
             LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
             LITERT_API_VERSION_PATCH);
    BuildId[sizeof(BuildId) - 1] = '\0';

    LITERT_LOG(LITERT_DEBUG, "[AML] LiteRtDispatchInitialize Leave");

    return kLiteRtStatusOk;
  }

  LiteRtStatus GetVendorId(const char **vendor_id)
  {
    *vendor_id = "Aml";
    return kLiteRtStatusOk;
  }

  LiteRtStatus GetBuildId(const char **build_id)
  {
    *build_id = BuildId;
    return kLiteRtStatusOk;
  }

  LiteRtStatus GetCapabilities(int *capabilities)
  {
    *capabilities = kLiteRtDispatchCapabilitiesBasic;
    return kLiteRtStatusOk;
  }

  LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext *device_context)
  {
    LITERT_LOG(LITERT_DEBUG, "[AML] DeviceContextCreate Enter");
    if (auto context = LiteRtDispatchDeviceContextT::Create(); context)
    {
      *device_context = context->release();
      return kLiteRtStatusOk;
    }
    else
    {
      LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
                 context.Error().Message().c_str());
      return context.Error().Status();
    }
  }

  LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context)
  {
    delete device_context;
    return kLiteRtStatusOk;
  }

  LiteRtStatus GetInputRequirements(
      LiteRtDispatchInvocationContext invocation_context, int input_index,
      const LiteRtRankedTensorType *tensor_type,
      LiteRtTensorBufferRequirements *tensor_buffer_requirements)
  {
    LITERT_LOG(LITERT_DEBUG, "[AML] GetInputRequirements Enter");
    if (auto requirements =
            invocation_context->GetInputRequirements(input_index, *tensor_type);
        requirements)
    {
      *tensor_buffer_requirements = *requirements;
      return kLiteRtStatusOk;
    }
    else
    {
      LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
                 requirements.Error().Message().c_str());
      return requirements.Error().Status();
    }
  }

  LiteRtStatus GetOutputRequirements(
      LiteRtDispatchInvocationContext invocation_context, int output_index,
      const LiteRtRankedTensorType *tensor_type,
      LiteRtTensorBufferRequirements *tensor_buffer_requirements)
  {
    LITERT_LOG(LITERT_DEBUG, "[AML] GetOutputRequirements Enter");
    if (auto requirements =
            invocation_context->GetOutputRequirements(output_index, *tensor_type);
        requirements)
    {
      *tensor_buffer_requirements = *requirements;
      return kLiteRtStatusOk;
    }
    else
    {
      LITERT_LOG(LITERT_ERROR, "Failed to get tensor buffer requirements: %s",
                 requirements.Error().Message().c_str());
      return requirements.Error().Status();
    }
  }

  LiteRtStatus RegisterTensorBuffer(
      LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
      LiteRtTensorBufferHandle *tensor_buffer_handle)
  {
    LITERT_LOG(LITERT_DEBUG, "[AML] RegisterTensorBuffer Enter");
    if (auto status = device_context->RegisterTensorBuffer(buffer); !status)
    {
      LITERT_LOG(LITERT_ERROR, "Failed to register buffer: %s",
                 status.Error().Message().c_str());
      return status.Error().Status();
    }
    else
    {
      *tensor_buffer_handle = *status;
      return kLiteRtStatusOk;
    }
  }

  LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                      LiteRtTensorBufferHandle handle)
  {
    if (auto status = device_context->UnregisterTensorBuffer(handle); !status)
    {
      LITERT_LOG(LITERT_ERROR, "Failed to unregister buffer: %s",
                 status.Error().Message().c_str());
      return status.Error().Status();
    }
    else
    {
      return kLiteRtStatusOk;
    }
  }

  LiteRtStatus InvocationContextCreate(
      LiteRtDispatchDeviceContext device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer *exec_bytecode_buffer, const char *function_name,
      int num_inputs, int num_outputs,
      LiteRtDispatchInvocationContext *invocation_context)
  {
    (void)exec_type;
    LITERT_LOG(LITERT_DEBUG,
               "[AML] InvocationContextCreate num_inputs=%d num_outputs=%d",
               num_inputs, num_outputs);
    auto context = LiteRtDispatchInvocationContextT::Create(
        *device_context, exec_bytecode_buffer, function_name, num_inputs, num_outputs);
    if (!context)
    {
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
      LiteRtDispatchInvocationContext invocation_context)
  {
    delete invocation_context;
    return kLiteRtStatusOk;
  }

  LiteRtStatus InvocationContextSetSchedulingInfo(
      LiteRtDispatchInvocationContext invocation_context,
      const LiteRtSchedulingInfo *scheduling_info)
  {
    if (invocation_context == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus InvocationContextSetOptions(
      LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options)
  {
    if (invocation_context == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                           int graph_input_index,
                           LiteRtTensorBufferHandle tensor_buffer_handle)
  {
    if (auto status = invocation_context->AttachInput(graph_input_index,
                                                      tensor_buffer_handle);
        !status)
    {
      LITERT_LOG(LITERT_ERROR, "Failed to attach input buffer: %s",
                 status.Error().Message().c_str());
      return status.Error().Status();
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                            int graph_output_index,
                            LiteRtTensorBufferHandle tensor_buffer_handle)
  {
    if (auto status = invocation_context->AttachOutput(graph_output_index,
                                                       tensor_buffer_handle);
        !status)
    {
      LITERT_LOG(LITERT_ERROR, "Failed to attach output buffer: %s",
                 status.Error().Message().c_str());
      return status.Error().Status();
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                           int graph_input_index,
                           LiteRtTensorBufferHandle tensor_buffer_handle)
  {
    LITERT_RETURN_IF_ERROR(
        invocation_context->DetachInput(graph_input_index, tensor_buffer_handle));
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                            int graph_output_index,
                            LiteRtTensorBufferHandle tensor_buffer_handle)
  {
    LITERT_RETURN_IF_ERROR(invocation_context->DetachOutput(
        graph_output_index, tensor_buffer_handle));
    return kLiteRtStatusOk;
  }

  LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context)
  {
    if (auto status = invocation_context->Execute(); !status)
    {
      LITERT_LOG(LITERT_ERROR, "Failed to execute invocation context: %s",
                 status.Error().Message().c_str());
      return status.Error().Status();
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus StartMetricsCollection(
      LiteRtDispatchInvocationContext invocation_context, int detail_level)
  {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus StopMetricsCollection(
      LiteRtDispatchInvocationContext invocation_context,
      LiteRtDispatchMetrics *metrics)
  {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus GetNumMetrics(LiteRtDispatchMetrics metrics, int *num_metrics)
  {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus GetMetric(LiteRtDispatchMetrics metrics, int metric_index,
                         LiteRtMetric *metric)
  {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus DestroyMetrics(LiteRtDispatchMetrics metrics)
  {
    return kLiteRtStatusErrorUnsupported;
  }

  LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                         LiteRtEnvironmentOptions env,
                                         LiteRtOptions options)
  {
    static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                   LITERT_API_VERSION_MINOR,
                                                   LITERT_API_VERSION_PATCH};
    if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
      return kLiteRtStatusErrorUnsupportedCompilerVersion;
    }
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
      /*.invocation_context_set_scheduling_info=*/
      InvocationContextSetSchedulingInfo,
      /*.attach_input=*/AttachInput,
      /*.attach_output=*/AttachOutput,
      /*.detach_input=*/DetachInput,
      /*.detach_output=*/DetachOutput,
      /*.invoke=*/Invoke,
      /*.start_metrics_collection=*/StartMetricsCollection,
      /*.stop_metrics_collection=*/StopMetricsCollection,
      /*.get_num_metrics=*/GetNumMetrics,
      /*.get_metric=*/GetMetric,
      /*.destroy_metrics=*/DestroyMetrics,
      /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
      /*.invocation_context_set_options=*/InvocationContextSetOptions,
  };

  LiteRtDispatchApi TheApi = {
      /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                    /*.minor=*/LITERT_API_VERSION_MINOR,
                    /*.patch=*/LITERT_API_VERSION_PATCH},
      /*.interface=*/&TheInterface,
      /*.async_interface=*/nullptr,
      /*.graph_interface=*/nullptr,
  };

} // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi *api)
{
  *api = TheApi;
  return kLiteRtStatusOk;
}
