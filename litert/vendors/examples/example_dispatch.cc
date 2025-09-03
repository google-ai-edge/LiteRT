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

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/examples/example_common.h"  // IWYU pragma: keep

namespace litert::example {
namespace {

LiteRtStatus GetVendorId(const char** vendor_id) {
  *vendor_id = "Example";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  *build_id = "ExampleBuild";
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus Initialize(LiteRtEnvironmentOptions environment_options,
                        LiteRtOptions options) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  // TODO
  return kLiteRtStatusErrorUnsupported;
}

// /////////////////////////////////////////////////////////////////////////////

LiteRtDispatchInterface ExampleInterface = {
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
};

LiteRtDispatchApi ExampleApi = {
    /*.version=*/{/*.major=*/LITERT_API_VERSION_MAJOR,
                  /*.minor=*/LITERT_API_VERSION_MINOR,
                  /*.patch=*/LITERT_API_VERSION_PATCH},
    /*.interface=*/&ExampleInterface,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace
}  // namespace litert::example

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = ::litert::example::ExampleApi;
  return kLiteRtStatusOk;
}
