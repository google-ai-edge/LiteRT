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

#include "litert/vendors/google_tensor/dispatch/dispatch_api.h"

#include "litert/c/internal/litert_scheduling_info.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif
#include <cstddef>
#include <optional>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_config.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace litert::google_tensor {

namespace {

constexpr size_t kEdgeTpuPadding = 64;

template <typename X, typename Align>
constexpr auto Pad(X x, Align align) {
  return ((x + align - 1) / align) * align;
}

LiteRtStatus CreateTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferRequirements& requirements) {
  if (tensor_type.layout.has_strides) {
    LITERT_LOG(LITERT_ERROR, "Tensor strides are not supported");
    return kLiteRtStatusErrorUnsupported;
  }

  LITERT_ASSIGN_OR_RETURN(size_t size_bytes,
                          litert::internal::GetNumPackedBytes(tensor_type));

  return LiteRtCreateTensorBufferRequirements(
      GetTheSupportedTensorBufferTypes().size(),
      GetTheSupportedTensorBufferTypes().data(),
      Pad(size_bytes, kEdgeTpuPadding), /*num_strides=*/0,
      /*strides=*/nullptr, &requirements);
}

}  // namespace

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus Initialize(LiteRtEnvironment env, LiteRtOptions options) {
  GT_LOG_RETURN_IF_SB_ERROR(thrInitialize(), "Failed to initialize SB");
  LiteRtEnvironmentOptions environment_options;
  LiteRtGetEnvironmentOptions(env, &environment_options);
  return InitializeDispatchApiConfig(environment_options, options);
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  GT_LOG_RETURN_IF_NULL(vendor_id);

  *vendor_id = "Google";
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  GT_LOG_RETURN_IF_NULL(build_id);

  *build_id = GetTheBuildId();
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  GT_LOG_RETURN_IF_NULL(capabilities);

  *capabilities = GetTheCapabilities();
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(LiteRtDispatchDeviceContext* device_context) {
  GT_LOG_RETURN_IF_NULL(device_context);

  return LiteRtDispatchDeviceContextT::Create(*device_context);
}

LiteRtStatus DeviceContextDestroy(LiteRtDispatchDeviceContext device_context) {
  GT_LOG_RETURN_IF_NULL(device_context);

  return device_context->Destroy();
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  GT_LOG_RETURN_IF_NULL(tensor_type);
  GT_LOG_RETURN_IF_NULL(tensor_buffer_requirements);

  return CreateTensorBufferRequirements(*tensor_type,
                                        *tensor_buffer_requirements);
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  GT_LOG_RETURN_IF_NULL(tensor_type);
  GT_LOG_RETURN_IF_NULL(tensor_buffer_requirements);

  return CreateTensorBufferRequirements(*tensor_type,
                                        *tensor_buffer_requirements);
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  GT_LOG_RETURN_IF_NULL(device_context);
  GT_LOG_RETURN_IF_NULL(tensor_buffer_handle);

  return device_context->RegisterTensorBuffer(buffer, *tensor_buffer_handle);
}

LiteRtStatus UnregisterTensorBuffer(LiteRtDispatchDeviceContext device_context,
                                    LiteRtTensorBufferHandle handle) {
  GT_LOG_RETURN_IF_NULL(device_context);

  return device_context->UnregisterTensorBuffer(handle);
}

LiteRtStatus InvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  GT_LOG_RETURN_IF_NULL(exec_bytecode_buffer);
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return LiteRtDispatchInvocationContextT::CreateFromBytecode(
      device_context, exec_type, *exec_bytecode_buffer, function_name,
      num_inputs, num_outputs, *invocation_context);
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->Destroy();
}

LiteRtStatus InvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  GT_LOG_RETURN_IF_NULL(invocation_context);
  return invocation_context->SetRunOptions(options);
}

LiteRtStatus InvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  GT_LOG_RETURN_IF_NULL(invocation_context);
  return invocation_context->SetSchedulingInfo(scheduling_info);
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->AttachInput(graph_input_index,
                                         tensor_buffer_handle);
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->AttachOutput(graph_output_index,
                                          tensor_buffer_handle);
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->DetachInput(graph_input_index,
                                         tensor_buffer_handle);
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->DetachOutput(graph_output_index,
                                          tensor_buffer_handle);
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->Invoke();
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus AttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->AttachInputEvent(graph_input_index, input_event);
}

LiteRtStatus InvokeAsync(LiteRtDispatchInvocationContext invocation_context,
                         int num_output_events, LiteRtEvent* output_events) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->InvokeAsync(
      absl::MakeSpan(output_events, num_output_events));
}

// /////////////////////////////////////////////////////////////////////////////
// Metrics API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus StartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return invocation_context->StartMetricsCollection(detail_level);
}

LiteRtStatus StopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  GT_LOG_RETURN_IF_NULL(invocation_context);
  GT_LOG_RETURN_IF_NULL(metrics);

  return invocation_context->StopMetricsCollection(*metrics);
}

LiteRtStatus GetNumMetrics(LiteRtDispatchMetrics metrics, int* num_metrics) {
  GT_LOG_RETURN_IF_NULL(metrics);
  GT_LOG_RETURN_IF_NULL(num_metrics);

  *num_metrics = metrics->GetNumMetrics();
  return kLiteRtStatusOk;
}

LiteRtStatus GetMetric(LiteRtDispatchMetrics metrics, int metric_index,
                       LiteRtMetric* metric) {
  GT_LOG_RETURN_IF_NULL(metrics);
  GT_LOG_RETURN_IF_NULL(metric);

  return metrics->GetMetric(metric_index, *metric);
}

LiteRtStatus DestroyMetrics(LiteRtDispatchMetrics metrics) {
  GT_LOG_RETURN_IF_NULL(metrics);

  delete metrics;
  return kLiteRtStatusOk;
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus GraphCreate(LiteRtDispatchDeviceContext device_context,
                         LiteRtDispatchGraph* graph) {
  GT_LOG_RETURN_IF_NULL(graph);

  return LiteRtDispatchGraphT::Create(device_context, *graph);
}

LiteRtStatus GraphDestroy(LiteRtDispatchGraph graph) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->Destroy();
}

LiteRtStatus AddNode(LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
                     LiteRtDispatchNodeType node_type) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->AddIndexedNode(node_id, node_type);
}

LiteRtStatus AddEdge(LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->AddEdge(edge_id);
}

LiteRtStatus ConnectNodeInput(LiteRtDispatchGraph graph,
                              LiteRtDispatchNodeId node_id, int input_index,
                              LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->ConnectIndexedNodeInput(node_id, input_index, edge_id);
}

LiteRtStatus ConnectNodeOutput(LiteRtDispatchGraph graph,
                               LiteRtDispatchNodeId node_id, int output_index,
                               LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->ConnectIndexedNodeOutput(node_id, output_index, edge_id);
}

LiteRtStatus ConnectGraphInput(LiteRtDispatchGraph graph, int input_index,
                               LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->ConnectGraphInput(edge_id);
}

LiteRtStatus ConnectGraphOutput(LiteRtDispatchGraph graph, int output_index,
                                LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->ConnectGraphOutput(edge_id);
}

LiteRtStatus LoadExecutable(LiteRtDispatchDeviceContext device_context,
                            LiteRtDispatchExecutableType type,
                            const LiteRtMemBuffer* bytecode_buffer,
                            LiteRtDispatchExecutableHandle* exec_handle) {
  GT_LOG_RETURN_IF_NULL(device_context);
  GT_LOG_RETURN_IF_NULL(bytecode_buffer);
  GT_LOG_RETURN_IF_NULL(exec_handle);

  return device_context->LoadExecutable(type, *bytecode_buffer, *exec_handle);
}

LiteRtStatus UnloadExecutable(LiteRtDispatchDeviceContext device_context,
                              LiteRtDispatchExecutableHandle exec_handle) {
  GT_LOG_RETURN_IF_NULL(device_context);

  return device_context->UnloadExecutable(exec_handle);
}

LiteRtStatus AssignNodeFunction(LiteRtDispatchGraph graph,
                                LiteRtDispatchNodeId node_id,
                                LiteRtDispatchExecutableHandle exec_handle,
                                const char* function_name) {
  GT_LOG_RETURN_IF_NULL(graph);

  return graph->AssignNodeFunction(node_id, exec_handle, function_name);
}

LiteRtStatus AnnotateGraph(LiteRtDispatchGraph graph, const char* key,
                           const char* value) {
  GT_LOG_RETURN_IF_NULL(graph);
  GT_LOG_RETURN_IF_NULL(key);
  GT_LOG_RETURN_IF_NULL(value);

  return graph->AnnotateGraph(key, value);
}

LiteRtStatus AnnotateNode(LiteRtDispatchGraph graph,
                          LiteRtDispatchNodeId node_id, const char* key,
                          const char* value) {
  GT_LOG_RETURN_IF_NULL(graph);
  GT_LOG_RETURN_IF_NULL(key);
  GT_LOG_RETURN_IF_NULL(value);

  return graph->AnnotateNode(node_id, key, value);
}

LiteRtStatus AnnotateEdge(LiteRtDispatchGraph graph,
                          LiteRtDispatchEdgeId edge_id, const char* key,
                          const char* value) {
  GT_LOG_RETURN_IF_NULL(graph);
  GT_LOG_RETURN_IF_NULL(key);
  GT_LOG_RETURN_IF_NULL(value);

  return graph->AnnotateEdge(edge_id, key, value);
}

LiteRtStatus InvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  GT_LOG_RETURN_IF_NULL(invocation_context);

  return LiteRtDispatchInvocationContextT::CreateFromGraph(
      device_context, /*exec_handle=*/std::nullopt, graph, *invocation_context);
}

LiteRtStatus InvocationContextGetGraph(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchGraph* graph) {
  GT_LOG_RETURN_IF_NULL(invocation_context);
  GT_LOG_RETURN_IF_NULL(graph);

  *graph = invocation_context->graph();
  return kLiteRtStatusOk;
}

}  // namespace litert::google_tensor

// /////////////////////////////////////////////////////////////////////////////

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::google_tensor::Initialize,
    .get_vendor_id = litert::google_tensor::GetVendorId,
    .get_build_id = litert::google_tensor::GetBuildId,
    .get_capabilities = litert::google_tensor::GetCapabilities,
    .device_context_create = litert::google_tensor::DeviceContextCreate,
    .device_context_destroy = litert::google_tensor::DeviceContextDestroy,
    .get_input_requirements = litert::google_tensor::GetInputRequirements,
    .get_output_requirements = litert::google_tensor::GetOutputRequirements,
    .register_tensor_buffer = litert::google_tensor::RegisterTensorBuffer,
    .unregister_tensor_buffer = litert::google_tensor::UnregisterTensorBuffer,
    .invocation_context_create = litert::google_tensor::InvocationContextCreate,
    .invocation_context_destroy =
        litert::google_tensor::InvocationContextDestroy,
    .invocation_context_set_scheduling_info =
        litert::google_tensor::InvocationContextSetSchedulingInfo,
    .attach_input = litert::google_tensor::AttachInput,
    .attach_output = litert::google_tensor::AttachOutput,
    .detach_input = litert::google_tensor::DetachInput,
    .detach_output = litert::google_tensor::DetachOutput,
    .invoke = litert::google_tensor::Invoke,
    .start_metrics_collection = litert::google_tensor::StartMetricsCollection,
    .stop_metrics_collection = litert::google_tensor::StopMetricsCollection,
    .get_num_metrics = litert::google_tensor::GetNumMetrics,
    .get_metric = litert::google_tensor::GetMetric,
    .destroy_metrics = litert::google_tensor::DestroyMetrics,
    .check_runtime_compatibility =
        litert::google_tensor::CheckRuntimeCompatibility,
    .invocation_context_set_options =
        litert::google_tensor::InvocationContextSetOptions,
};

LiteRtDispatchAsyncInterface TheAsyncInterface = {
    .attach_input_event = litert::google_tensor::AttachInputEvent,
    .invoke_async = litert::google_tensor::InvokeAsync,
};

LiteRtDispatchGraphInterface TheGraphInterface = {
    .graph_create = litert::google_tensor::GraphCreate,
    .graph_destroy = litert::google_tensor::GraphDestroy,
    .add_node = litert::google_tensor::AddNode,
    .add_edge = litert::google_tensor::AddEdge,
    .connect_node_input = litert::google_tensor::ConnectNodeInput,
    .connect_node_output = litert::google_tensor::ConnectNodeOutput,
    .connect_graph_input = litert::google_tensor::ConnectGraphInput,
    .connect_graph_output = litert::google_tensor::ConnectGraphOutput,
    .load_executable = litert::google_tensor::LoadExecutable,
    .unload_executable = litert::google_tensor::UnloadExecutable,
    .assign_node_function = litert::google_tensor::AssignNodeFunction,
    .annotate_graph = litert::google_tensor::AnnotateGraph,
    .annotate_node = litert::google_tensor::AnnotateNode,
    .annotate_edge = litert::google_tensor::AnnotateEdge,
    .invocation_context_create_from_graph =
        litert::google_tensor::InvocationContextCreateFromGraph,
    .invocation_context_get_graph =
        litert::google_tensor::InvocationContextGetGraph,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = &TheAsyncInterface,
    .graph_interface = &TheGraphInterface,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  GT_LOG_RETURN_IF_NULL(api);

  *api = TheApi;
  return kLiteRtStatusOk;
}
