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

#include "litert/vendors/c/litert_dispatch.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"

LiteRtStatus LiteRtDispatchInitialize(
    LiteRtEnvironmentOptions environment_options, LiteRtOptions options) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetApiVersion(LiteRtApiVersion* api_version) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetVendorId(const char** vendor_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetBuildId(const char** build_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetCapabilities(int* capabilities) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchStartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchStopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetNumMetrics(LiteRtDispatchMetrics metrics,
                                         int* num_metrics) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGetMetric(LiteRtDispatchMetrics metrics,
                                     int metric_index, LiteRtMetric* metric) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchDestroyMetrics(LiteRtDispatchMetrics metrics) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGraphCreate(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchGraphDestroy(LiteRtDispatchGraph graph) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAddNode(LiteRtDispatchGraph graph,
                                   LiteRtDispatchNodeId node_id,
                                   LiteRtDispatchNodeType node_type) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAddEdge(LiteRtDispatchGraph graph,
                                   LiteRtDispatchEdgeId edge_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchConnectNodeInput(LiteRtDispatchGraph graph,
                                            LiteRtDispatchNodeId node_id,
                                            int input_index,
                                            LiteRtDispatchEdgeId edge_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchConnectNodeOutput(LiteRtDispatchGraph graph,
                                             LiteRtDispatchNodeId node_id,
                                             int output_index,
                                             LiteRtDispatchEdgeId edge_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchConnectGraphInput(LiteRtDispatchGraph graph,
                                             int input_index,
                                             LiteRtDispatchEdgeId edge_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchConnectGraphOutput(LiteRtDispatchGraph graph,
                                              int output_index,
                                              LiteRtDispatchEdgeId edge_id) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchLoadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer,
    LiteRtDispatchExecutableHandle* exec_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchUnloadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAssignNodeFunction(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAnnotateGraph(LiteRtDispatchGraph* graph,
                                         const char* key, const char* value) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAnnotateNode(LiteRtDispatchGraph* graph,
                                        LiteRtDispatchNodeId node_id,
                                        const char* key, const char* value) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchAnnotateEdge(LiteRtDispatchGraph* graph,
                                        LiteRtDispatchEdgeId edge_id,
                                        const char* key, const char* value) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  return kLiteRtStatusErrorUnsupported;
}

LiteRtStatus LiteRtDispatchInvocationContextGetGraph(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchGraph* graph) {
  return kLiteRtStatusErrorUnsupported;
}
