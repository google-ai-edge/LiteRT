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

#ifndef ODML_LITERT_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_
#define ODML_LITERT_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_

#include <stddef.h>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/vendors/c/litert_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current ABI version for LiteRT Dispatch interface tables (Layer 3).
#define LITERT_DISPATCH_ABI_VERSION_MAJOR 1
#define LITERT_DISPATCH_ABI_VERSION_MINOR 0
#define LITERT_DISPATCH_ABI_VERSION_PATCH 0

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchInitializeT)(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment environment,
    LiteRtOptions options);

typedef LiteRtStatus (*LiteRtDispatchGetVendorIdT)(const char** vendor_id);

typedef LiteRtStatus (*LiteRtDispatchGetBuildIdT)(const char** build_id);

typedef LiteRtStatus (*LiteRtDispatchGetCapabilitiesT)(int* capabilities);

typedef LiteRtStatus (*LiteRtDispatchDeviceContextCreateT)(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
    LiteRtDispatchDeviceContext* device_context);

typedef LiteRtStatus (*LiteRtDispatchDeviceContextDestroyT)(
    LiteRtDispatchDeviceContext device_context);

typedef LiteRtStatus (*LiteRtDispatchGetInputRequirementsT)(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

typedef LiteRtStatus (*LiteRtDispatchGetOutputRequirementsT)(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

typedef LiteRtStatus (*LiteRtDispatchRegisterTensorBufferT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchUnregisterTensorBufferT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle handle);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextCreateT)(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextDestroyT)(
    LiteRtDispatchInvocationContext invocation_context);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextSetOptionsT)(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextSetSchedulingInfoT)(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info);

typedef LiteRtStatus (*LiteRtDispatchAttachInputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchAttachOutputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchDetachInputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchDetachOutputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchAttachEdgeBufferT)(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchDetachEdgeBufferT)(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchInvokeT)(
    LiteRtDispatchInvocationContext invocation_context);

typedef LiteRtStatus (*LiteRtDispatchStartMetricsCollectionT)(
    LiteRtDispatchInvocationContext invocation_context, int detail_level);

typedef LiteRtStatus (*LiteRtDispatchStopMetricsCollectionT)(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics);

typedef LiteRtStatus (*LiteRtDispatchGetNumMetricsT)(
    LiteRtDispatchMetrics metrics, int* num_metrics);

typedef LiteRtStatus (*LiteRtDispatchGetMetricT)(LiteRtDispatchMetrics metrics,
                                                 int metric_index,
                                                 LiteRtMetric* metric);

typedef LiteRtStatus (*LiteRtDispatchDestroyMetricsT)(
    LiteRtDispatchMetrics metrics);

typedef LiteRtStatus (*LiteRtDispatchCheckRuntimeCompatibilityT)(
    LiteRtApiVersion api_version, LiteRtEnvironmentOptions env,
    LiteRtOptions options);

typedef LiteRtStatus (*LiteRtDispatchGetHooksT)(
    LiteRtDispatchDeviceContext device_context, LiteRtHook* hook,
    void** user_data);

typedef struct LiteRtDispatchInterface_V1 {
  LiteRtAbiHeader abi_header;

  LiteRtDispatchInitializeT initialize;
  LiteRtDispatchGetVendorIdT get_vendor_id;
  LiteRtDispatchGetBuildIdT get_build_id;
  LiteRtDispatchGetCapabilitiesT get_capabilities;
  LiteRtDispatchDeviceContextCreateT device_context_create;
  LiteRtDispatchDeviceContextDestroyT device_context_destroy;
  LiteRtDispatchGetInputRequirementsT get_input_requirements;
  LiteRtDispatchGetOutputRequirementsT get_output_requirements;
  LiteRtDispatchRegisterTensorBufferT register_tensor_buffer;
  LiteRtDispatchUnregisterTensorBufferT unregister_tensor_buffer;
  LiteRtDispatchInvocationContextCreateT invocation_context_create;
  LiteRtDispatchInvocationContextDestroyT invocation_context_destroy;
  LiteRtDispatchInvocationContextSetSchedulingInfoT
      invocation_context_set_scheduling_info;
  LiteRtDispatchAttachInputT attach_input;
  LiteRtDispatchAttachOutputT attach_output;
  LiteRtDispatchDetachInputT detach_input;
  LiteRtDispatchDetachOutputT detach_output;
  LiteRtDispatchInvokeT invoke;
  LiteRtDispatchStartMetricsCollectionT start_metrics_collection;
  LiteRtDispatchStopMetricsCollectionT stop_metrics_collection;
  LiteRtDispatchGetNumMetricsT get_num_metrics;
  LiteRtDispatchGetMetricT get_metric;
  LiteRtDispatchDestroyMetricsT destroy_metrics;
  LiteRtDispatchCheckRuntimeCompatibilityT check_runtime_compatibility;
  LiteRtDispatchInvocationContextSetOptionsT invocation_context_set_options;
  LiteRtDispatchGetHooksT get_hooks;

  // Optional extensions (capability-gated).
  LiteRtDispatchAttachEdgeBufferT attach_edge_buffer;
  LiteRtDispatchDetachEdgeBufferT detach_edge_buffer;
} LiteRtDispatchInterface_V1;

LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtDispatchInterface_V1, abi_header) == 0,
    "LiteRtDispatchInterface_V1 abi_header must be at offset 0");

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchAttachInputEventT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event);

typedef LiteRtStatus (*LiteRtDispatchInvokeAsyncT)(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events);

typedef struct LiteRtDispatchAsyncInterface_V1 {
  LiteRtAbiHeader abi_header;

  LiteRtDispatchAttachInputEventT attach_input_event;
  LiteRtDispatchInvokeAsyncT invoke_async;
} LiteRtDispatchAsyncInterface_V1;

LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtDispatchAsyncInterface_V1, abi_header) == 0,
    "LiteRtDispatchAsyncInterface_V1 abi_header must be at offset 0");

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchGraphCreateT)(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph);

typedef LiteRtStatus (*LiteRtDispatchGraphDestroyT)(LiteRtDispatchGraph graph);

typedef LiteRtStatus (*LiteRtDispatchAddNodeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchNodeType node_type);

typedef LiteRtStatus (*LiteRtDispatchAddEdgeT)(LiteRtDispatchGraph graph,
                                               LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectNodeInputT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int input_index,
    LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectNodeOutputT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int output_index,
    LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectGraphInputT)(
    LiteRtDispatchGraph graph, int input_index, LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectGraphOutputT)(
    LiteRtDispatchGraph graph, int output_index, LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchLoadExecutableT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer,
    LiteRtDispatchExecutableHandle* exec_handle);

typedef LiteRtStatus (*LiteRtDispatchUnloadExecutableT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle);

typedef LiteRtStatus (*LiteRtDispatchGetScratchpadRequirementsT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name,
    LiteRtTensorBufferRequirements* scratchpad_requirements);

typedef LiteRtStatus (*LiteRtDispatchAttachScratchpadBufferT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name,
    LiteRtTensorBufferHandle scratchpad_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchAssignNodeFunctionT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextCreateFromGraphT)(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextGetGraphT)(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchGraph* graph);

typedef LiteRtStatus (*LiteRtDispatchAnnotateGraphT)(LiteRtDispatchGraph graph,
                                                     const char* key,
                                                     const char* value);

typedef LiteRtStatus (*LiteRtDispatchAnnotateNodeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, const char* key,
    const char* value);

typedef LiteRtStatus (*LiteRtDispatchAnnotateEdgeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id, const char* key,
    const char* value);

typedef struct LiteRtDispatchGraphInterface_V1 {
  LiteRtAbiHeader abi_header;

  LiteRtDispatchGraphCreateT graph_create;
  LiteRtDispatchGraphDestroyT graph_destroy;
  LiteRtDispatchAddNodeT add_node;
  LiteRtDispatchAddEdgeT add_edge;
  LiteRtDispatchConnectNodeInputT connect_node_input;
  LiteRtDispatchConnectNodeOutputT connect_node_output;
  LiteRtDispatchConnectGraphInputT connect_graph_input;
  LiteRtDispatchConnectGraphOutputT connect_graph_output;
  LiteRtDispatchLoadExecutableT load_executable;
  LiteRtDispatchUnloadExecutableT unload_executable;
  LiteRtDispatchAssignNodeFunctionT assign_node_function;
  LiteRtDispatchAnnotateGraphT annotate_graph;
  LiteRtDispatchAnnotateNodeT annotate_node;
  LiteRtDispatchAnnotateEdgeT annotate_edge;
  LiteRtDispatchInvocationContextCreateFromGraphT
      invocation_context_create_from_graph;
  LiteRtDispatchInvocationContextGetGraphT invocation_context_get_graph;
  // Optional extensions (capability-gated).
  LiteRtDispatchGetScratchpadRequirementsT get_scratchpad_requirements;
  LiteRtDispatchAttachScratchpadBufferT attach_scratchpad_buffer;
} LiteRtDispatchGraphInterface_V1;

LITERT_ABI_STATIC_ASSERT(
    offsetof(LiteRtDispatchGraphInterface_V1, abi_header) == 0,
    "LiteRtDispatchGraphInterface_V1 abi_header must be at offset 0");

// /////////////////////////////////////////////////////////////////////////////

/// A internal struct that holds pointers to all the Dispatch API functions.
/// Dispatch Delegate will look up this struct to extract the Dispatch API
/// functions.
///
/// @note This concrete type is shared between the runtime and the Dispatch
///     plugin, so it must be ABI stable.
typedef enum {
  kLiteRtInterfaceBasic = 0,
  kLiteRtInterfaceAsync = 1,
  kLiteRtInterfaceGraph = 2,
  kLiteRtInterfaceCustomTensorBufferHandlers = 3,
} LiteRtDispatchInterfaceId;

typedef LiteRtStatus (*LiteRtDispatchQueryInterfaceT)(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface);

LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface);

// Pointer to a statically linked dispatch API implementation.
// Vendors that are statically linked can set this pointer to their
// implementation of LiteRtDispatchQueryInterface during static initialization.
// The storage for this pointer is defined in the internal runtime at
// litert/runtime/dispatch/litert_dispatch.cc.
// The runtime will invoke this function to get the API instead of loading a
// dynamic library if it is not null.
extern LiteRtStatus (*LiteRtStaticLinkedDispatchQueryInterface)(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface);

#ifdef __cplusplus
}
#endif  // __cplusplus

#ifdef __cplusplus
#include "absl/strings/string_view.h"  // from @com_google_absl

static constexpr absl::string_view kLiteRtDispatchQueryInterface =
    "LiteRtDispatchQueryInterface";

#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_
