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

#include "litert/c/litert_any.h"

#if !defined(LITERT_WINDOWS_OS)
#include <dlfcn.h>
#endif  // !defined(LITERT_WINDOWS_OS)

#include <string>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_metrics.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/util/perfetto_profiling.h"
#include "litert/core/version.h"
#include "litert/vendors/c/litert_dispatch_api.h"

#define INVOKE_FUNC(function, ...)                                \
  if (!TheApi.interface) {                                        \
    LITERT_LOG(LITERT_ERROR, "Dispatch API interface not found"); \
    return kLiteRtStatusErrorRuntimeFailure;                      \
  }                                                               \
  if (!TheApi.interface->function) {                              \
    LITERT_LOG(LITERT_ERROR, #function " not found");             \
    return kLiteRtStatusErrorRuntimeFailure;                      \
  }                                                               \
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API " #function);         \
  return TheApi.interface->function(__VA_ARGS__);

#define INVOKE_ASYNC_FUNC(function, ...)                                \
  if (!TheApi.async_interface) {                                        \
    LITERT_LOG(LITERT_ERROR, "Dispatch API async interface not found"); \
    return kLiteRtStatusErrorRuntimeFailure;                            \
  }                                                                     \
  if (!TheApi.async_interface->function) {                              \
    LITERT_LOG(LITERT_ERROR, #function " not found");                   \
    return kLiteRtStatusErrorRuntimeFailure;                            \
  }                                                                     \
  return TheApi.async_interface->function(__VA_ARGS__);

#define INVOKE_GRAPH_FUNC(function, ...)                                \
  if (!TheApi.graph_interface) {                                        \
    LITERT_LOG(LITERT_ERROR, "Dispatch API graoh interface not found"); \
    return kLiteRtStatusErrorRuntimeFailure;                            \
  }                                                                     \
  if (!TheApi.graph_interface->function) {                              \
    LITERT_LOG(LITERT_ERROR, #function " not found");                   \
    return kLiteRtStatusErrorRuntimeFailure;                            \
  }                                                                     \
  return TheApi.graph_interface->function(__VA_ARGS__);

namespace {

litert::SharedLibrary* DispatchSharedLibrary = nullptr;
bool IsTheApiInitialized = false;
LiteRtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/0, /*.minor=*/0, /*.patch=*/0},
    /*.interface=*/nullptr,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

LiteRtStatus Initialize(LiteRtEnvironment env, LiteRtOptions options) {
  INVOKE_FUNC(initialize, env, options);
}

litert::Expected<std::string> GetSharedLibraryPath(
    LiteRtEnvironmentOptions env_options) {
  std::vector<std::string> dispatch_lib_paths;
  LiteRtAny dispatch_lib_dir;
  auto status = LiteRtGetEnvironmentOptionsValue(
      env_options, kLiteRtEnvOptionTagDispatchLibraryDir, &dispatch_lib_dir);
  if (status == kLiteRtStatusOk) {
    litert::internal::FindLiteRtDispatchSharedLibs(dispatch_lib_dir.str_value,
                                                   dispatch_lib_paths);
  }
  if (dispatch_lib_paths.empty()) {
    LITERT_LOG(LITERT_ERROR, "No dispatch library found in %s",
               dispatch_lib_dir.str_value);
    return litert::Error(kLiteRtStatusErrorRuntimeFailure);
  }
  if (dispatch_lib_paths.size() > 1) {
    LITERT_LOG(LITERT_WARNING, "Multiple dispatch libraries found");
  }
  return dispatch_lib_paths.front();
}

}  // namespace

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchInitialize(LiteRtEnvironment env,
                                      LiteRtOptions options) {
  if (IsTheApiInitialized) {
    return kLiteRtStatusOk;
  }
  LiteRtEnvironmentOptions env_options;
  LITERT_RETURN_IF_ERROR(LiteRtGetEnvironmentOptions(env, &env_options));

  // TODO(piyu): support Android systems where libraries are not unpacked in the
  // system directory.
  LITERT_ASSIGN_OR_RETURN(auto shared_lib_path,
                          GetSharedLibraryPath(env_options));

  LITERT_LOG(LITERT_INFO, "Loading shared library: %s",
             shared_lib_path.c_str());

  if (!DispatchSharedLibrary) {
    DispatchSharedLibrary = new litert::SharedLibrary();
  }

  LITERT_ASSIGN_OR_RETURN(
      *DispatchSharedLibrary,
      litert::SharedLibrary::Load(shared_lib_path,
                                  litert::RtldFlags::Now().Local()));

  using LiteRtDispatchGetApi_t = LiteRtStatus (*)(LiteRtDispatchApi*);
  LITERT_ASSIGN_OR_RETURN(
      auto LiteRtDispatchGetApi,
      DispatchSharedLibrary->LookupSymbol<LiteRtDispatchGetApi_t>(
          "LiteRtDispatchGetApi"));

  if (auto status = LiteRtDispatchGetApi(&TheApi); status != kLiteRtStatusOk) {
    return status;
  }

  if (!litert::internal::IsSameVersionAsRuntime(TheApi.version)) {
    LITERT_LOG(LITERT_ERROR, "Unsupported dispatch runtime version");
    return kLiteRtStatusErrorWrongVersion;
  }

  auto status = Initialize(env, options);
  if (status == kLiteRtStatusOk) {
    IsTheApiInitialized = true;
  }
  return status;
}

LiteRtStatus LiteRtDispatchGetApiVersion(LiteRtApiVersion* api_version) {
  if (!api_version) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *api_version = TheApi.version;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetVendorId(const char** vendor_id) {
  if (!vendor_id) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_vendor_id, vendor_id);
}

LiteRtStatus LiteRtDispatchGetBuildId(const char** build_id) {
  if (!build_id) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_build_id, build_id);
}

LiteRtStatus LiteRtDispatchGetCapabilities(int* capabilities) {
  if (!capabilities) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_capabilities, capabilities);
}

LiteRtStatus LiteRtDispatchDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(device_context_create, device_context);
}

LiteRtStatus LiteRtDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(device_context_destroy, device_context);
}

LiteRtStatus LiteRtDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_input_requirements, invocation_context, input_index,
              tensor_type, tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_output_requirements, invocation_context, output_index,
              tensor_type, tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (!device_context || !tensor_buffer || !tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(register_tensor_buffer, device_context, tensor_buffer,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(unregister_tensor_buffer, device_context, tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (!device_context || !exec_bytecode_buffer || !invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_create, device_context, exec_type,
              exec_bytecode_buffer, function_name, num_inputs, num_outputs,
              invocation_context);
}

LiteRtStatus LiteRtDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_destroy, invocation_context);
}

LiteRtStatus LiteRtDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(attach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!TheApi.interface) {
    LITERT_LOG(LITERT_ERROR, "Dispatch API interface not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (!TheApi.interface->attach_output) {
    LITERT_LOG(LITERT_ERROR, "attach_output_tensor_buffer not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  INVOKE_FUNC(attach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invoke, invocation_context);
}

LiteRtStatus LiteRtDispatchCheckRuntimeCompatibility(
    LiteRtApiVersion api_version, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  INVOKE_FUNC(check_runtime_compatibility, api_version, env, options);
}

LiteRtStatus LiteRtDispatchStartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  } else if (detail_level < 0) {
    LITERT_LOG(LITERT_ERROR, "Invalid detail level");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(start_metrics_collection, invocation_context, detail_level);
}

LiteRtStatus LiteRtDispatchStopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  if (!invocation_context || !metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(stop_metrics_collection, invocation_context, metrics);
}

LiteRtStatus LiteRtDispatchGetNumMetrics(LiteRtDispatchMetrics metrics,
                                         int* num_metrics) {
  if (!metrics || !num_metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_num_metrics, metrics, num_metrics);
}

LiteRtStatus LiteRtDispatchGetMetric(LiteRtDispatchMetrics metrics,
                                     int metric_index, LiteRtMetric* metric) {
  if (!metrics || !metric) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_metric, metrics, metric_index, metric);
}

LiteRtStatus LiteRtDispatchDestroyMetrics(LiteRtDispatchMetrics metrics) {
  if (!metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(destroy_metrics, metrics);
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchAttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  if (!invocation_context || !input_event) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(attach_input_event, invocation_context, graph_input_index,
                    input_event);
}

LiteRtStatus LiteRtDispatchInvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events) {
  if (!invocation_context || !output_events) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(invoke_async, invocation_context, num_output_events,
                    output_events);
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchGraphCreate(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph) {
  if (!device_context || !graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_create, device_context, graph);
}

LiteRtStatus LiteRtDispatchGraphDestroy(LiteRtDispatchGraph graph) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_destroy, graph);
}

LiteRtStatus LiteRtDispatchAddNode(LiteRtDispatchGraph graph,
                                   LiteRtDispatchNodeId node_id,
                                   LiteRtDispatchNodeType node_type) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_node, graph, node_id, node_type);
}

LiteRtStatus LiteRtDispatchAddEdge(LiteRtDispatchGraph graph,
                                   LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_edge, graph, edge_id);
}

LiteRtStatus LiteRtDispatchConnectNodeInput(LiteRtDispatchGraph graph,
                                            LiteRtDispatchNodeId node_id,
                                            int input_index,
                                            LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_input, graph, node_id, input_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectNodeOutput(LiteRtDispatchGraph graph,
                                             LiteRtDispatchNodeId node_id,
                                             int output_index,
                                             LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_output, graph, node_id, output_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectGraphInput(LiteRtDispatchGraph graph,
                                             int input_index,
                                             LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_input, graph, input_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectGraphOutput(LiteRtDispatchGraph graph,
                                              int output_index,
                                              LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_output, graph, output_index, edge_id);
}

LiteRtStatus LiteRtDispatchLoadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer,
    LiteRtDispatchExecutableHandle* exec_handle) {
  if (!device_context || !bytecode_buffer || !exec_handle) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!TheApi.graph_interface) {
    LITERT_LOG(LITERT_ERROR, "Dispatch API graph interface not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (!TheApi.graph_interface->load_executable) {
    LITERT_LOG(LITERT_ERROR, "load_executable not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  INVOKE_GRAPH_FUNC(load_executable, device_context, type, bytecode_buffer,
                    exec_handle);
}

LiteRtStatus LiteRtDispatchUnloadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(unload_executable, device_context, exec_handle);
}

LiteRtStatus LiteRtDispatchAssignNodeFunction(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(assign_node_function, graph, node_id, exec_handle,
                    function_name);
}

LiteRtStatus LiteRtDispatchAnnotateGraph(LiteRtDispatchGraph graph,
                                         const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_graph, graph, key, value);
}

LiteRtStatus LiteRtDispatchAnnotateNode(LiteRtDispatchGraph graph,
                                        LiteRtDispatchNodeId node_id,
                                        const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_node, graph, node_id, key, value);
}

LiteRtStatus LiteRtDispatchAnnotateEdge(LiteRtDispatchGraph graph,
                                        LiteRtDispatchEdgeId edge_id,
                                        const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_edge, graph, edge_id, key, value);
}

LiteRtStatus LiteRtDispatchInvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (!device_context || !graph || !invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(invocation_context_create_from_graph, device_context, graph,
                    invocation_context);
}

LiteRtStatus LiteRtDispatchInvocationContextGetGraph(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchGraph* graph) {
  if (!invocation_context || !graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!TheApi.graph_interface ||
      !TheApi.graph_interface->invocation_context_get_graph) {
    LITERT_LOG(LITERT_ERROR, "invocation_context_get_graph not found");
    return kLiteRtStatusErrorUnsupported;
  }
  INVOKE_GRAPH_FUNC(invocation_context_get_graph, invocation_context, graph);
}
