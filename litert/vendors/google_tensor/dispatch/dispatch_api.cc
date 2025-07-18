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

#include "litert/cc/litert_macros.h"

#if LITERT_HAS_AHWB_SUPPORT
#include <android/hardware_buffer.h>
#endif

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/common/vendor_traits.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/dispatch/southbound.h"

namespace {
litert::google_tensor::Southbound* TheSouthbound = nullptr;
std::string TheBuildId;
}  // namespace

namespace litert {
namespace vendors {

// Implement trait methods for Google Tensor
LiteRtStatus VendorTraits<GoogleTensorTag>::Initialize(
    const std::string& lib_dir) {
  // Extract shared library directory
  auto shared_library_dir_opt =
      lib_dir.empty() ? std::nullopt : std::make_optional(lib_dir);

  // Create Southbound interface
  auto southbound = google_tensor::Southbound::Create(shared_library_dir_opt);
  if (!southbound) {
    LITERT_LOG(LITERT_INFO, "Initialization failure: %s",
               southbound.Error().Message().c_str());
    return southbound.Error().Status();
  }
  TheSouthbound = southbound->release();

  // Initialize the Southbound API
  auto thr_initialize = TheSouthbound->api().thr_initialize;
  if (!thr_initialize) {
    LITERT_LOG(LITERT_INFO, "thr_initialize not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (auto status = thr_initialize(); status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_INFO, "thr_initialize failed: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // Get version information
  auto thr_get_vendor_api_version =
      TheSouthbound->api().thr_get_vendor_api_version;
  const char* vendor_version =
      thr_get_vendor_api_version ? thr_get_vendor_api_version() : "N.A.";
  auto thr_get_vendor_id = TheSouthbound->api().thr_get_vendor_id;
  const char* sb_vendor_id = thr_get_vendor_id ? thr_get_vendor_id() : "N.A.";

  // Build version string
  char build_id[256];
  snprintf(
      build_id, sizeof(build_id),
      "GoogleTensor Dispatch API version %d.%d.%d, Darwinn API version %s, "
      "vendor id: %s",
      LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
      LITERT_API_VERSION_PATCH, vendor_version, sb_vendor_id);
  TheBuildId = build_id;

  return kLiteRtStatusOk;
}

std::string VendorTraits<GoogleTensorTag>::GetBuildId() { return TheBuildId; }

Expected<std::unique_ptr<VendorDeviceContext>>
VendorTraits<GoogleTensorTag>::CreateDeviceContext(
    const LiteRtDispatchDeviceContext* device_context_options) {
  // Google Tensor doesn't use device context options, just southbound
  auto result = LiteRtDispatchDeviceContextT::Create(*TheSouthbound);
  if (!result) {
    return Unexpected(result.Error().Status(), result.Error().Message());
  }
  // Cast the unique_ptr to the base type
  return std::unique_ptr<VendorDeviceContext>(result.Value().release());
}

LiteRtStatus VendorTraits<GoogleTensorTag>::RegisterTensorBuffer(
    VendorDeviceContext* context, LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  auto* gt_context = static_cast<LiteRtDispatchDeviceContextT*>(context);
  auto result = gt_context->RegisterTensorBuffer(tensor_buffer);
  if (!result) {
    return result.Error().Status();
  }
  *tensor_buffer_handle = *result;
  return kLiteRtStatusOk;
}

LiteRtStatus VendorTraits<GoogleTensorTag>::UnregisterTensorBuffer(
    VendorDeviceContext* context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* gt_context = static_cast<LiteRtDispatchDeviceContextT*>(context);
  auto result = gt_context->UnregisterTensorBuffer(tensor_buffer_handle);
  if (!result) {
    return result.Error().Status();
  }
  return kLiteRtStatusOk;
}

Expected<std::unique_ptr<VendorInvocationContext>>
VendorTraits<GoogleTensorTag>::CreateInvocationContext(
    VendorDeviceContext* device_context, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name) {
  auto* gt_device_context =
      static_cast<LiteRtDispatchDeviceContextT*>(device_context);

  // Create LiteRtMemBuffer from the raw pointer and size
  LiteRtMemBuffer mem_buffer = {.fd = -1,
                                .base_addr = exec_bytecode_ptr,
                                .offset = 0,
                                .size = exec_bytecode_size};

  // Determine executable type based on content or use default
  LiteRtDispatchExecutableType exec_type =
      kLiteRtDispatchExecutableTypeDspLibrary;

  // We need to use CreateFromBytecode which returns Expected<Ptr>
  auto result = LiteRtDispatchInvocationContextT::CreateFromBytecode(
      *TheSouthbound,
      reinterpret_cast<LiteRtDispatchDeviceContext>(gt_device_context),
      exec_type, &mem_buffer, function_name, /*num_inputs=*/0,
      /*num_outputs=*/0);

  if (!result) {
    return Unexpected(result.Error().Status(), result.Error().Message());
  }

  // Cast to base type
  return std::unique_ptr<VendorInvocationContext>(result.Value().release());
}

// Add async interface methods to traits
LiteRtStatus VendorTraits<GoogleTensorTag>::AttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  if (auto result =
          invocation_context->AttachInputEvent(graph_input_index, input_event);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to attach input event: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::InvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events) {
  if (auto result =
          invocation_context->InvokeAsync(num_output_events, output_events);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to invoke asynchronously: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

// Add graph interface methods to traits
LiteRtStatus VendorTraits<GoogleTensorTag>::GraphCreate(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph) {
  if (auto result = device_context->CreateGraph(); result) {
    *graph = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::GraphDestroy(
    LiteRtDispatchGraph graph) {
  auto device_context = graph->device_context();
  if (auto result = device_context->DestroyGraph(graph); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to destroy graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AddNode(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchNodeType node_type) {
  if (auto result = graph->AddNode(node_id, node_type); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to add node: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AddEdge(
    LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->AddEdge(edge_id); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to add edge: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::ConnectNodeInput(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int input_index,
    LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectNodeInput(node_id, input_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect node input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::ConnectNodeOutput(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int output_index,
    LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectNodeOutput(node_id, output_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect node output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::ConnectGraphInput(
    LiteRtDispatchGraph graph, int graph_input_index,
    LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectGraphInput(graph_input_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect graph input: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::ConnectGraphOutput(
    LiteRtDispatchGraph graph, int graph_output_index,
    LiteRtDispatchEdgeId edge_id) {
  if (auto result = graph->ConnectGraphOutput(graph_output_index, edge_id);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to connect graph output: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::LoadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode,
    LiteRtDispatchExecutableHandle* exec_handle) {
  if (auto result = device_context->LoadExecutable(type, bytecode); result) {
    *exec_handle = *result;
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to load executable: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::UnloadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle) {
  if (auto result = device_context->UnloadExecutable(exec_handle); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to unload executable: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AssignNodeFunction(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name) {
  // TODO - b/397771624: Southbound currently doesn't support function names, so
  // overriding function names to empty strings as a temporary fix. We need to
  // investigate with the CoreML team to find a more robust solution.
  function_name = "";
  if (auto result =
          graph->AssignNodeFunction(node_id, exec_handle, function_name);
      result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to assign node function: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AnnotateGraph(
    LiteRtDispatchGraph graph, const char* key, const char* value) {
  if (auto result = graph->AnnotateGraph(key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate graph: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AnnotateNode(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, const char* key,
    const char* value) {
  if (auto result = graph->AnnotateNode(node_id, key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate node: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::AnnotateEdge(
    LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id, const char* key,
    const char* value) {
  if (auto result = graph->AnnotateEdge(edge_id, key, value); result) {
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to annotate edge: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

LiteRtStatus VendorTraits<GoogleTensorTag>::InvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (auto result = LiteRtDispatchInvocationContextT::CreateFromGraph(
          *TheSouthbound, device_context, graph);
      result) {
    *invocation_context = result->release();
    return kLiteRtStatusOk;
  } else {
    LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: %s",
               result.Error().Message().c_str());
    return result.Error().Status();
  }
}

}  // namespace vendors
}  // namespace litert

// The async and graph interfaces are now provided by the base class
// using the trait methods defined below

// Use the macro to define the dispatch entry point
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(litert::vendors::GoogleTensorTag)