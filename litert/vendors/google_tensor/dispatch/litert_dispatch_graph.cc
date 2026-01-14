// Copyright 2025 Google LLC.
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

#include "litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"

#include <cinttypes>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_macros.h"
#include "litert/vendors/google_tensor/dispatch/dispatch_api_utils.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

namespace gt = litert::google_tensor;

LiteRtStatus LiteRtDispatchGraphT::Create(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph& graph) {
  GT_LOG_RETURN_IF_NULL(device_context);

  ThrGraph* thr_graph = thrGraphCreate(device_context->thr_context());
  if (thr_graph == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to create SB graph");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  graph = new LiteRtDispatchGraphT(device_context, thr_graph);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AddNode(LiteRtDispatchNodeId node_id,
                                           LiteRtDispatchNodeType node_type) {
  ThrNodeType thr_node_type;
  switch (node_type) {
    case kLiteRtDispatchNodeTypeDsp:
      thr_node_type = kThrNodeTypeDsp;
      break;
    case kLiteRtDispatchNodeTypeNpu:
      thr_node_type = kThrNodeTypeNpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Invalid node type %d", node_type);
      return kLiteRtStatusErrorInvalidArgument;
  }

  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAddSqNodeWithInterfaceBindingMode(
          thr_graph_, gt::ToThrNodeId(node_id), thr_node_type,
          kThrNodeInterfaceBindingModeIndexed),
      "Failed to add node %" PRIu64 " with type %d to SB graph", node_id,
      node_type);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AddEdge(LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAddEdge(thr_graph_, gt::ToThrEdgeId(edge_id), kThrEdgeTypeTensor),
      "Failed to add edge %" PRIu64 " to SB graph", edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::ConnectNodeInput(
    LiteRtDispatchNodeId node_id, int input_index,
    LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_SB_ERROR(thrGraphConnectNodeInputWithPortIndex(
                                thr_graph_, gt::ToThrNodeId(node_id),
                                gt::ToThrEdgeId(edge_id), input_index),
                            "Failed to connect node %" PRIu64
                            " to input edge %" PRIu64 " at index %d",
                            node_id, edge_id, input_index);

  AddInputEdge(input_index, edge_id);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::ConnectNodeOutput(
    LiteRtDispatchNodeId node_id, int output_index,
    LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_SB_ERROR(thrGraphConnectNodeOutputWithPortIndex(
                                thr_graph_, gt::ToThrNodeId(node_id),
                                gt::ToThrEdgeId(edge_id), output_index),
                            "Failed to connect node %" PRIu64
                            " to output edge %" PRIu64 " at index %d",
                            node_id, edge_id, output_index);

  AddOutputEdge(output_index, edge_id);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::ConnectGraphInput(
    LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphSetInputEdge(thr_graph_, gt::ToThrEdgeId(edge_id)),
      "Failed to set input edge %" PRIu64 " on SB graph", edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::ConnectGraphOutput(
    LiteRtDispatchEdgeId edge_id) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphSetOutputEdge(thr_graph_, gt::ToThrEdgeId(edge_id)),
      "Failed to set output edge %" PRIu64 " on SB graph", edge_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AssignNodeFunction(
    LiteRtDispatchNodeId node_id, LiteRtDispatchExecutableHandle exec_handle,
    const char* absl_nullable function_name) {
  // An empty function name represents no function name being provided, and
  // therefore we must pass `nullptr` to the call below, otherwise SB expects a
  // model with a signature - b/378913220.
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAssignSq(
          thr_graph_, gt::ToThrNodeId(node_id), exec_handle,
          !function_name || *function_name == '\0' ? nullptr : function_name),
      "Failed to assign function '%s' from executable %" PRIu64
      " to node %" PRIu64,
      function_name, exec_handle, node_id);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AnnotateGraph(
    const char* absl_nonnull key, const char* absl_nonnull value) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAnnotateGraph(thr_graph_, key, value),
      "Failed to annotate SB graph with key '%s' and value '%s'", key, value);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AnnotateNode(
    LiteRtDispatchNodeId node_id, const char* absl_nonnull key,
    const char* absl_nonnull value) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAnnotateNode(thr_graph_, gt::ToThrNodeId(node_id), key, value),
      "Failed to annotate node %" PRIu64 " with key '%s' and value '%s'",
      node_id, key, value);

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGraphT::AnnotateEdge(
    LiteRtDispatchEdgeId edge_id, const char* absl_nonnull key,
    const char* absl_nonnull value) {
  GT_LOG_RETURN_IF_SB_ERROR(
      thrGraphAnnotateEdge(thr_graph_, gt::ToThrEdgeId(edge_id), key, value),
      "Failed to annotate edge %" PRIu64 " with key '%s' and value '%s'",
      edge_id, key, value);

  return kLiteRtStatusOk;
}
