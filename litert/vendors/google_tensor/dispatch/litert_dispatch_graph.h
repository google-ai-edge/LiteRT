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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_

#include <cstddef>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// This class is thread-compatible.
class LiteRtDispatchGraphT {
 public:
  static LiteRtStatus Create(LiteRtDispatchDeviceContext device_context,
                             LiteRtDispatchGraph& graph);

  LiteRtStatus Destroy();

  // To remain compatible with older SouthBound versions, this class exposes
  // both positional and indexed node connection APIs.
  //
  // The APIs cannot be "mixed and matched", and therefore a call sequence such
  // as the following will produce an error:
  //
  // ```
  // LiteRtDispatchNodeId node_id = 0;
  // graph->AddPositionalNode(node_id, ...);
  // graph->ConnectIndexedNodeInput(node_id, ...);
  // ```

  // The positional node connection API.
  LiteRtStatus AddPositionalNode(LiteRtDispatchNodeId node_id,
                                 LiteRtDispatchNodeType node_type);

  LiteRtStatus ConnectPositionalNodeInput(LiteRtDispatchNodeId node_id,
                                          LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectPositionalNodeOutput(LiteRtDispatchNodeId node_id,
                                           LiteRtDispatchEdgeId edge_id);

  // The indexed node connection API.
  LiteRtStatus AddIndexedNode(LiteRtDispatchNodeId node_id,
                              LiteRtDispatchNodeType node_type);

  LiteRtStatus ConnectIndexedNodeInput(LiteRtDispatchNodeId node_id,
                                       int input_index,
                                       LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectIndexedNodeOutput(LiteRtDispatchNodeId node_id,
                                        int output_index,
                                        LiteRtDispatchEdgeId edge_id);

  LiteRtStatus AddEdge(LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectGraphInput(LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectGraphOutput(LiteRtDispatchEdgeId edge_id);

  LiteRtStatus AssignNodeFunction(LiteRtDispatchNodeId node_id,
                                  LiteRtDispatchExecutableHandle exec_handle,
                                  const char* absl_nullable function_name);

  LiteRtStatus AnnotateSystemAttribute(const char* absl_nonnull key,
                                       const char* absl_nonnull value);

  LiteRtStatus AnnotateGraph(const char* absl_nonnull key,
                             const char* absl_nonnull value);

  LiteRtStatus AnnotateNode(LiteRtDispatchNodeId node_id,
                            const char* absl_nonnull key,
                            const char* absl_nonnull value);

  LiteRtStatus AnnotateEdge(LiteRtDispatchEdgeId edge_id,
                            const char* absl_nonnull key,
                            const char* absl_nonnull value);

  // Registers an invocation context with the graph.
  //
  // This has the effect of guaranteeing that the graph remains alive until the
  // invocation context is unregistered, meaning that a subsequent call to
  // `UnregisterInvocationContext` is required to permit the graph to be
  // destroyed.
  //
  // NOTE: an invocation context may only be registered once.
  LiteRtStatus RegisterInvocationContext(
      LiteRtDispatchInvocationContext icontext);

  LiteRtStatus UnregisterInvocationContext(
      LiteRtDispatchInvocationContext icontext);

  LiteRtStatus GetInputEdgeId(int input_index,
                              LiteRtDispatchEdgeId& edge_id) const {
    auto iter = input_edge_ids_.find(input_index);
    if (iter == input_edge_ids_.end()) {
      LITERT_LOG(LITERT_ERROR, "Edge ID not found for input index %d",
                 input_index);
      return kLiteRtStatusErrorNotFound;
    }

    edge_id = iter->second;
    return kLiteRtStatusOk;
  }

  LiteRtStatus GetOutputEdgeId(int output_index,
                               LiteRtDispatchEdgeId& edge_id) const {
    auto iter = output_edge_ids_.find(output_index);
    if (iter == output_edge_ids_.end()) {
      LITERT_LOG(LITERT_ERROR, "Edge ID not found for output index %d",
                 output_index);
      return kLiteRtStatusErrorNotFound;
    }

    edge_id = iter->second;
    return kLiteRtStatusOk;
  }

  ThrGraph* absl_nonnull thr_graph() { return thr_graph_; }

  size_t NumOutputEdges() const { return output_edge_ids_.size(); }

 private:
  LiteRtDispatchGraphT(LiteRtDispatchDeviceContext device_context,
                       ThrGraph* absl_nonnull thr_graph)
      : device_context_(device_context), thr_graph_(thr_graph) {}

  // Consumers of this class must use `Destroy` to delete the instance.
  ~LiteRtDispatchGraphT() = default;

  LiteRtDispatchDeviceContext device_context_;
  ThrGraph* absl_nonnull thr_graph_;
  // Set to `true` after the graph is successfully registered with its device
  // context. This prevents 'Destroy' from attempting to unregister the graph
  // from its device context when the graph has not previously been registered.
  bool registered_with_device_context_ = false;
  // Associates an input index with its edge ID.
  absl::flat_hash_map<int, LiteRtDispatchEdgeId> input_edge_ids_;
  // Associates an output index with its edge ID.
  absl::flat_hash_map<int, LiteRtDispatchEdgeId> output_edge_ids_;
  // A graph cannot be destroyed with any registered invocation contexts.
  absl::flat_hash_set<LiteRtDispatchInvocationContext> registered_icontexts_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_
