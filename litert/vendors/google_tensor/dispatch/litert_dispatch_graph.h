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
#include <map>
#include <memory>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

class LiteRtDispatchGraphT {
 public:
  static LiteRtStatus Create(LiteRtDispatchDeviceContext device_context,
                             LiteRtDispatchGraph& graph);

  static LiteRtStatus Create(
      LiteRtDispatchDeviceContext device_context,
      absl_nullable std::unique_ptr<LiteRtDispatchGraphT>& graph) {
    LiteRtDispatchGraph raw_graph;
    LITERT_RETURN_IF_ERROR(Create(device_context, raw_graph));

    graph = std::unique_ptr<LiteRtDispatchGraphT>(raw_graph);
    return kLiteRtStatusOk;
  }

  ThrGraph* absl_nonnull thr_graph() { return thr_graph_; }

  LiteRtDispatchDeviceContext device_context() { return device_context_; }

  LiteRtStatus AddNode(LiteRtDispatchNodeId node_id,
                       LiteRtDispatchNodeType node_type);

  LiteRtStatus AddEdge(LiteRtDispatchEdgeId edge_id);

  litert::Expected<LiteRtDispatchEdgeId> InputEdge(int input_index) const {
    return IoEdge(input_index, input_edges_);
  }

  litert::Expected<LiteRtDispatchEdgeId> OutputEdge(int output_index) const {
    return IoEdge(output_index, output_edges_);
  }

  size_t NumOutputs() const { return output_edges_.size(); }

  LiteRtStatus ConnectNodeInput(LiteRtDispatchNodeId node_id, int input_index,
                                LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectNodeOutput(LiteRtDispatchNodeId node_id, int output_index,
                                 LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectGraphInput(LiteRtDispatchEdgeId edge_id);

  LiteRtStatus ConnectGraphOutput(LiteRtDispatchEdgeId edge_id);

  LiteRtStatus AssignNodeFunction(LiteRtDispatchNodeId node_id,
                                  LiteRtDispatchExecutableHandle exec_handle,
                                  const char* absl_nullable function_name);

  LiteRtStatus AnnotateGraph(const char* absl_nonnull key,
                             const char* absl_nonnull value);

  LiteRtStatus AnnotateNode(LiteRtDispatchNodeId node_id,
                            const char* absl_nonnull key,
                            const char* absl_nonnull value);

  LiteRtStatus AnnotateEdge(LiteRtDispatchEdgeId edge_id,
                            const char* absl_nonnull key,
                            const char* absl_nonnull value);

 private:
  using IoIndexToEdgeIdMap = std::map<int, LiteRtDispatchEdgeId>;

  LiteRtDispatchGraphT(LiteRtDispatchDeviceContext device_context,
                       ThrGraph* absl_nonnull thr_graph)
      : device_context_(device_context), thr_graph_(thr_graph) {}

  litert::Expected<LiteRtDispatchEdgeId> IoEdge(
      int io_index, const IoIndexToEdgeIdMap& map) const {
    auto iter = map.find(io_index);
    if (iter == map.end()) {
      return litert::Unexpected(kLiteRtStatusErrorNotFound,
                                "Unexpected graph input/output index");
    }
    return iter->second;
  }

  void AddInputEdge(int input_index, LiteRtDispatchEdgeId edge_id) {
    input_edges_[input_index] = edge_id;
  }

  void AddOutputEdge(int output_index, LiteRtDispatchEdgeId edge_id) {
    output_edges_[output_index] = edge_id;
  }

  LiteRtDispatchDeviceContext device_context_;
  ThrGraph* absl_nonnull thr_graph_;
  IoIndexToEdgeIdMap input_edges_;
  IoIndexToEdgeIdMap output_edges_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_
