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

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/litert_model.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {

// Algorithm class for managing "scope" when mapping litert Subgraphs
// to QNN Graphs.
class GraphMapper {
 public:
  GraphMapper(LiteRtSubgraph subgraph, QnnManager& qnn,
              Qnn_ContextHandle_t context_handle)
      : subgraph_(Subgraph(subgraph)),
        qnn_(qnn),
        context_handle_(context_handle) {}

  // QNN Sdk Accessors
  QnnManager& Qnn();
  Qnn_GraphHandle_t& QnnGraph();

  // CC Convenience Accessors
  const Subgraph& Graph() const { return subgraph_; }

  // Can implementation handle given LiteRtSubgraph topology (see comment at
  // bottom of file).
  LiteRtStatus IsLiteRtSubgraphSupported();

  // Initialize QNN Graph with given name. Call this after parsing
  // LiteRtSubgraph.
  LiteRtStatus InitQnnGraph(absl::string_view qnn_graph_name,
                            const ::qnn::Options& options);

  // Finalize QNN Graph. Call this after all ops have been mapped.
  LiteRtStatus Finalize();

  inline void RegisterOutput(LiteRtTensor litert_tensor) {
    graph_outpus_.insert(litert_tensor);
  }

  // Pick graph config based on subgraph.
  absl::Span<const QnnGraph_Config_t*> PickGraphConfigHeuristic(
      const ::qnn::Options& options);

  inline bool IsTensorOutput(LiteRtTensor litert_tensor) {
    return graph_outpus_.contains(litert_tensor);
  }

 private:
  const Subgraph subgraph_;

  // Set of all outputs of the graph.
  absl::flat_hash_set<LiteRtTensor> graph_outpus_;

  //
  // QNN Sdk State
  //
  QnnManager& qnn_;
  Qnn_ContextHandle_t context_handle_;
  Qnn_GraphHandle_t qnn_graph_ = nullptr;
};

}  // namespace litert::qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_
