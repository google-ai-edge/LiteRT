/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_GRAPH_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

class YnnpackGraph : public NnpackGraph {
 public:
  struct YnnSubgraphDeleter {
    void operator()(ynn_subgraph_t sg) const { ynn_delete_subgraph(sg); }
  };
  using UniquePtr = std::unique_ptr<ynn_subgraph, YnnSubgraphDeleter>;

  explicit YnnpackGraph(UniquePtr subgraph = nullptr)
      : subgraph_(std::move(subgraph)) {}

  YnnpackGraph(YnnpackGraph&&) = default;
  YnnpackGraph& operator=(YnnpackGraph&&) = default;

  absl::Status ResetSubgraph(uint32_t external_value_ids, uint32_t flags) {
    ynn_subgraph_t subgraph = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        ynn_create_subgraph(external_value_ids, flags, &subgraph));
    subgraph_.reset(subgraph);
    return absl::OkStatus();
  }

  ~YnnpackGraph() override = default;

  ynn_subgraph_t GetSubgraph() const { return subgraph_.get(); }
  ynn_subgraph_t ReleaseSubgraph() { return subgraph_.release(); }

 private:
  UniquePtr subgraph_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_YNNPACK_GRAPH_H_
