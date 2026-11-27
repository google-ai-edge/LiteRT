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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

// Deterministic content and size metadata recorded for each constant tensor
// buffer defined on an `XnnpackGraph`, enabling `SharedXnnpackState` to match
// disk-backed `.xnwc` entries by weight/bias content identity and validate
// minimum packed byte bounds.
struct XnnConstantFingerprint {
  const void* data_ptr = nullptr;
  const void* source_buffer = nullptr;
  const void* source_quantization = nullptr;
  uint64_t content_hash = 0;
  uint64_t min_packed_bytes = 0;
};

class XnnpackGraph : public NnpackGraph {
 public:
  struct XnnSubgraphDeleter {
    void operator()(xnn_subgraph_t sg) const { xnn_delete_subgraph(sg); }
  };
  using UniquePtr = std::unique_ptr<xnn_subgraph, XnnSubgraphDeleter>;

  explicit XnnpackGraph(UniquePtr subgraph = nullptr)
      : subgraph_(std::move(subgraph)) {}

  XnnpackGraph(XnnpackGraph&&) = default;
  XnnpackGraph& operator=(XnnpackGraph&&) = default;

  absl::Status ResetSubgraph(uint32_t external_value_ids, uint32_t flags) {
    xnn_subgraph_t subgraph = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_create_subgraph(external_value_ids, flags, &subgraph));
    subgraph_.reset(subgraph);
    constant_fingerprints_.clear();
    return absl::OkStatus();
  }

  ~XnnpackGraph() override = default;

  xnn_subgraph_t GetSubgraph() const { return subgraph_.get(); }
  xnn_subgraph_t ReleaseSubgraph() { return subgraph_.release(); }
  void DeleteSubgraph() { subgraph_.reset(); }

  std::vector<XnnConstantFingerprint>& constant_fingerprints() {
    return constant_fingerprints_;
  }
  const std::vector<XnnConstantFingerprint>& constant_fingerprints() const {
    return constant_fingerprints_;
  }

  struct BiasedQb4wBuffer {
    uint32_t value_id = UINT32_MAX;
    std::vector<uint8_t> data;
  };
  // Temporary unsigned-biased (`^ 0x88`, `zero_point = 8`) copies of signed
  // `kI4` blockwise weights so XNNPACK unlocks Arm64 KleidiAI `qp8_f32_qb4w`
  // kernels. Released after `xnn_create_runtime_v4` packs `packed_only` values.
  std::vector<BiasedQb4wBuffer>& biased_qb4w_buffers() {
    return biased_qb4w_buffers_;
  }
  const std::vector<BiasedQb4wBuffer>& biased_qb4w_buffers() const {
    return biased_qb4w_buffers_;
  }

  // Maps temporary biased weight pointers back to their original constant
  // buffer pointers for `xnn_weights_cache_provider::alias_data`.
  std::vector<std::pair<const void*, const void*>>& weight_aliases() {
    return weight_aliases_;
  }
  const std::vector<std::pair<const void*, const void*>>& weight_aliases()
      const {
    return weight_aliases_;
  }

 private:
  UniquePtr subgraph_;
  std::vector<XnnConstantFingerprint> constant_fingerprints_;
  std::vector<BiasedQb4wBuffer> biased_qb4w_buffers_;
  std::vector<std::pair<const void*, const void*>> weight_aliases_;
};

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_
