/* Copyright 2025 Google LLC.

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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"  // IWYU pragma: export
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

struct xnn_subgraph;

namespace litert::tensor {

// Represents an XNNPACK graph.
class XnnpackGraph {
 public:
  XnnpackGraph(xnn_subgraph* subgraph, std::vector<XnnpackValue> values,
               absl::flat_hash_map<graph::Tensor, size_t> tensor_index,
               absl::flat_hash_set<graph::Tensor> external_outputs,
               std::vector<std::vector<float>> dequantized_buffers = {},
               std::vector<std::vector<fp16_t>> fp16_buffers = {},
               std::vector<std::vector<char>> constant_buffers = {},
               std::vector<std::shared_ptr<Buffer>> keep_alive_buffers = {});
  ~XnnpackGraph();

  // Returns the XNNPACK subgraph.
  xnn_subgraph* subgraph() const { return subgraph_; }

  // Returns the values in the XNNPACK graph.
  std::vector<XnnpackValue>& mutable_values() { return values_; }

  // Returns the XnnpackValue vector in the XNNPACK graph.
  const std::vector<XnnpackValue>& values() const { return values_; }

  // Looks up the index of the given tensor in the XNNPACK graph.
  absl::StatusOr<size_t> Lookup(const TensorHandle& tensor) const;

  // Returns the external outputs of the XNNPACK graph.
  const absl::flat_hash_set<graph::Tensor>& external_outputs() const {
    return external_outputs_;
  }

 private:
  xnn_subgraph* subgraph_ = nullptr;
  std::vector<XnnpackValue> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  std::vector<std::vector<float>> dequantized_buffers_;
  std::vector<std::vector<fp16_t>> fp16_buffers_;
  std::vector<std::vector<char>> constant_buffers_;
  std::vector<std::shared_ptr<Buffer>> keep_alive_buffers_;
};

// Builds an XNNPACK graph from the given outputs.
absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs);

// Lowers the implementation graph of an operation to XNNPACK.
//
// It maps the operation's inputs and outputs to the corresponding inlined
// tensors. `inlined_inputs` must be positionally aligned with `op.inputs`.
// If an input is not used or is a constant created in-place
// (not mapped to `op.inputs`), an invalid `graph::Tensor` (default constructed)
// should be used as a placeholder in `inlined_inputs` to maintain alignment.
//
// It then traverses the inlined graph between the inlined inputs and outputs
// and lowers all operations.
absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs, XnnpackBuildContext& ctx);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
