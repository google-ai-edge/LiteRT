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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/tensor.h"

namespace litert::tensor {

absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info, absl::Span<const std::byte> raw_data);

// Order operations in topological execution order.
//
// Parameters:
// - `output`: the output tensor of the graph to sort.
// - `inlined_inputs`: input tensors of the graph to sort. This allows us
//   sorting a partial graph.
// - `visited_tensors`: a set tracking visited tensors across traversals.
// - `visited_ops`: a set tracking visited operations across traversals.
// - `ordered_ops`: the list of operations in topological order.
absl::Status TopologicalSort(
    graph::Tensor output,
    const absl::flat_hash_set<graph::Tensor>& inlined_inputs,
    absl::flat_hash_set<graph::Tensor>& visited_tensors,
    absl::flat_hash_set<const graph::Operation*>& visited_ops,
    std::vector<const graph::Operation*>& ordered_ops);

class NnpackBuildContext {
 public:
  explicit NnpackBuildContext(
      std::vector<TensorHandle> outputs,
      absl::flat_hash_map<graph::Tensor, uint32_t> external_ids = {});
  virtual ~NnpackBuildContext() = default;

  absl::Status Init();
  absl::StatusOr<std::unique_ptr<NnpackGraph>> Finalize();
  absl::StatusOr<uint32_t> DefineValue(const graph::Tensor& tensor);
  absl::Status AliasValue(const graph::Tensor& source,
                          const graph::Tensor& target);
  void RemoveTensor(const graph::Tensor& tensor);

  template <class Sequence>
  void RemoveTensors(Sequence&& tensors) {
    for (auto&& t : tensors) {
      RemoveTensor(t);
    }
  }

  absl::StatusOr<uint32_t> DefineConstant(const void* data, size_t bytes,
                                          Type datatype,
                                          std::vector<size_t> shape);

  NnpackGraph& graph() { return *graph_; }
  const NnpackGraph& graph() const { return *graph_; }

  virtual absl::string_view BackendName() const = 0;
  virtual uint32_t FlagExternalInput() const = 0;
  virtual uint32_t FlagExternalOutput() const = 0;

 protected:
  virtual absl::Status EnsureInitialized() = 0;
  virtual std::unique_ptr<NnpackGraph> CreateEmptyGraph() = 0;
  virtual absl::Status CreateSubgraph(size_t external_value_ids,
                                      uint32_t flags) = 0;
  virtual absl::Status DefineTensorValue(const graph::Tensor& tensor,
                                         NnpackValue& value) = 0;
  virtual absl::Status DefineConstantTensor(Type datatype,
                                            absl::Span<const size_t> shape,
                                            const void* data, uint32_t* id) = 0;
  virtual absl::Status LowerOp(const graph::Operation& op) = 0;

  friend absl::Status InlineImplementationGraphFor(
      const graph::Operation& op,
      absl::Span<const graph::Tensor> inlined_inputs,
      absl::Span<const graph::Tensor> inlined_outputs, NnpackBuildContext& ctx);

  friend absl::Status BuildNnpackGraph(NnpackBuildContext& ctx);

  std::vector<TensorHandle> outputs_;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids_;
  std::unique_ptr<NnpackGraph> graph_;
};

absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs, NnpackBuildContext& ctx);

absl::Status BuildNnpackGraph(NnpackBuildContext& ctx);

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_COMMON_NNPACK_CONVERSION_H_
