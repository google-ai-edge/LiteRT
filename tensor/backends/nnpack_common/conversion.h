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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_CONVERSION_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/nnpack_common/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

struct NnpackValue {
  graph::TensorInformation info;
  uint32_t id = UINT32_MAX;
  uint32_t flags = 0;
  LockedBufferSpan<const std::byte> data =
      LockedBufferSpan<const std::byte>::Empty();
};

absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info, absl::Span<const std::byte> raw_data);

absl::Status TopologicalSort(
    graph::Tensor output,
    const absl::flat_hash_set<graph::Tensor>& inlined_inputs,
    absl::flat_hash_set<graph::Tensor>& visited_tensors,
    absl::flat_hash_set<const graph::Operation*>& visited_ops,
    std::vector<const graph::Operation*>& ordered_ops);

template <typename Traits>
class NnpackGraph {
 public:
  using SubgraphType = typename Traits::SubgraphType;
  using ValueType = typename Traits::ValueType;

  NnpackGraph(SubgraphType subgraph, std::vector<ValueType> values,
              absl::flat_hash_map<graph::Tensor, size_t> tensor_index,
              absl::flat_hash_set<graph::Tensor> external_outputs,
              std::vector<std::vector<float>> dequantized_buffers = {},
              std::vector<std::vector<fp16_t>> fp16_buffers = {},
              std::vector<std::vector<char>> constant_buffers = {},
              std::vector<std::shared_ptr<Buffer>> keep_alive_buffers = {})
      : subgraph_(subgraph),
        values_(std::move(values)),
        tensor_index_(std::move(tensor_index)),
        external_outputs_(std::move(external_outputs)),
        dequantized_buffers_(std::move(dequantized_buffers)),
        fp16_buffers_(std::move(fp16_buffers)),
        constant_buffers_(std::move(constant_buffers)),
        keep_alive_buffers_(std::move(keep_alive_buffers)) {}

  ~NnpackGraph() {
    if (subgraph_ != nullptr) {
      Traits::DeleteSubgraph(subgraph_);
    }
  }

  SubgraphType subgraph() const { return subgraph_; }
  std::vector<ValueType>& mutable_values() { return values_; }
  const std::vector<ValueType>& values() const { return values_; }
  const absl::flat_hash_set<graph::Tensor>& external_outputs() const {
    return external_outputs_;
  }

  absl::StatusOr<size_t> Lookup(const TensorHandle& tensor) const {
    auto it = tensor_index_.find(tensor.GetRaw());
    if (it == tensor_index_.end()) {
      return absl::NotFoundError(
          absl::StrFormat("Tensor %s not found in graph.", tensor.GetName()));
    }
    return it->second;
  }

 private:
  SubgraphType subgraph_ = nullptr;
  std::vector<ValueType> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  std::vector<std::vector<float>> dequantized_buffers_;
  std::vector<std::vector<fp16_t>> fp16_buffers_;
  std::vector<std::vector<char>> constant_buffers_;
  std::vector<std::shared_ptr<Buffer>> keep_alive_buffers_;
};

template <typename Traits>
class NnpackBuildContext {
 public:
  using SubgraphType = typename Traits::SubgraphType;
  using ValueType = typename Traits::ValueType;
  using GraphType = NnpackGraph<Traits>;

  explicit NnpackBuildContext(
      std::vector<TensorHandle> outputs,
      absl::flat_hash_map<graph::Tensor, uint32_t> external_ids = {})
      : outputs_(std::move(outputs)), external_ids_(std::move(external_ids)) {
    for (const auto& out : outputs_) {
      external_outputs_.insert(out.GetRaw());
    }
  }

  ~NnpackBuildContext() {
    if (subgraph_ != nullptr) {
      Traits::DeleteSubgraph(subgraph_);
    }
  }

  absl::Status Init() {
    LRT_TENSOR_RETURN_IF_ERROR(Traits::EnsureInitialized());
    LRT_TENSOR_RETURN_IF_ERROR(
        Traits::CreateSubgraph(external_ids_.size(), /*flags=*/0, &subgraph_));
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<GraphType>> Finalize() {
    for (const auto& out : outputs_) {
      LRT_TENSOR_RETURN_IF_ERROR(DefineValue(out.GetRaw()).status());
    }
    auto graph = std::make_unique<GraphType>(
        subgraph_, std::move(values_), std::move(tensor_index_),
        std::move(external_outputs_), std::move(dequantized_buffers_),
        std::move(fp16_buffers_), std::move(constant_buffers_),
        std::move(keep_alive_buffers_));
    subgraph_ = nullptr;
    return graph;
  }

  absl::StatusOr<uint32_t> DefineValue(const graph::Tensor& tensor) {
    if (auto it = tensor_index_.find(tensor); it != tensor_index_.end()) {
      return values_[it->second].id;
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(tensor));
    ValueType value;
    value.info = info;

    if (auto it = external_ids_.find(tensor); it != external_ids_.end()) {
      value.id = it->second;
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> producer,
                                graph::GetProducer(tensor));
    const bool is_external_input =
        info.buffer == nullptr && producer == nullptr;
    const bool is_external_output = external_outputs_.contains(tensor);

    if (is_external_output) {
      value.flags |= Traits::kFlagExternalOutput;
    }
    if (is_external_input) {
      value.flags |= Traits::kFlagExternalInput;
    }

    LRT_TENSOR_RETURN_IF_ERROR(Traits::DefineTensorValue(*this, tensor, value));

    tensor_index_[tensor] = values_.size();
    values_.push_back(std::move(value));
    return values_.back().id;
  }

  absl::Status AliasValue(const graph::Tensor& source,
                          const graph::Tensor& target) {
    if (auto it = tensor_index_.find(target); it != tensor_index_.end()) {
      tensor_index_[source] = it->second;
      return absl::OkStatus();
    }
    LRT_TENSOR_RETURN_IF_ERROR(DefineValue(target).status());
    tensor_index_[source] = tensor_index_[target];
    return absl::OkStatus();
  }

  void RemoveTensor(const graph::Tensor& tensor) {
    tensor_index_.erase(tensor);
  }

  template <typename DatatypeT>
  absl::StatusOr<uint32_t> DefineConstant(const void* data, size_t bytes,
                                          DatatypeT datatype,
                                          std::vector<size_t> shape) {
    constant_buffers_.emplace_back(reinterpret_cast<const char*>(data),
                                   reinterpret_cast<const char*>(data) + bytes);
    const void* copied_data_ptr = constant_buffers_.back().data();

    uint32_t id = UINT32_MAX;
    LRT_TENSOR_RETURN_IF_ERROR(Traits::DefineConstantTensor(
        subgraph_, datatype, shape, copied_data_ptr, &id));
    return id;
  }

  SubgraphType subgraph() { return subgraph_; }
  std::vector<std::vector<float>>& dequantized_buffers() {
    return dequantized_buffers_;
  }
  std::vector<std::vector<fp16_t>>& fp16_buffers() { return fp16_buffers_; }
  std::vector<std::vector<char>>& constant_buffers() {
    return constant_buffers_;
  }
  std::vector<std::shared_ptr<Buffer>>& keep_alive_buffers() {
    return keep_alive_buffers_;
  }

 private:
  std::vector<TensorHandle> outputs_;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  SubgraphType subgraph_ = nullptr;
  std::vector<ValueType> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  std::vector<std::vector<float>> dequantized_buffers_;
  std::vector<std::vector<fp16_t>> fp16_buffers_;
  std::vector<std::vector<char>> constant_buffers_;
  std::vector<std::shared_ptr<Buffer>> keep_alive_buffers_;
};

template <typename Traits>
absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs,
    NnpackBuildContext<Traits>& ctx) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> op_outputs,
                              graph::GetOutputs(op));
  if (op_outputs.size() != inlined_outputs.size()) {
    return absl::InvalidArgumentError("Output size mismatch");
  }
  for (size_t i = 0; i < op_outputs.size(); ++i) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ctx.AliasValue(inlined_outputs[i], op_outputs[i]));
  }

  for (size_t i = 0; i < std::min(inlined_inputs.size(), op.inputs.size());
       ++i) {
    if (graph::GetStatus(inlined_inputs[i]).ok() &&
        graph::GetStatus(op.inputs[i]).ok()) {
      LRT_TENSOR_RETURN_IF_ERROR(
          ctx.AliasValue(inlined_inputs[i], op.inputs[i]));
    }
  }

  absl::flat_hash_set<graph::Tensor> inlined_inputs_set;
  for (const auto& t : inlined_inputs) {
    if (graph::GetStatus(t).ok()) {
      inlined_inputs_set.insert(t);
    }
  }

  absl::flat_hash_set<graph::Tensor> visited_tensors;
  absl::flat_hash_set<const graph::Operation*> visited_ops;
  std::vector<const graph::Operation*> ordered_ops;

  for (const auto& out : inlined_outputs) {
    LRT_TENSOR_RETURN_IF_ERROR(TopologicalSort(
        out, inlined_inputs_set, visited_tensors, visited_ops, ordered_ops));
  }

  for (const auto* inline_op : ordered_ops) {
    auto op_ext =
        inline_op->template GetExtension<typename Traits::OpExtensionType>();
    if (op_ext == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Operation ", inline_op->GetName(), " does not implement ",
          Traits::kBackendName, " operation."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(Traits::LowerOp(*op_ext, *inline_op, ctx));
  }

  for (const auto& t : inlined_outputs) {
    ctx.RemoveTensor(t);
  }
  for (const auto& t : inlined_inputs) {
    ctx.RemoveTensor(t);
  }
  return absl::OkStatus();
}

template <typename Traits>
absl::StatusOr<std::unique_ptr<NnpackGraph<Traits>>> BuildNnpackGraph(
    std::vector<TensorHandle> outputs) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(outputs));

  uint32_t next_id = 0;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  for (const TensorHandle& out : outputs) {
    auto [it, inserted] = external_ids.insert({out.GetRaw(), next_id});
    next_id += inserted;
  }
  for (const graph::Operation* op : plan) {
    for (const graph::Tensor& t : op->inputs) {
      if (auto info_or = graph::GetInfo(t);
          !info_or.ok() || info_or->buffer != nullptr) {
        continue;
      }
      if (auto producer_or = graph::GetProducer(t);
          producer_or.ok() && *producer_or != nullptr) {
        continue;
      }
      auto [it, inserted] = external_ids.insert({t, next_id});
      next_id += inserted;
    }
  }

  NnpackBuildContext<Traits> ctx(std::move(outputs), std::move(external_ids));
  LRT_TENSOR_RETURN_IF_ERROR(ctx.Init());

  for (const graph::Operation* op : plan) {
    auto op_ext = op->template GetExtension<typename Traits::OpExtensionType>();
    if (op_ext == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", op->GetName(), " does not implement ",
                       Traits::kBackendName, " operation."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(Traits::LowerOp(*op_ext, *op, ctx))
        << "Failed to convert " << op->GetName() << " to "
        << Traits::kBackendName << ".";
  }

  return ctx.Finalize();
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_BACKENDS_NNPACK_COMMON_CONVERSION_H_
