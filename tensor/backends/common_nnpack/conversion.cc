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

#include "tensor/backends/common_nnpack/conversion.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError("Missing quantization parameters");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& pcq,
      info.quantization->As<const graph::PerChannelAffineQuantization>());

  if (info.shape.size() != 2) {
    return absl::InvalidArgumentError("Only 2D weights currently supported");
  }
  const size_t out_channels = info.shape[0];
  const size_t in_channels = info.shape[1];
  const size_t num_elements = out_channels * in_channels;

  if (raw_data.size() < num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Raw data size %zu is smaller than expected %zu",
                        raw_data.size(), num_elements));
  }

  std::vector<float> dequantized(num_elements);
  const int8_t* quantized_data =
      reinterpret_cast<const int8_t*>(raw_data.data());

  for (size_t o = 0; o < out_channels; ++o) {
    const float scale = pcq.scales[o];
    const int32_t zero_point =
        pcq.zero_points.empty() ? 0 : static_cast<int32_t>(pcq.zero_points[o]);
    const size_t row_offset = o * in_channels;
    for (size_t i = 0; i < in_channels; ++i) {
      const size_t idx = row_offset + i;
      dequantized[idx] =
          static_cast<float>(quantized_data[idx] - zero_point) * scale;
    }
  }

  return dequantized;
}

absl::Status TopologicalSort(
    graph::Tensor output,
    const absl::flat_hash_set<graph::Tensor>& inlined_inputs,
    absl::flat_hash_set<graph::Tensor>& visited_tensors,
    absl::flat_hash_set<const graph::Operation*>& visited_ops,
    std::vector<const graph::Operation*>& ordered_ops) {
  using StackEntry = std::variant<graph::Tensor, const graph::Operation*>;
  std::vector<StackEntry> stack;
  stack.push_back(output);

  while (!stack.empty()) {
    StackEntry entry = std::move(stack.back());
    stack.pop_back();

    if (std::holds_alternative<const graph::Operation*>(entry)) {
      ordered_ops.push_back(std::get<const graph::Operation*>(entry));
      continue;
    }

    graph::Tensor tensor = std::get<graph::Tensor>(entry);

    if (inlined_inputs.contains(tensor)) {
      continue;
    }
    if (!visited_tensors.insert(tensor).second) {
      continue;
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> producer,
                                graph::GetProducer(tensor));
    if (producer == nullptr) {
      continue;
    }

    if (!visited_ops.insert(producer.get()).second) {
      continue;
    }

    stack.push_back(producer.get());

    for (auto it = producer->inputs.rbegin(); it != producer->inputs.rend();
         ++it) {
      stack.push_back(*it);
    }
  }

  return absl::OkStatus();
}

NnpackBuildContext::NnpackBuildContext(
    std::vector<TensorHandle> outputs,
    absl::flat_hash_map<graph::Tensor, uint32_t> external_ids)
    : outputs_(std::move(outputs)), external_ids_(std::move(external_ids)) {}

absl::Status NnpackBuildContext::Init() {
  LRT_TENSOR_RETURN_IF_ERROR(EnsureInitialized());
  graph_ = CreateEmptyGraph();
  if (graph_ == nullptr) {
    return absl::InternalError("Failed to create graph instance");
  }
  for (const TensorHandle& out : outputs_) {
    graph_->mutable_external_outputs().insert(out.GetRaw());
  }
  LRT_TENSOR_RETURN_IF_ERROR(CreateSubgraph(external_ids_.size(), /*flags=*/0));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<NnpackGraph>> NnpackBuildContext::Finalize() {
  for (const TensorHandle& out : outputs_) {
    LRT_TENSOR_RETURN_IF_ERROR(DefineValue(out.GetRaw()).status());
  }
  return std::move(graph_);
}

absl::StatusOr<uint32_t> NnpackBuildContext::DefineValue(
    const graph::Tensor& tensor) {
  absl::flat_hash_map<graph::Tensor, size_t>& tensor_index =
      graph_->mutable_tensor_index();
  std::vector<NnpackValue>& values = graph_->mutable_values();
  if (auto it = tensor_index.find(tensor); it != tensor_index.end()) {
    return values[it->second].id;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  NnpackValue value;
  value.info = info;

  if (auto it = external_ids_.find(tensor); it != external_ids_.end()) {
    value.id = it->second;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> producer,
                              graph::GetProducer(tensor));
  const bool is_external_input = info.buffer == nullptr && producer == nullptr;
  const bool is_external_output = graph_->external_outputs().contains(tensor);

  if (is_external_output) {
    value.flags |= FlagExternalOutput();
  }
  if (is_external_input) {
    value.flags |= FlagExternalInput();
  }

  LRT_TENSOR_RETURN_IF_ERROR(DefineTensorValue(tensor, value));

  tensor_index[tensor] = values.size();
  values.push_back(std::move(value));
  return values.back().id;
}

absl::Status NnpackBuildContext::AliasValue(const graph::Tensor& source,
                                            const graph::Tensor& target) {
  absl::flat_hash_map<graph::Tensor, size_t>& tensor_index =
      graph_->mutable_tensor_index();
  if (auto it = tensor_index.find(target); it != tensor_index.end()) {
    tensor_index[source] = it->second;
    return absl::OkStatus();
  }
  LRT_TENSOR_RETURN_IF_ERROR(DefineValue(target).status());
  tensor_index[source] = tensor_index[target];
  return absl::OkStatus();
}

void NnpackBuildContext::RemoveTensor(const graph::Tensor& tensor) {
  graph_->mutable_tensor_index().erase(tensor);
}

absl::StatusOr<uint32_t> NnpackBuildContext::DefineConstant(
    const void* data, size_t bytes, Type datatype, std::vector<size_t> shape) {
  auto& constant_buffers = graph_->constant_buffers();
  constant_buffers.emplace_back(reinterpret_cast<const char*>(data),
                                reinterpret_cast<const char*>(data) + bytes);
  const void* copied_data_ptr = constant_buffers.back().data();

  uint32_t id = UINT32_MAX;
  LRT_TENSOR_RETURN_IF_ERROR(
      DefineConstantTensor(datatype, shape, copied_data_ptr, &id));
  return id;
}

absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs, NnpackBuildContext& ctx) {
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> op_outputs,
                              graph::GetOutputs(op));
  if (op_outputs.size() != inlined_outputs.size()) {
    return absl::InvalidArgumentError("Output size mismatch");
  }
  for (size_t i = 0; i < op_outputs.size(); ++i) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ctx.AliasValue(inlined_outputs[i], op_outputs[i]));
  }

  if (op.inputs.size() != inlined_inputs.size()) {
    return absl::InvalidArgumentError("Input size mismatch");
  }
  for (size_t i = 0; i < op.inputs.size(); ++i) {
    if (graph::GetStatus(inlined_inputs[i]).ok() &&
        graph::GetStatus(op.inputs[i]).ok()) {
      LRT_TENSOR_RETURN_IF_ERROR(
          ctx.AliasValue(inlined_inputs[i], op.inputs[i]));
    }
  }

  absl::flat_hash_set<graph::Tensor> inlined_inputs_set;
  for (const graph::Tensor& t : inlined_inputs) {
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
    LRT_TENSOR_RETURN_IF_ERROR(ctx.LowerOp(*inline_op));
  }

  ctx.RemoveTensors(inlined_outputs);
  ctx.RemoveTensors(inlined_inputs);
  return absl::OkStatus();
}

absl::Status BuildNnpackGraph(NnpackBuildContext& ctx) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(ctx.outputs_));
  LRT_TENSOR_RETURN_IF_ERROR(ctx.Init());

  for (const graph::Operation* op : plan) {
    LRT_TENSOR_RETURN_IF_ERROR(ctx.LowerOp(*op))
        << "Failed to convert " << op->GetName() << " to " << ctx.BackendName()
        << ".";
  }

  return absl::OkStatus();
}

}  // namespace litert::tensor
