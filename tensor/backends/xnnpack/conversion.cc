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

#include "tensor/backends/xnnpack/conversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

absl::once_flag g_xnn_init_once;
absl::NoDestructor<absl::Status> g_xnn_init_status(absl::OkStatus());

absl::Status EnsureXnnInitialized() {
  absl::call_once(g_xnn_init_once, []() {
    *g_xnn_init_status =
        XnnStatusToAbsl(xnn_initialize(nullptr), "xnn_initialize");
  });
  return *g_xnn_init_status;
}

xnn_datatype GetXnnpackType(const XnnpackValue& value) {
  switch (value.info.type) {
    case Type::kUnknown:
    case Type::kBOOL:
    case Type::kI2:
    case Type::kI4:
      if (value.info.quantization) {
        if (value.info.quantization->As<PerChannelAffineQuantization>().ok()) {
          return xnn_datatype_qcint4;
        } else if (value.info.quantization->As<BlockwiseQuantization>().ok()) {
          return xnn_datatype_qbint4;
        }
      }
      break;
    case Type::kI8:
      if (value.info.quantization) {
        if (auto it =
                value.info.quantization->As<PerChannelAffineQuantization>();
            it.ok()) {
          return it->scales.size() > 1 ? xnn_datatype_qcint8
                                       : xnn_datatype_qint8;
        }
      }
      break;
    case Type::kI16:
    case Type::kI64:
    case Type::kU4:
    case Type::kU8:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kFP16:
      return xnn_datatype_fp16;
    case Type::kI32:
      return xnn_datatype_int32;
    case Type::kFP32:
      return xnn_datatype_fp32;
    case Type::kFP64:
      break;
    case Type::kBF16:
      return xnn_datatype_bf16;
  }
  return xnn_datatype_invalid;
}

// TODO: b/493560478 - Decide whether to delete this from here.
[[maybe_unused]]
size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

absl::StatusOr<size_t> NumElements(const graph::TensorInformation& info) {
  size_t num_elements = 1;
  for (int32_t dim : info.shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError("Negative tensor dimension.");
    }
    if (dim == 0) {
      return static_cast<size_t>(0);
    }
    if (num_elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return absl::InvalidArgumentError("Tensor element count overflow.");
    }
    num_elements *= static_cast<size_t>(dim);
  }
  return num_elements;
}

enum class QuantParamMode {
  kScalar,
  kPerChannelDim0,
  kPerElement,
};

absl::StatusOr<QuantParamMode> DetermineQuantParamMode(
    size_t param_size, const graph::TensorInformation& info,
    size_t num_elements, absl::string_view param_name) {
  if (param_size == 1) {
    return QuantParamMode::kScalar;
  }
  if (param_size == num_elements) {
    return QuantParamMode::kPerElement;
  }
  if (!info.shape.empty() && info.shape[0] > 0 &&
      param_size == static_cast<size_t>(info.shape[0])) {
    return QuantParamMode::kPerChannelDim0;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported ", param_name, " size for tensor ", info.name));
}

size_t QuantParamIndex(QuantParamMode mode, size_t element_index,
                       size_t channel_index) {
  switch (mode) {
    case QuantParamMode::kScalar:
      return 0;
    case QuantParamMode::kPerChannelDim0:
      return channel_index;
    case QuantParamMode::kPerElement:
      return element_index;
  }
  return 0;
}

// TODO: b/493560478 - Decide whether to delete this from here.
[[maybe_unused]]
absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError(
        "INT8 constant tensor missing quantization metadata.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& quantization,
      info.quantization->As<const PerChannelAffineQuantization>());
  if (quantization.quantized_dimension != 0) {
    return absl::UnimplementedError(
        absl::StrCat("Only quantized_dimension=0 is supported for XNNPACK "
                     "dequantization, got ",
                     quantization.quantized_dimension));
  }
  if (quantization.scales.empty() || quantization.zero_points.empty()) {
    return absl::InvalidArgumentError(
        "INT8 constant tensor has empty quantization params.");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto num_elements, NumElements(info));
  if (raw_data.size() != num_elements * sizeof(int8_t)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "INT8 constant tensor byte size mismatch for ", info.name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto scale_mode, DetermineQuantParamMode(quantization.scales.size(), info,
                                               num_elements, "scale"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto zero_point_mode,
      DetermineQuantParamMode(quantization.zero_points.size(), info,
                              num_elements, "zero_point"));

  size_t channel_size = 0;
  if (!info.shape.empty() && info.shape[0] > 0) {
    channel_size = num_elements / static_cast<size_t>(info.shape[0]);
  }

  std::vector<float> values(num_elements);
  const int8_t* quantized_data =
      reinterpret_cast<const int8_t*>(raw_data.data());
  for (size_t i = 0; i < num_elements; ++i) {
    const size_t channel_index = channel_size == 0 ? 0 : i / channel_size;
    const size_t scale_index = QuantParamIndex(scale_mode, i, channel_index);
    const size_t zp_index = QuantParamIndex(zero_point_mode, i, channel_index);
    values[i] = static_cast<float>(
                    static_cast<int32_t>(quantized_data[i]) -
                    static_cast<int32_t>(quantization.zero_points[zp_index])) *
                quantization.scales[scale_index];
  }
  return values;
}

// Performs a topological sort of the operations required to produce
// `output`, traversing backwards from `output to its producers.
// This function assumes the graph is acyclic and does not verify this property.
// This function performs a depth-first search (DFS) starting from
// `output`. It stops traversing back when it hits a tensor in
// `inlined_inputs`.
// To support multiple outputs, this function can be called repeatedly for each
// output tensor. By sharing `visited_tensors`, `visited_ops`, and `ordered_ops`
// across calls, it ensures that each operation is visited and ordered exactly
// once.
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
      // Already in visited_tensors.
      continue;
    }

    LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> producer,
                                graph::GetProducer(tensor));
    if (producer == nullptr) {
      continue;
    }

    if (!visited_ops.insert(producer.get()).second) {
      // Already in visited_ops.
      continue;
    }

    stack.push_back(producer.get());

    // Push inputs in reverse order onto the stack so they are popped and
    // processed in their natural left-to-right order.
    for (auto it = producer->inputs.rbegin(); it != producer->inputs.rend();
         ++it) {
      stack.push_back(*it);
    }
  }

  return absl::OkStatus();
}

}  // namespace

XnnpackGraph::XnnpackGraph(
    xnn_subgraph* subgraph, std::vector<XnnpackValue> values,
    absl::flat_hash_map<graph::Tensor, size_t> tensor_index,
    absl::flat_hash_set<graph::Tensor> external_outputs,
    std::vector<std::vector<float>> dequantized_buffers,
    std::vector<std::vector<fp16_t>> fp16_buffers,
    std::vector<std::vector<char>> constant_buffers,
    std::vector<std::shared_ptr<Buffer>> keep_alive_buffers)
    : subgraph_(subgraph),
      values_(std::move(values)),
      tensor_index_(std::move(tensor_index)),
      external_outputs_(std::move(external_outputs)),
      dequantized_buffers_(std::move(dequantized_buffers)),
      fp16_buffers_(std::move(fp16_buffers)),
      constant_buffers_(std::move(constant_buffers)),
      keep_alive_buffers_(std::move(keep_alive_buffers)) {}

XnnpackGraph::~XnnpackGraph() {
  if (subgraph_ != nullptr) {
    xnn_delete_subgraph(subgraph_);
  }
}

absl::StatusOr<size_t> XnnpackGraph::Lookup(const TensorHandle& tensor) const {
  auto it = tensor_index_.find(tensor.GetRaw());
  if (it == tensor_index_.end()) {
    return absl::NotFoundError("Tensor not part of XNNPACK graph");
  }
  return it->second;
}

XnnpackBuildContext::XnnpackBuildContext(
    std::vector<TensorHandle> outputs,
    absl::flat_hash_map<graph::Tensor, uint32_t> external_ids)
    : external_ids_(std::move(external_ids)) {
  outputs_.reserve(outputs.size());
  for (TensorHandle& handle : outputs) {
    outputs_.push_back(handle.GetRaw());
    external_outputs_.insert(handle.GetRaw());
  }
}

XnnpackBuildContext::~XnnpackBuildContext() {
  if (subgraph_ != nullptr) {
    xnn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;
  }
}

absl::Status XnnpackBuildContext::Init() {
  LRT_TENSOR_RETURN_IF_ERROR(EnsureXnnInitialized());
  const size_t reserved_external_ids = external_ids_.size();
  LRT_TENSOR_RETURN_IF_ERROR(xnn_create_subgraph(
      /*external_value_ids=*/reserved_external_ids, /*flags=*/0, &subgraph_))
      << "xnn_create_subgraph";
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> XnnpackBuildContext::Finalize() {
  for (const graph::Tensor& tensor : outputs_) {
    LRT_TENSOR_RETURN_IF_ERROR(DefineValue(tensor));
  }

  auto* subgraph = subgraph_;
  subgraph_ = nullptr;

  return std::make_unique<XnnpackGraph>(
      subgraph, std::move(values_), std::move(tensor_index_),
      std::move(external_outputs_), std::move(dequantized_buffers_),
      std::move(fp16_buffers_), std::move(constant_buffers_),
      std::move(keep_alive_buffers_));
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
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
      // If the tensor doesn't have a producer, then it's an external input to
      // the graph.
      if (auto producer_or = graph::GetProducer(t);
          producer_or.ok() && *producer_or != nullptr) {
        continue;
      }
      auto [it, inserted] = external_ids.insert({t, next_id});
      next_id += inserted;
    }
  }

  XnnpackBuildContext ctx(std::move(outputs), std::move(external_ids));
  LRT_TENSOR_RETURN_IF_ERROR(ctx.Init());

  for (const graph::Operation* op : plan) {
    auto xnn_op = op->GetExtension<XnnpackOperation>();
    if (xnn_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", op->GetName(),
                       " does not implement XnnpackOperation."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(xnn_op->ToXnnpack(*op, ctx))
        << "Failed to convert " << op->GetName() << " to XNNPack.";
  }

  return ctx.Finalize();
}

absl::StatusOr<uint32_t> XnnpackBuildContext::DefineValue(
    const graph::Tensor& tensor) {
  if (auto it = tensor_index_.find(tensor); it != tensor_index_.end()) {
    return values_[it->second].id;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> & producer,
                              graph::GetProducer(tensor));
  const bool is_external_input = info.buffer == nullptr && producer == nullptr;
  const bool is_external_output = external_outputs_.contains(tensor);
  const bool is_external = is_external_input || is_external_output;

  XnnpackValue value{
      .info = info,
      .flags = (is_external_input ? XNN_VALUE_FLAG_EXTERNAL_INPUT : 0u) |
               (is_external_output ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0u),
  };

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
    KeepAlive(info.buffer);
  }

  if (absl::c_any_of(info.shape, [](int d){return d < 0;})) {
    return absl::InvalidArgumentError(absl::StrFormat(
             "%s: tensor has negative dimension; dynamic shapes are not "
             "supported in the XNNPACK backend", info.name));
  }
  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const void* data_ptr = value.data.data();

  uint32_t external_id = XNN_INVALID_VALUE_ID;
  if (is_external) {
    auto it = external_ids_.find(tensor);
    if (it == external_ids_.end()) {
      return absl::InternalError(
          "External tensor missing reserved external ID");
    }
    external_id = it->second;
  }

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_define_tensor_value(subgraph_, GetXnnpackType(value), dims.size(),
                                dims.empty() ? nullptr : dims.data(), data_ptr,
                                external_id, value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else if (auto maybe_pcq =
                 info.quantization->As<PerChannelAffineQuantization>();
             maybe_pcq.ok()) {
    const auto& pcq = maybe_pcq.value();
    if (pcq.scales.size() == 1) {
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
          subgraph_, GetXnnpackType(value),
          pcq.zero_points.empty() ? 0 : pcq.zero_points[0], pcq.scales[0],
          dims.size(), dims.empty() ? nullptr : dims.data(), data_ptr,
          external_id, value.flags, &value.id))
          << "Could not define a new quantized tensor value.";
    } else {
      bool all_zeros = true;
      for (int64_t zp : pcq.zero_points) {
        if (zp != 0) {
          all_zeros = false;
          break;
        }
      }
      if (!all_zeros) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            std::vector<float> f32_data,
            DequantizeInt8ConstantTensor(info, value.data));
        dequantized_buffers_.push_back(std::move(f32_data));
        data_ptr = dequantized_buffers_.back().data();
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
            subgraph_, xnn_datatype_fp32, dims.size(),
            dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
            value.flags, &value.id))
            << "Could not define a new tensor value after dequantization.";
      } else {
        if (pcq.quantized_dimension < 0 ||
            static_cast<size_t>(pcq.quantized_dimension) >= dims.size() ||
            pcq.scales.size() < dims[pcq.quantized_dimension]) {
          return absl::InvalidArgumentError(absl::StrCat(
              info.name, ": per-channel scale count (", pcq.scales.size(),
              ") is smaller than the channel dimension size (",
              pcq.quantized_dimension >= 0 &&
                      static_cast<size_t>(pcq.quantized_dimension) < dims.size()
                  ? dims[pcq.quantized_dimension]
                  : static_cast<size_t>(0),
              ")"));
        }
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                subgraph_, GetXnnpackType(value), /*zero_point=*/0,
                pcq.scales.data(), dims.size(), pcq.quantized_dimension,
                dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                value.flags, &value.id, /*channelwise_zero_point=*/nullptr))
            << "Could not define a new channelwise quantized tensor value.";
      }
    }
  } else if (auto maybe_bwq = info.quantization->As<BlockwiseQuantization>();
             maybe_bwq.ok()) {
    const auto& bwq = maybe_bwq.value();
    if (dims.size() < 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name,
          ": blockwise quantized tensor requires at least 2 dimensions"));
    }
    if (bwq.block_size == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise quantized tensor block_size must be > 0"));
    }
    const size_t expected_block_count =
        dims[0] * dims[1] / bwq.block_size;
    if (bwq.scales.size() < expected_block_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise scale count (", bwq.scales.size(),
          ") is smaller than block_count (", expected_block_count, ")"));
    }
    fp16_buffers_.emplace_back(bwq.scales.begin(), bwq.scales.end());
    const void* scale_ptr = fp16_buffers_.back().data();
    int32_t zero_point = bwq.zero_points.empty() ? 0 : bwq.zero_points[0];
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_blockwise_quantized_tensor_value_v2(
        subgraph_, GetXnnpackType(value), zero_point, scale_ptr, dims.size(),
        bwq.quantized_dimension, bwq.block_size,
        dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
        value.flags, xnn_datatype_fp16, &value.id))
        << "Could not define a new blockwise quantized tensor value.";
  } else {
    return absl::UnimplementedError("Unsupported quantization type.");
  }

  tensor_index_.emplace(tensor, values_.size());
  values_.emplace_back(std::move(value));
  return values_.back().id;
}

absl::StatusOr<uint32_t> XnnpackBuildContext::DefineConstant(
    const void* data, size_t bytes, ::xnn_datatype datatype,
    std::vector<size_t> shape) {
  constant_buffers_.emplace_back(reinterpret_cast<const char*>(data),
                                 reinterpret_cast<const char*>(data) + bytes);
  const void* copied_data_ptr = constant_buffers_.back().data();

  uint32_t id = XNN_INVALID_VALUE_ID;
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph_, datatype, shape.size(), shape.empty() ? nullptr : shape.data(),
      copied_data_ptr, XNN_INVALID_VALUE_ID,
      /*flags=*/0, &id))
      << "Could not define a new constant tensor value.";
  return id;
}

::xnn_subgraph* XnnpackBuildContext::subgraph() { return subgraph_; }

absl::Status InlineImplementationGraphFor(
    const graph::Operation& op, absl::Span<const graph::Tensor> inlined_inputs,
    absl::Span<const graph::Tensor> inlined_outputs, XnnpackBuildContext& ctx) {
  // Alias outputs
  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<graph::Tensor> op_outputs,
                              graph::GetOutputs(op));
  if (op_outputs.size() != inlined_outputs.size()) {
    return absl::InvalidArgumentError("Output size mismatch");
  }
  for (size_t i = 0; i < op_outputs.size(); ++i) {
    LRT_TENSOR_RETURN_IF_ERROR(
        ctx.AliasValue(inlined_outputs[i], op_outputs[i]));
  }

  // Alias inputs
  for (size_t i = 0; i < std::min(inlined_inputs.size(), op.inputs.size());
       ++i) {
    if (graph::GetStatus(inlined_inputs[i]).ok() &&
        graph::GetStatus(op.inputs[i]).ok()) {
      LRT_TENSOR_RETURN_IF_ERROR(
          ctx.AliasValue(inlined_inputs[i], op.inputs[i]));
    }
  }

  // Only add valid inputs to the boundary set. If an input is invalid (e.g.,
  // a missing optional input), we want TopologicalSort to fail if it attempts
  // to traverse it (by trying to get its producer), rather than stopping at it
  // as if it were a valid boundary. Outputs don't need this check here because
  // they are the starting points of the sort and will trigger a failure
  // immediately if invalid.
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

  // Lower ops
  for (const auto* inline_op : ordered_ops) {
    auto xnn_op = inline_op->GetExtension<XnnpackOperation>();
    if (xnn_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", inline_op->GetName(),
                       " does not implement XnnpackOperation."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(xnn_op->ToXnnpack(*inline_op, ctx));
  }

  // Remove aliases
  for (const auto& t : inlined_outputs) {
    ctx.RemoveTensor(t);
  }
  for (const auto& t : inlined_inputs) {
    ctx.RemoveTensor(t);
  }

  return absl::OkStatus();
}

}  // namespace litert::tensor
