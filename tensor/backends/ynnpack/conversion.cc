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

#include "tensor/backends/ynnpack/conversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/conversion.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/ynnpack/graph.h"
#include "tensor/backends/ynnpack/utils.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

// Returns the storage type of a quantized value, i.e. the YNNPACK type the
// packed integer weights are stored as.
ynn_type GetQuantizedStorageType(const graph::TensorInformation& info) {
  switch (info.type) {
    // `kUnknown` is used by the safetensor loader for sub-byte weights whose
    // element type can only be recovered from the quantization parameters.
    case Type::kI2:
      return ynn_type_int2;
    case Type::kI4:
      return ynn_type_int4;
    case Type::kI8:
      return ynn_type_int8;
    case Type::kU2:
      return ynn_type_uint2;
    default:
      return ToYnnType(info.type);
  }
}

// Appends `values` to the graph's constant buffers and returns a stable pointer
// to the copy. YNNPACK keeps a view over constant data, so the backing storage
// has to outlive the subgraph.
template <class T>
const void* KeepAlive(NnpackGraph& graph, absl::Span<const T> values) {
  const char* begin = reinterpret_cast<const char*>(values.data());
  graph.constant_buffers().emplace_back(begin,
                                        begin + values.size() * sizeof(T));
  return graph.constant_buffers().back().data();
}

}  // namespace

ynn_type ToYnnType(Type type) {
  switch (type) {
    case Type::kI2:
      return ynn_type_int2;
    case Type::kI4:
      return ynn_type_int4;
    case Type::kI8:
      return ynn_type_int8;
    case Type::kI32:
      return ynn_type_int32;
    case Type::kU2:
      return ynn_type_uint2;
    case Type::kU4:
      return ynn_type_uint4;
    case Type::kU8:
    case Type::kBOOL:
      return ynn_type_uint8;
    case Type::kFP16:
      return ynn_type_fp16;
    case Type::kFP32:
      return ynn_type_fp32;
    case Type::kFP64:
      return ynn_type_fp64;
    case Type::kBF16:
      return ynn_type_bf16;
    case Type::kUnknown:
    case Type::kI16:
    case Type::kI64:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
      return ynn_type_invalid;
  }
  return ynn_type_invalid;
}

absl::Status YnnpackBuildContext::EnsureInitialized() {
  // YNNPACK has no global initialization entry point.
  return absl::OkStatus();
}

std::unique_ptr<NnpackGraph> YnnpackBuildContext::CreateEmptyGraph() {
  return std::make_unique<YnnpackGraph>();
}

absl::Status YnnpackBuildContext::CreateSubgraph(size_t external_value_ids,
                                                 uint32_t flags) {
  LRT_TENSOR_RETURN_IF_ERROR(static_cast<YnnpackGraph*>(graph_.get())
                                 ->ResetSubgraph(external_value_ids, flags));
  return absl::OkStatus();
}

absl::Status YnnpackBuildContext::DefineTensorValue(const graph::Tensor& tensor,
                                                    NnpackValue& value) {
  const graph::TensorInformation& info = value.info;
  const bool is_external =
      (value.flags & (FlagExternalInput() | FlagExternalOutput())) != 0;

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
    graph_->keep_alive_buffers().push_back(info.buffer);
  }

  const std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const size_t* dims_ptr = dims.empty() ? nullptr : dims.data();
  const void* data_ptr = value.data.data();
  ynn_subgraph_t sg = subgraph();

  // Only external values may reuse a pre-assigned id; everything else is an
  // internal value that YNNPACK allocates an id for.
  const bool is_external_input = (value.flags & FlagExternalInput()) != 0;
  uint32_t id = is_external ? value.id : kInferredValueId;

  // A tensor that is both an external input and an external output is an
  // identity passthrough: no node produces it, so its value is byte for byte
  // the data the caller wrote. YNNPACK cannot express that, because
  // `ynn_runtime::build` counts such a value as both an input and an output
  // when it builds the slinky pipeline while `ynn_runtime::setup` counts it
  // only as an input, and the resulting output count mismatch trips an
  // assertion inside `slinky::pipeline::setup`.
  //
  // Declaring the value as an input only sidesteps this. `NnpackValue::flags`
  // keeps both bits, so the runner still treats the tensor as an output: it
  // shares one host buffer between the input and the output, and
  // `ynn_get_external_value_shape` reports the shape from the input's buffer
  // descriptor. Reading the output therefore yields the passthrough data.
  //
  // TODO: Remove once YNNPACK supports values that are both an
  // external input and an external output.
  uint32_t ynn_flags = value.flags;
  if (is_external_input) {
    ynn_flags &= ~FlagExternalOutput();
  }

  // An extent declared here is static for the lifetime of the subgraph, so
  // external inputs are declared with dynamic extents (0) to let the runner
  // reshape them. Extent 1 is the exception: YNNPACK only treats a dimension as
  // broadcastable when it is declared with extent 1, and a dynamic extent that
  // happens to be 1 at runtime does not broadcast. Graphs rely on this for
  // things like an attention mask of shape [1, 1, seq_q, seq_k] that has to
  // broadcast over the head dimension, so extent 1 dimensions stay static. The
  // extents of every other value are inferred from the nodes producing them.
  std::vector<size_t> external_input_dims;
  if (is_external_input) {
    external_input_dims.reserve(dims.size());
    for (size_t dim : dims) {
      external_input_dims.push_back(dim == 1 ? 1 : 0);
    }
    dims_ptr =
        external_input_dims.empty() ? nullptr : external_input_dims.data();
  }

  if (info.quantization == nullptr) {
    const ynn_type type = ToYnnType(info.type);
    if (type == ynn_type_invalid) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": unsupported tensor type ",
                       static_cast<int>(info.type)));
    }
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(
        sg, type, dims.size(), dims_ptr, data_ptr, ynn_flags, &id))
        << "Could not define a new tensor value.";
    value.id = id;
    return absl::OkStatus();
  }

  if (is_external) {
    return absl::UnimplementedError(
        absl::StrCat(info.name,
                     ": quantized external values are not supported by the "
                     "YNNPACK backend."));
  }

  // YNNPACK doesn't carry quantization parameters on values. Instead the packed
  // integer weights are defined as a plain tensor and an explicit dequantize
  // node produces the float value the rest of the graph consumes. YNNPACK's
  // fusion passes fold that node back into its consumer (e.g. a dot) so this
  // stays a zero-copy representation.
  const ynn_type storage_type = GetQuantizedStorageType(info);
  if (storage_type == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": unsupported quantized storage type."));
  }

  uint32_t quantized_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(sg, storage_type, dims.size(),
                                               dims_ptr, data_ptr,
                                               /*flags=*/0, &quantized_id))
      << "Could not define the packed weights of a quantized tensor.";

  if (absl::StatusOr<const graph::BlockwiseQuantization&> bwq =
          info.quantization->As<const graph::BlockwiseQuantization>();
      bwq.ok()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        value.id, DefineBlockwiseDequantize(info, dims, *bwq, quantized_id));
    return absl::OkStatus();
  }
  if (absl::StatusOr<const graph::PerChannelAffineQuantization&> pcq =
          info.quantization->As<const graph::PerChannelAffineQuantization>();
      pcq.ok()) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        value.id, DefinePerChannelDequantize(info, dims, *pcq, quantized_id));
    return absl::OkStatus();
  }
  return absl::UnimplementedError(
      absl::StrCat(info.name, ": unsupported quantization type."));
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineQuantizationScales(
    absl::Span<const float> scales, absl::Span<const size_t> dims) {
  const void* data = KeepAlive(*graph_, scales);
  uint32_t scale_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(subgraph(), ynn_type_fp32,
                                               dims.size(), dims.data(), data,
                                               /*flags=*/0, &scale_id))
      << "Could not define quantization scales.";
  return scale_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineQuantizationZeroPoints(
    absl::Span<const int64_t> zero_points, absl::Span<const size_t> dims) {
  // A symmetric quantization scheme doesn't need a zero point node at all, and
  // YNNPACK has a faster path for that case.
  const bool all_zeros = std::all_of(zero_points.begin(), zero_points.end(),
                                     [](int64_t zp) { return zp == 0; });
  if (all_zeros) {
    return static_cast<uint32_t>(YNN_INVALID_VALUE_ID);
  }

  const std::vector<int32_t> narrowed(zero_points.begin(), zero_points.end());
  const void* data = KeepAlive(*graph_, absl::MakeConstSpan(narrowed));
  uint32_t zero_point_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(subgraph(), ynn_type_int32,
                                               dims.size(), dims.data(), data,
                                               /*flags=*/0, &zero_point_id))
      << "Could not define quantization zero points.";
  return zero_point_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefinePerChannelDequantize(
    const graph::TensorInformation& info, absl::Span<const size_t> dims,
    const graph::PerChannelAffineQuantization& quantization,
    uint32_t quantized_id) {
  const int quantized_dimension = quantization.quantized_dimension;
  if (quantized_dimension < 0 ||
      static_cast<size_t>(quantized_dimension) >= dims.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": quantized_dimension is out of range."));
  }
  if (quantization.scales.size() != dims[quantized_dimension]) {
    return absl::InvalidArgumentError(absl::StrCat(
        info.name, ": per-channel scale count (", quantization.scales.size(),
        ") doesn't match the quantized dimension size (",
        dims[quantized_dimension], ")."));
  }

  // The scales apply along `quantized_dimension`; every other dimension is
  // declared with extent 1, which makes it a broadcast dimension, so the scale
  // tensor is elementwise-compatible with the weights.
  std::vector<size_t> param_dims(dims.size(), 1);
  param_dims[quantized_dimension] = quantization.scales.size();

  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t scale_id,
      DefineQuantizationScales(absl::MakeConstSpan(quantization.scales),
                               param_dims));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t zero_point_id,
      DefineQuantizationZeroPoints(
          absl::MakeConstSpan(quantization.zero_points), param_dims));

  uint32_t dequantized_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_dequantize(subgraph(), quantized_id, zero_point_id, scale_id,
                            ynn_type_fp32, &dequantized_id, /*flags=*/0))
      << "Could not define the dequantize node of a quantized tensor.";
  return dequantized_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineBlockwiseDequantize(
    const graph::TensorInformation& info, absl::Span<const size_t> dims,
    const graph::BlockwiseQuantization& quantization, uint32_t quantized_id) {
  ynn_subgraph_t sg = subgraph();
  if (quantization.block_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": block_size must be > 0."));
  }
  if (dims.size() != 2) {
    return absl::UnimplementedError(absl::StrCat(
        info.name, ": blockwise quantization requires a rank 2 tensor."));
  }
  const size_t block_size = static_cast<size_t>(quantization.block_size);
  if (dims[1] % block_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": the reduction dimension (", dims[1],
                     ") must be a multiple of block_size (", block_size, ")."));
  }
  const size_t num_blocks = dims[1] / block_size;
  if (quantization.scales.size() != dims[0] * num_blocks) {
    return absl::InvalidArgumentError(absl::StrCat(
        info.name, ": blockwise scale count (", quantization.scales.size(),
        ") doesn't match the expected block count (", dims[0] * num_blocks,
        ")."));
  }

  // There is one scale per block, so the scales can't broadcast against the
  // [channels, reduction] weights as they are. The weights can't be reshaped
  // into blocks either: for a sub-byte type such as int4 the reduction
  // dimension is the packed one, and splitting it reinterprets the packing.
  // Instead the parameters are grown to the shape of the weights, which is the
  // scheme `ynn_define_dequantize` documents for blockwise quantization: a
  // [channels, num_blocks, 1] tensor is broadcast by the block size and then
  // reshaped back to [channels, reduction].
  const std::vector<size_t> param_dims = {dims[0], num_blocks, 1};
  LRT_TENSOR_ASSIGN_OR_RETURN(
      uint32_t scale_id,
      DefineQuantizationScales(absl::MakeConstSpan(quantization.scales),
                               param_dims));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      scale_id, ExpandBlockwiseParam(scale_id, block_size, info.name));

  LRT_TENSOR_ASSIGN_OR_RETURN(
      uint32_t zero_point_id,
      DefineQuantizationZeroPoints(
          absl::MakeConstSpan(quantization.zero_points), param_dims));
  if (zero_point_id != YNN_INVALID_VALUE_ID) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        zero_point_id,
        ExpandBlockwiseParam(zero_point_id, block_size, info.name));
  }

  uint32_t dequantized_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_dequantize(sg, quantized_id, zero_point_id, scale_id,
                            ynn_type_fp32, &dequantized_id, /*flags=*/0))
      << "Could not define the dequantize node of a blockwise quantized "
         "tensor.";
  return dequantized_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::ExpandBlockwiseParam(
    uint32_t param_id, size_t block_size, absl::string_view name) {
  ynn_subgraph_t sg = subgraph();

  // [channels, num_blocks, 1] -> [channels, num_blocks, block_size]. A zero in
  // `new_dims` leaves that dimension alone.
  const std::vector<size_t> broadcast_dims = {0, 0, block_size};
  uint32_t broadcast_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_broadcast(
      sg, broadcast_dims.size(), broadcast_dims.data(), param_id, &broadcast_id,
      /*flags=*/0))
      << name << ": could not broadcast a blockwise quantization parameter.";

  // [channels, num_blocks, block_size] -> [channels, reduction].
  uint32_t fused_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_fuse_dim(sg, /*axis=*/1,
                                                 /*axes_count=*/2, broadcast_id,
                                                 &fused_id,
                                                 /*flags=*/0))
      << name << ": could not reshape a blockwise quantization parameter.";
  return fused_id;
}

absl::Status YnnpackBuildContext::DefineConstantTensor(
    Type datatype, absl::Span<const size_t> shape, const void* data,
    uint32_t* id) {
  const ynn_type type = ToYnnType(datatype);
  if (type == ynn_type_invalid) {
    return absl::InvalidArgumentError("Unsupported constant datatype");
  }
  *id = kInferredValueId;
  return YnnStatusToAbsl(
      ynn_define_tensor(subgraph(), type, shape.size(),
                        shape.empty() ? nullptr : shape.data(), data,
                        /*flags=*/0, id),
      "ynn_define_tensor");
}

absl::Status YnnpackBuildContext::LowerOp(const graph::Operation& op) {
  const YnnpackOperation* op_ext = op.GetExtension<YnnpackOperation>();
  if (op_ext == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Operation ", op.GetName(), " does not implement YNNPACK operation."));
  }
  return op_ext->ToYnnpack(op, *this);
}

absl::StatusOr<std::unique_ptr<YnnpackGraph>> BuildYnnpackGraph(
    std::vector<TensorHandle> outputs) {
  uint32_t next_id = 0;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  for (const TensorHandle& out : outputs) {
    auto [it, inserted] = external_ids.insert({out.GetRaw(), next_id});
    next_id += inserted;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<const graph::Operation*> plan,
                              GetExecutionPlan(outputs));
  for (const graph::Operation* op : plan) {
    for (const graph::Tensor& t : op->inputs) {
      if (absl::StatusOr<graph::TensorInformation> info_or = graph::GetInfo(t);
          !info_or.ok() || info_or->buffer != nullptr) {
        continue;
      }
      if (absl::StatusOr<std::shared_ptr<graph::Operation>> producer_or =
              graph::GetProducer(t);
          producer_or.ok() && *producer_or != nullptr) {
        continue;
      }
      auto [it, inserted] = external_ids.insert({t, next_id});
      next_id += inserted;
    }
  }

  YnnpackBuildContext ctx(std::move(outputs), std::move(external_ids));
  LRT_TENSOR_RETURN_IF_ERROR(BuildNnpackGraph(ctx));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::unique_ptr<NnpackGraph> graph,
                              ctx.Finalize());
  return std::unique_ptr<YnnpackGraph>(
      static_cast<YnnpackGraph*>(graph.release()));
}

}  // namespace litert::tensor
