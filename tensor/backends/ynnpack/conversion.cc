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

// Returns the zero points of `quantization`, as stored in it.
absl::Span<const int64_t> GetRawZeroPoints(const Quantization& quantization) {
  if (absl::StatusOr<const BlockwiseQuantization&> bwq =
          quantization.As<const BlockwiseQuantization>();
      bwq.ok()) {
    return bwq->zero_points;
  }
  if (absl::StatusOr<const PerChannelAffineQuantization&> pcq =
          quantization.As<const PerChannelAffineQuantization>();
      pcq.ok()) {
    return pcq->zero_points;
  }
  return {};
}

// Quantization parameters may omit the zero points (symmetric quantization) or
// give a single one for all the scales. Otherwise there must be one per scale.
absl::Status CheckZeroPointCount(size_t num_zero_points, size_t num_scales,
                                 absl::string_view name) {
  if (num_zero_points > 1 && num_zero_points != num_scales) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, ": zero point count (", num_zero_points,
                     ") doesn't match the scale count (", num_scales, ")."));
  }
  return absl::OkStatus();
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

ynn_type YnnpackBuildContext::GetQuantizedStorageType(
    const graph::TensorInformation& info) {
  return ToYnnType(info.type);
}

size_t YnnpackBuildContext::GetBlockSize(const Quantization& quantization) {
  absl::StatusOr<const BlockwiseQuantization&> bwq =
      quantization.As<const BlockwiseQuantization>();
  return bwq.ok() ? static_cast<size_t>(bwq->block_size) : 0;
}

int YnnpackBuildContext::GetQuantizedDimension(
    const Quantization& quantization) {
  if (absl::StatusOr<const BlockwiseQuantization&> bwq =
          quantization.As<const BlockwiseQuantization>();
      bwq.ok()) {
    return bwq->quantized_dimension;
  }
  if (absl::StatusOr<const PerChannelAffineQuantization&> pcq =
          quantization.As<const PerChannelAffineQuantization>();
      pcq.ok() && pcq->scales.size() != 1) {
    return pcq->quantized_dimension;
  }
  return -1;
}

absl::Span<const float> YnnpackBuildContext::GetScales(
    const Quantization& quantization) {
  if (absl::StatusOr<const BlockwiseQuantization&> bwq =
          quantization.As<const BlockwiseQuantization>();
      bwq.ok()) {
    return bwq->scales;
  }
  if (absl::StatusOr<const PerChannelAffineQuantization&> pcq =
          quantization.As<const PerChannelAffineQuantization>();
      pcq.ok()) {
    return pcq->scales;
  }
  return {};
}

std::vector<int32_t> YnnpackBuildContext::GetZeroPoints(
    const Quantization& quantization) {
  const size_t num_scales = GetScales(quantization).size();
  const absl::Span<const int64_t> zero_points = GetRawZeroPoints(quantization);
  if (zero_points.empty()) {
    return std::vector<int32_t>(num_scales, 0);
  }
  if (zero_points.size() == 1) {
    return std::vector<int32_t>(num_scales,
                                static_cast<int32_t>(zero_points[0]));
  }
  return std::vector<int32_t>(zero_points.begin(), zero_points.end());
}

bool YnnpackBuildContext::IsSymmetric(const Quantization& quantization) {
  const absl::Span<const int64_t> zero_points = GetRawZeroPoints(quantization);
  return std::all_of(zero_points.begin(), zero_points.end(),
                     [](int64_t zp) { return zp == 0; });
}

size_t YnnpackBuildContext::ElementsPerByte(ynn_type type) {
  switch (type) {
    case ynn_type_int2:
    case ynn_type_uint2:
      return 4;
    case ynn_type_int4:
    case ynn_type_uint4:
      return 2;
    case ynn_type_int8:
    case ynn_type_uint8:
      return 1;
    default:
      return 0;
  }
}

absl::Status YnnpackBuildContext::ValidateQuantization(
    const graph::TensorInformation& info) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": the tensor isn't quantized."));
  }
  if (GetQuantizedStorageType(info) == ynn_type_invalid) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": unsupported quantized storage type."));
  }
  const graph::Shape& dims = info.shape;

  if (absl::StatusOr<const BlockwiseQuantization&> bwq =
          info.quantization->As<const BlockwiseQuantization>();
      bwq.ok()) {
    if (bwq->block_size <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": block_size must be > 0."));
    }
    if (dims.size() != 2) {
      return absl::UnimplementedError(absl::StrCat(
          info.name, ": blockwise quantization requires a rank 2 tensor."));
    }
    const int quantized_dimension = bwq->quantized_dimension;
    if (quantized_dimension != 0 && quantized_dimension != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": quantized_dimension is out of range."));
    }
    const size_t block_size = static_cast<size_t>(bwq->block_size);
    const size_t channels = dims[quantized_dimension];
    const size_t reduction = dims[1 - quantized_dimension];
    if (reduction % block_size != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": the blocked dimension (", reduction,
          ") must be a multiple of block_size (", block_size, ")."));
    }
    const size_t num_blocks = reduction / block_size;
    if (bwq->scales.size() != channels * num_blocks) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise scale count (", bwq->scales.size(),
          ") doesn't match the expected block count (", channels * num_blocks,
          ")."));
    }
    return CheckZeroPointCount(bwq->zero_points.size(), bwq->scales.size(),
                               info.name);
  }

  if (absl::StatusOr<const PerChannelAffineQuantization&> pcq =
          info.quantization->As<const PerChannelAffineQuantization>();
      pcq.ok()) {
    // A single scale is per-tensor quantization, which applies to any rank.
    if (pcq->scales.size() != 1) {
      const int quantized_dimension = pcq->quantized_dimension;
      if (quantized_dimension < 0 ||
          static_cast<size_t>(quantized_dimension) >= dims.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat(info.name, ": quantized_dimension is out of range."));
      }
      if (pcq->scales.size() !=
          static_cast<size_t>(dims[quantized_dimension])) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name, ": per-channel scale count (", pcq->scales.size(),
            ") doesn't match the quantized dimension size (",
            dims[quantized_dimension], ")."));
      }
    }
    return CheckZeroPointCount(pcq->zero_points.size(), pcq->scales.size(),
                               info.name);
  }

  return absl::UnimplementedError(
      absl::StrCat(info.name, ": unsupported quantization type."));
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineDequantizedValue(
    const graph::Tensor& tensor, uint32_t data_id, uint32_t fp32_id) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  LRT_TENSOR_RETURN_IF_ERROR(ValidateQuantization(info));
  if (GetBlockSize(*info.quantization) > 0) {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        fp32_id, DefineBlockwiseDequantize(info, data_id, fp32_id));
  } else {
    LRT_TENSOR_ASSIGN_OR_RETURN(
        fp32_id, DefinePerChannelDequantize(info, data_id, fp32_id));
  }
  quantized_data_ids_[tensor] = data_id;
  return fp32_id;
}

absl::Status YnnpackBuildContext::DefineTensorValue(const graph::Tensor& tensor,
                                                    NnpackValue& value) {
  const graph::TensorInformation& info = value.info;
  const bool is_external =
      (value.flags & (FlagExternalInput() | FlagExternalOutput())) != 0;

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
    graph_->keep_alive_buffers().push_back(info.buffer);
    if (const size_t expected_bytes = BufferSize(info.type, info.GetSize());
        value.data.size() < expected_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": the buffer holds ", value.data.size(),
                       " bytes, but the tensor needs ", expected_bytes, "."));
    }
  }

  const std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const size_t* dims_ptr = dims.empty() ? nullptr : dims.data();
  const void* data_ptr = value.data.data();
  ynn_subgraph_t sg = subgraph();

  // Only external values may reuse a pre-assigned id; everything else is an
  // internal value that YNNPACK allocates an id for.
  const bool is_external_input = (value.flags & FlagExternalInput()) != 0;
  uint32_t id = is_external ? value.id : kInferredValueId;

  // YNNPACK doesn't support tensors that are both an external input and an
  // external output. We declare the value as an input only to work around this.
  // The runner still treats the tensor as an output: the same host buffer is
  // shared between the input and the output and `ynn_get_external_value_shape`
  // reports the shape from the input's buffer descriptor. Reading the output
  // therefore yields the passthrough data.
  uint32_t ynn_flags = value.flags;
  if (is_external_input) {
    ynn_flags &= ~FlagExternalOutput();
  }

  // An extent declared here is static for the lifetime of the subgraph, so
  // every dimension of an external input is declared dynamic (0) to let the
  // runner reshape it. Operations that broadcast one of those dimensions mark
  // it as a broadcast dimension explicitly (see `DefineBroadcastOperand` in
  // arithmetic.cc). The extents of every other value are inferred from the
  // nodes producing them.
  const std::vector<size_t> external_input_dims(dims.size(), 0);
  if (is_external_input) {
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

  // YNNPACK doesn't carry quantization parameters on values, so the backend
  // works on the fp32 value of quantized tensors, and keeps the id of their
  // integer data on the side for the lowerings that can use it (see
  // `FindQuantizedDataId`).
  //
  // A quantized tensor that an op produces (e.g. `Transpose` propagates the
  // quantization of its input) is defined as a plain fp32 value that its
  // producer writes. If the producer can move the integers instead, it records
  // the integer data itself with `DefineDequantizedValue`.
  if (info.buffer == nullptr) {
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(sg, ynn_type_fp32, dims.size(),
                                                 dims_ptr, /*data=*/nullptr,
                                                 /*flags=*/0, &id))
        << "Could not define the value of a quantized tensor.";
    value.id = id;
    return absl::OkStatus();
  }

  // A quantized constant is defined as its integer data plus a dequantize
  // producing the fp32 value. The dot lowerings consume the data directly, in
  // which case YNNPACK removes the unused dequantize. Other consumers read the
  // fp32 value: YNNPACK doesn't constant fold the dequantize because its output
  // is larger than its input, so it runs on every invoke.
  LRT_TENSOR_RETURN_IF_ERROR(ValidateQuantization(info));
  uint32_t data_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(
      sg, GetQuantizedStorageType(info), dims.size(), dims_ptr, data_ptr,
      /*flags=*/0, &data_id))
      << "Could not define the integer data of a quantized tensor.";
  LRT_TENSOR_ASSIGN_OR_RETURN(value.id,
                              DefineDequantizedValue(tensor, data_id));
  return absl::OkStatus();
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineQuantizationScales(
    absl::Span<const float> scales, absl::Span<const size_t> dims) {
  const void* data = KeepAlive(*graph_, scales);
  uint32_t scale_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(
      subgraph(), ynn_type_fp32, dims.size(),
      dims.empty() ? nullptr : dims.data(), data, /*flags=*/0, &scale_id))
      << "Could not define quantization scales.";
  return scale_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineQuantizationZeroPoints(
    absl::Span<const int32_t> zero_points, absl::Span<const size_t> dims) {
  // A symmetric quantization scheme doesn't need a zero point node at all, and
  // YNNPACK has a faster path for that case.
  const bool all_zeros = std::all_of(zero_points.begin(), zero_points.end(),
                                     [](int32_t zp) { return zp == 0; });
  if (all_zeros) {
    return static_cast<uint32_t>(YNN_INVALID_VALUE_ID);
  }

  const void* data = KeepAlive(*graph_, zero_points);
  uint32_t zero_point_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_tensor(
      subgraph(), ynn_type_int32, dims.size(),
      dims.empty() ? nullptr : dims.data(), data, /*flags=*/0, &zero_point_id))
      << "Could not define quantization zero points.";
  return zero_point_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefinePerChannelDequantize(
    const graph::TensorInformation& info, uint32_t data_id, uint32_t fp32_id) {
  const Quantization& quantization = *info.quantization;
  const absl::Span<const float> scales = GetScales(quantization);
  const std::vector<int32_t> zero_points = GetZeroPoints(quantization);

  // The scales apply along the quantized dimension; every other dimension is
  // declared with extent 1, which makes it a broadcast dimension, so the scale
  // tensor is elementwise-compatible with the weights. A per-tensor scale is a
  // scalar, which broadcasts in every dimension.
  std::vector<size_t> param_dims;
  if (const int quantized_dimension = GetQuantizedDimension(quantization);
      quantized_dimension >= 0) {
    param_dims.assign(info.shape.size(), 1);
    param_dims[quantized_dimension] = scales.size();
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const uint32_t scale_id,
                              DefineQuantizationScales(scales, param_dims));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const uint32_t zero_point_id,
      DefineQuantizationZeroPoints(absl::MakeConstSpan(zero_points),
                                   param_dims));

  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_dequantize(subgraph(), data_id, zero_point_id, scale_id,
                            ynn_type_fp32, &fp32_id, /*flags=*/0))
      << info.name
      << ": could not define the dequantize node of a quantized tensor.";
  return fp32_id;
}

absl::StatusOr<uint32_t> YnnpackBuildContext::DefineBlockwiseDequantize(
    const graph::TensorInformation& info, uint32_t data_id, uint32_t fp32_id) {
  const Quantization& quantization = *info.quantization;
  const int quantized_dimension = GetQuantizedDimension(quantization);
  const size_t block_size = GetBlockSize(quantization);
  const size_t channels = info.shape[quantized_dimension];
  const size_t num_blocks = info.shape[1 - quantized_dimension] / block_size;
  const absl::Span<const float> scales = GetScales(quantization);
  const std::vector<int32_t> zero_points = GetZeroPoints(quantization);
  const absl::string_view name = info.name;
  ynn_subgraph_t sg = subgraph();

  // To dequantize using blocks, we split the quantization axis using the block
  // size, this allows `ynn_define_dequantize` to broadcast the scales/zero
  // points along the created axis. We then fuse back the output tensor to the
  // orginal shape.
  //
  // Example:
  // ```
  // block size of 4, quantized axis = 1
  // data [2, 8]         scales [2, 2]
  //       ┌──┤ split               │ extend last dimension to broadcast
  //   [2, 2, 4]                [2, 2, 1]
  //       └────── dequantize ──────┘
  //                    │
  //                [2, 2, 4]
  //                    └──┤ fuse
  //                   [2, 8]
  // ```
  //
  // The parameters are stored as [channels, num_blocks]. When the weights are
  // [reduction, channels], the blocks run along dimension 0, so the data is
  // split into [num_blocks, block_size, channels], and the parameters are
  // transposed to [num_blocks, 1, channels].
  const int32_t blocked_axis = quantized_dimension == 0 ? 1 : 0;
  // The extent 1 dimension of the parameters is a broadcast dimension.
  const size_t param_dims[] = {channels, num_blocks, 1};
  // Moves the channels of [channels, num_blocks, 1] parameters last.
  auto to_param_layout = [&](uint32_t param_id) -> absl::StatusOr<uint32_t> {
    if (quantized_dimension == 0 || param_id == YNN_INVALID_VALUE_ID) {
      return param_id;
    }
    const int32_t axes[] = {1, 2, 0};
    uint32_t transposed_id = kInferredValueId;
    LRT_TENSOR_RETURN_IF_ERROR(ynn_define_static_transpose(
        sg, /*num_axes=*/3, axes, param_id, &transposed_id, /*flags=*/0))
        << name << ": could not transpose a blockwise quantization parameter.";
    return transposed_id;
  };

  // Splitting the innermost dimension of sub-byte data requires the blocks to
  // fill whole bytes.
  const size_t elements_per_byte =
      ElementsPerByte(GetQuantizedStorageType(info));
  if (quantized_dimension == 0 && elements_per_byte > 1 &&
      block_size % elements_per_byte != 0) {
    return absl::UnimplementedError(absl::StrCat(
        name, ": the blocks of sub-byte weights must fill whole bytes."));
  }
  const size_t splits[] = {num_blocks, block_size};
  uint32_t split_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_split_dim(sg, blocked_axis,
                                                  /*num_splits=*/2, splits,
                                                  data_id, &split_id,
                                                  /*flags=*/0))
      << name << ": could not split the blocks of blockwise quantized data.";

  LRT_TENSOR_ASSIGN_OR_RETURN(uint32_t scale_id,
                              DefineQuantizationScales(scales, param_dims));
  LRT_TENSOR_ASSIGN_OR_RETURN(scale_id, to_param_layout(scale_id));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      uint32_t zero_point_id,
      DefineQuantizationZeroPoints(absl::MakeConstSpan(zero_points),
                                   param_dims));
  LRT_TENSOR_ASSIGN_OR_RETURN(zero_point_id, to_param_layout(zero_point_id));

  uint32_t dequantized_id = kInferredValueId;
  LRT_TENSOR_RETURN_IF_ERROR(
      ynn_define_dequantize(sg, split_id, zero_point_id, scale_id,
                            ynn_type_fp32, &dequantized_id, /*flags=*/0))
      << name
      << ": could not define the dequantize node of a blockwise quantized "
         "tensor.";

  LRT_TENSOR_RETURN_IF_ERROR(ynn_define_fuse_dim(sg, blocked_axis,
                                                 /*axes_count=*/2,
                                                 dequantized_id, &fp32_id,
                                                 /*flags=*/0))
      << name << ": could not fuse the blocks of dequantized values.";
  return fp32_id;
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
