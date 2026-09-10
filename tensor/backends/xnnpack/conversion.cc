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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/backends/common_nnpack/conversion.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

absl::once_flag g_xnn_init_once;
absl::NoDestructor<absl::Status> g_xnn_init_status(absl::OkStatus());

xnn_datatype GetXnnpackType(const NnpackValue& value) {
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

}  // namespace

absl::Status XnnpackBuildContext::EnsureInitialized() {
  absl::call_once(g_xnn_init_once, []() {
    *g_xnn_init_status =
        XnnStatusToAbsl(xnn_initialize(nullptr), "xnn_initialize");
  });
  return *g_xnn_init_status;
}

std::unique_ptr<NnpackGraph> XnnpackBuildContext::CreateEmptyGraph() {
  return std::make_unique<XnnpackGraph>();
}

absl::Status XnnpackBuildContext::CreateSubgraph(size_t external_value_ids,
                                                 uint32_t flags) {
  LRT_TENSOR_RETURN_IF_ERROR(static_cast<XnnpackGraph*>(graph_.get())
                                 ->ResetSubgraph(external_value_ids, flags));
  return absl::OkStatus();
}

absl::Status XnnpackBuildContext::DefineTensorValue(const graph::Tensor& tensor,
                                                    NnpackValue& value) {
  const auto& info = value.info;
  const bool is_external =
      (value.flags & (FlagExternalInput() | FlagExternalOutput())) != 0;

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
    graph_->keep_alive_buffers().push_back(info.buffer);
  }

  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const void* data_ptr = value.data.data();
  uint32_t external_id = is_external ? value.id : XNN_INVALID_VALUE_ID;
  xnn_subgraph_t sg = subgraph();

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_define_tensor_value(sg, GetXnnpackType(value), dims.size(),
                                dims.empty() ? nullptr : dims.data(), data_ptr,
                                external_id, value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else if (auto maybe_pcq =
                 info.quantization->As<PerChannelAffineQuantization>();
             maybe_pcq.ok()) {
    const auto& pcq = maybe_pcq.value();
    if (pcq.scales.size() == 1) {
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
          sg, GetXnnpackType(value),
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
            DequantizeInt8ConstantTensor(
                info, absl::MakeSpan(value.data.data(), value.data.size())));
        graph_->dequantized_buffers().push_back(std::move(f32_data));
        data_ptr = graph_->dequantized_buffers().back().data();
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
            sg, xnn_datatype_fp32, dims.size(),
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
                sg, GetXnnpackType(value), /*zero_point=*/0, pcq.scales.data(),
                dims.size(), pcq.quantized_dimension,
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
    const size_t expected_block_count = dims[0] * dims[1] / bwq.block_size;
    if (bwq.scales.size() < expected_block_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise scale count (", bwq.scales.size(),
          ") is smaller than block_count (", expected_block_count, ")"));
    }
    graph_->fp16_buffers().emplace_back(bwq.scales.begin(), bwq.scales.end());
    const void* scale_ptr = graph_->fp16_buffers().back().data();
    int32_t zero_point = bwq.zero_points.empty() ? 0 : bwq.zero_points[0];
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_blockwise_quantized_tensor_value_v2(
        sg, GetXnnpackType(value), zero_point, scale_ptr, dims.size(),
        bwq.quantized_dimension, bwq.block_size,
        dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
        value.flags, xnn_datatype_fp16, &value.id))
        << "Could not define a new blockwise quantized tensor value.";
  } else {
    return absl::UnimplementedError("Unsupported quantization type.");
  }
  return absl::OkStatus();
}

absl::Status XnnpackBuildContext::DefineConstantTensor(
    Type datatype, absl::Span<const size_t> shape, const void* data,
    uint32_t* id) {
  xnn_datatype xnn_type = xnn_datatype_invalid;
  switch (datatype) {
    case Type::kFP32:
      xnn_type = xnn_datatype_fp32;
      break;
    case Type::kFP16:
      xnn_type = xnn_datatype_fp16;
      break;
    case Type::kI32:
      xnn_type = xnn_datatype_int32;
      break;
    case Type::kI8:
      xnn_type = xnn_datatype_qint8;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported constant datatype");
  }

  return XnnStatusToAbsl(
      xnn_define_tensor_value(subgraph(), xnn_type, shape.size(),
                              shape.empty() ? nullptr : shape.data(), data,
                              XNN_INVALID_VALUE_ID, /*flags=*/0, id),
      "xnn_define_tensor_value");
}

absl::Status XnnpackBuildContext::LowerOp(const graph::Operation& op) {
  auto op_ext = op.GetExtension<XnnpackOperation>();
  if (op_ext == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Operation ", op.GetName(), " does not implement XNNPACK operation."));
  }
  return op_ext->ToXnnpack(op, *this);
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs) {
  uint32_t next_id = 0;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  for (const TensorHandle& out : outputs) {
    auto [it, inserted] = external_ids.insert({out.GetRaw(), next_id});
    next_id += inserted;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(outputs));
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

  XnnpackBuildContext ctx(std::move(outputs), std::move(external_ids));
  LRT_TENSOR_RETURN_IF_ERROR(BuildNnpackGraph(ctx));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto graph, ctx.Finalize());
  return std::unique_ptr<XnnpackGraph>(
      static_cast<XnnpackGraph*>(graph.release()));
}

}  // namespace litert::tensor
