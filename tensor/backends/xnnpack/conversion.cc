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

#include <array>
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
#include "tensor/backends/nnpack_common/conversion.h"
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/runners/nnpack_common/runner.h"
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

absl::Status XnnpackTraits::EnsureInitialized() {
  absl::call_once(g_xnn_init_once, []() {
    *g_xnn_init_status =
        XnnStatusToAbsl(xnn_initialize(nullptr), "xnn_initialize");
  });
  return *g_xnn_init_status;
}

absl::Status XnnpackTraits::CreateSubgraph(size_t external_value_ids,
                                           uint32_t flags,
                                           SubgraphType* subgraph) {
  return XnnStatusToAbsl(
      xnn_create_subgraph(external_value_ids, flags, subgraph),
      "xnn_create_subgraph");
}

void XnnpackTraits::DeleteSubgraph(SubgraphType subgraph) {
  if (subgraph != nullptr) {
    xnn_delete_subgraph(subgraph);
  }
}

absl::Status XnnpackTraits::DefineTensorValue(
    NnpackBuildContext<XnnpackTraits>& ctx, const graph::Tensor& tensor,
    ValueType& value) {
  const auto& info = value.info;
  const bool is_external =
      (value.flags & (kFlagExternalInput | kFlagExternalOutput)) != 0;

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
    ctx.keep_alive_buffers().push_back(info.buffer);
  }

  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const void* data_ptr = value.data.data();
  uint32_t external_id = is_external ? value.id : XNN_INVALID_VALUE_ID;

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
        ctx.subgraph(), GetXnnpackType(value), dims.size(),
        dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
        value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else if (auto maybe_pcq =
                 info.quantization->As<PerChannelAffineQuantization>();
             maybe_pcq.ok()) {
    const auto& pcq = maybe_pcq.value();
    if (pcq.scales.size() == 1) {
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
          ctx.subgraph(), GetXnnpackType(value),
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
        ctx.dequantized_buffers().push_back(std::move(f32_data));
        data_ptr = ctx.dequantized_buffers().back().data();
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
            ctx.subgraph(), xnn_datatype_fp32, dims.size(),
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
                ctx.subgraph(), GetXnnpackType(value), /*zero_point=*/0,
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
    const size_t expected_block_count = dims[0] * dims[1] / bwq.block_size;
    if (bwq.scales.size() < expected_block_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise scale count (", bwq.scales.size(),
          ") is smaller than block_count (", expected_block_count, ")"));
    }
    ctx.fp16_buffers().emplace_back(bwq.scales.begin(), bwq.scales.end());
    const void* scale_ptr = ctx.fp16_buffers().back().data();
    int32_t zero_point = bwq.zero_points.empty() ? 0 : bwq.zero_points[0];
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_blockwise_quantized_tensor_value_v2(
        ctx.subgraph(), GetXnnpackType(value), zero_point, scale_ptr,
        dims.size(), bwq.quantized_dimension, bwq.block_size,
        dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
        value.flags, xnn_datatype_fp16, &value.id))
        << "Could not define a new blockwise quantized tensor value.";
  } else {
    return absl::UnimplementedError("Unsupported quantization type.");
  }
  return absl::OkStatus();
}

absl::Status XnnpackTraits::DefineConstantTensor(SubgraphType subgraph,
                                                 ::xnn_datatype datatype,
                                                 absl::Span<const size_t> shape,
                                                 const void* data,
                                                 uint32_t* id) {
  return XnnStatusToAbsl(
      xnn_define_tensor_value(subgraph, datatype, shape.size(),
                              shape.empty() ? nullptr : shape.data(), data,
                              XNN_INVALID_VALUE_ID, /*flags=*/0, id),
      "xnn_define_tensor_value");
}

absl::Status XnnpackTraits::LowerOp(const XnnpackOperation& ext,
                                    const graph::Operation& op,
                                    NnpackBuildContext<XnnpackTraits>& ctx) {
  return ext.ToXnnpack(op, ctx);
}

absl::Status XnnpackTraits::SetExternalValueShape(
    RuntimeType* runtime, uint32_t id, absl::Span<const size_t> dims) {
  return XnnStatusToAbsl(
      xnn_reshape_external_value(runtime, id, dims.size(),
                                 dims.empty() ? nullptr : dims.data()),
      "xnn_reshape_external_value");
}

absl::Status XnnpackTraits::ReshapeRuntime(RuntimeType* runtime) {
  return XnnStatusToAbsl(xnn_reshape_runtime(runtime), "xnn_reshape_runtime");
}

absl::Status XnnpackTraits::GetExternalValueShape(RuntimeType* runtime,
                                                  uint32_t id,
                                                  std::vector<size_t>& dims) {
  size_t num_dims = 0;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> shape_arr{};
  LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
      xnn_get_external_value_shape(runtime, id, &num_dims, shape_arr.data()),
      "xnn_get_external_value_shape"));
  dims.assign(shape_arr.begin(), shape_arr.begin() + num_dims);
  return absl::OkStatus();
}

absl::Status XnnpackTraits::SetupExternalValues(
    RuntimeType* runtime, absl::Span<ValueType> values,
    absl::flat_hash_map<uint32_t, ExternalBuffer>& external_buffers) {
  std::vector<xnn_external_value> externals;
  externals.reserve(values.size());
  for (ValueType& value : values) {
    if (value.flags == 0) {
      continue;
    }
    auto buffer = external_buffers[value.id].data();
    if (buffer.data() == nullptr) {
      return absl::FailedPreconditionError(
          "External value missing host buffer");
    }
    externals.push_back({.id = value.id, .data = buffer.data()});
  }
  return XnnStatusToAbsl(
      xnn_setup_runtime_v2(runtime, externals.size(), externals.data()),
      "xnn_setup_runtime_v2");
}

absl::Status XnnpackTraits::InvokeRuntime(RuntimeType* runtime) {
  return XnnStatusToAbsl(xnn_invoke_runtime(runtime), "xnn_invoke_runtime");
}

}  // namespace litert::tensor
