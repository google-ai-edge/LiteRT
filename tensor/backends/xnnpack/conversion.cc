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
#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/arithmetic_graph.h"
#include "tensor/backends/common_nnpack/conversion.h"
#include "tensor/backends/common_nnpack/graph.h"
#include "tensor/backends/xnnpack/dequantize.h"
#include "tensor/backends/xnnpack/graph.h"
#include "tensor/backends/xnnpack/utils.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/graph_traversal.h"
#include "tensor/internal/tensor_hash.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

using ::litert::tensor::graph::hash_internal::ComputeConstantTensorHash;
using ::litert::tensor::graph::hash_internal::FnvMixSampledBytes;
using ::litert::tensor::graph::hash_internal::FnvMixScalar;
using ::litert::tensor::graph::hash_internal::kFnvOffsetBasis;

absl::once_flag g_xnn_init_once;
absl::NoDestructor<absl::Status> g_xnn_init_status(absl::OkStatus());

struct FingerprintSinkRegistry {
  absl::Mutex mu;
  absl::flat_hash_map<xnn_weights_cache_t, XnnConstantFingerprintSink> sinks
      ABSL_GUARDED_BY(mu);
};

FingerprintSinkRegistry& GetFingerprintSinkRegistry() {
  static absl::NoDestructor<FingerprintSinkRegistry> registry;
  return *registry;
}

uint64_t ComputeMinPackedBytesForXnnType(xnn_datatype dtype,
                                         absl::Span<const size_t> dims) {
  uint64_t num_elems = 1;
  for (size_t d : dims) {
    if (d == 0) return 0;
    num_elems *= static_cast<uint64_t>(d);
  }
  switch (dtype) {
    case xnn_datatype_qcint2:
    case xnn_datatype_qint2:
      return (num_elems + 3) / 4;
    case xnn_datatype_qcint4:
    case xnn_datatype_qbint4:
    case xnn_datatype_qint4:
      return (num_elems + 1) / 2;
    case xnn_datatype_qcint8:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      return num_elems;
    case xnn_datatype_fp16:
    case xnn_datatype_bf16:
    case xnn_datatype_pfp16:
      return num_elems * 2;
    case xnn_datatype_fp32:
    case xnn_datatype_int32:
    case xnn_datatype_pfp32:
      return num_elems * 4;
    default:
      return 0;
  }
}

xnn_datatype GetXnnpackType(const NnpackValue& value) {
  switch (value.info.type) {
    case Type::kI2:
      if (value.info.quantization &&
          value.info.quantization->As<PerChannelAffineQuantization>().ok()) {
        return xnn_datatype_qcint2;
      }
      break;
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
    case Type::kFP16:
      return xnn_datatype_fp16;
    case Type::kBF16:
      return xnn_datatype_bf16;
    case Type::kI32:
      return xnn_datatype_int32;
    case Type::kFP32:
      return xnn_datatype_fp32;
    case Type::kUnknown:
    case Type::kBOOL:
    case Type::kI16:
    case Type::kI64:
    case Type::kU2:
    case Type::kU4:
    case Type::kU8:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kFP64:
      break;
  }
  return xnn_datatype_invalid;
}

std::shared_ptr<const std::vector<bf16_t>> GetSharedWarmDummyBf16Scales(
    size_t min_blocks) {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<std::shared_ptr<const std::vector<bf16_t>>> cached;
  absl::MutexLock lock(*mu);
  if (*cached == nullptr || (*cached)->size() < min_blocks) {
    *cached =
        std::make_shared<const std::vector<bf16_t>>(min_blocks, bf16_t(1.0f));
  }
  return *cached;
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
  if (!warm_fc_buffers_.empty()) {
    if (warm_max_rows_ > 0) {
      graph_->dequantized_buffers().emplace_back(warm_max_rows_, 1.0f);
      warm_dummy_f32_scales_ = graph_->dequantized_buffers().back().data();
    }
    if (warm_max_blocks_ > 0) {
      auto bf16_vec = GetSharedWarmDummyBf16Scales(warm_max_blocks_);
      warm_dummy_bf16_scales_ = bf16_vec->data();
      graph_->bf16_scale_refs().push_back(std::move(bf16_vec));
    }
  }
  return absl::OkStatus();
}

absl::Status XnnpackBuildContext::DefineTensorValue(const graph::Tensor& tensor,
                                                    NnpackValue& value) {
  const auto& info = value.info;
  const bool is_external =
      (value.flags & (FlagExternalInput() | FlagExternalOutput())) != 0;
  const bool is_warm_fc = !is_external && info.buffer != nullptr &&
                          warm_fc_buffers_.contains(info.buffer.get());

  if (info.buffer && !is_external) {
    if (!is_warm_fc) {
      value.data = info.buffer->Lock();
    }
    graph_->keep_alive_buffers().push_back(info.buffer);
  }

  std::vector<size_t> dims;
  dims.reserve(info.shape.size());
  for (int dim : info.shape) {
    if (dim < -1) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": invalid negative tensor dimension ", dim));
    }
    dims.push_back(dim == -1 ? 0 : static_cast<size_t>(dim));
  }
  const void* data_ptr =
      is_warm_fc
          ? reinterpret_cast<const void*>(
                reinterpret_cast<uintptr_t>(info.buffer.get()) | uintptr_t{1})
          : value.data.data();
  xnn_datatype defined_dtype = GetXnnpackType(value);
  uint32_t external_id = is_external ? value.id : XNN_INVALID_VALUE_ID;
  xnn_subgraph_t sg = subgraph();

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
        sg, defined_dtype, dims.size(), dims.empty() ? nullptr : dims.data(),
        data_ptr, external_id, value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else if (auto maybe_pcq =
                 info.quantization->As<PerChannelAffineQuantization>();
             maybe_pcq.ok()) {
    const auto& pcq = maybe_pcq.value();
    // The per-channel scales of a channelwise XNNPACK value: `pcq.scales`, or
    // its single scale repeated for every channel.
    const auto channel_scales = [&]() -> absl::StatusOr<const float*> {
      if (pcq.quantized_dimension < 0 ||
          static_cast<size_t>(pcq.quantized_dimension) >= dims.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat(info.name, ": invalid quantized dimension ",
                         pcq.quantized_dimension));
      }
      const size_t channels = dims[pcq.quantized_dimension];
      if (pcq.scales.size() == 1 && channels != 1) {
        graph_->dequantized_buffers().push_back(
            std::vector<float>(channels, pcq.scales[0]));
        return graph_->dequantized_buffers().back().data();
      }
      if (pcq.scales.size() != channels) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name, ": per-channel scale count (", pcq.scales.size(),
            ") doesn't match the channel dimension size (", channels, ")"));
      }
      return pcq.scales.data();
    };
    if (info.type == Type::kI2) {
      LRT_TENSOR_ASSIGN_OR_RETURN(const float* scales, channel_scales());
      const size_t channels = dims[pcq.quantized_dimension];
      const float* channelwise_zp_ptr = nullptr;
      if (!pcq.zero_points.IsSymmetric()) {
        LRT_TENSOR_ASSIGN_OR_RETURN(std::vector<float> channelwise_zps,
                                    pcq.zero_points.CopyAs<float>(channels));
        graph_->dequantized_buffers().push_back(std::move(channelwise_zps));
        channelwise_zp_ptr = graph_->dequantized_buffers().back().data();
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          xnn_define_channelwise_quantized_tensor_value_v3(
              sg, defined_dtype, /*zero_point=*/0, scales, dims.size(),
              pcq.quantized_dimension, dims.empty() ? nullptr : dims.data(),
              data_ptr, external_id, value.flags, &value.id,
              channelwise_zp_ptr))
          << "Could not define a new channelwise quantized int2 tensor value.";
    } else if (const absl::StatusOr<int32_t> scalar_zp =
                   pcq.zero_points.Scalar<int32_t>();
               pcq.scales.size() == 1 && info.type != Type::kI4 &&
               scalar_zp.ok()) {
      // Single integer zero point (or symmetric). Non-integer float zero points
      // and per-channel zero points are dequantized below.
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
          sg, defined_dtype, *scalar_zp, pcq.scales[0], dims.size(),
          dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
          value.flags, &value.id))
          << "Could not define a new quantized tensor value.";
    } else {
      if (!pcq.zero_points.IsSymmetric()) {
        // XNNPACK's channelwise quantized values have no zero points, so only
        // constant weights can be run, by dequantizing them here.
        if (!info.buffer || is_external) {
          // Tensors computed by the graph, e.g. by a `Transpose`, are unnamed.
          return absl::UnimplementedError(absl::StrCat(
              info.name, info.name.empty() ? "" : ": ",
              "XNNPACK only supports per-channel zero points of constants"));
        }
        LRT_TENSOR_ASSIGN_OR_RETURN(
            std::vector<float> f32_data,
            DequantizeConstantTensor(
                info, absl::MakeSpan(value.data.data(), value.data.size())));
        graph_->dequantized_buffers().push_back(std::move(f32_data));
        data_ptr = graph_->dequantized_buffers().back().data();
        defined_dtype = xnn_datatype_fp32;
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
            sg, xnn_datatype_fp32, dims.size(),
            dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
            value.flags, &value.id))
            << "Could not define a new tensor value after dequantization.";
      } else {
        LRT_TENSOR_ASSIGN_OR_RETURN(const float* scales, channel_scales());
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                sg, defined_dtype, /*zero_point=*/0, scales, dims.size(),
                pcq.quantized_dimension, dims.empty() ? nullptr : dims.data(),
                data_ptr, external_id, value.flags, &value.id,
                /*channelwise_zero_point=*/nullptr))
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
    if (is_warm_fc) {
      if (bwq.NumBlocks() < expected_block_count) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name, ": blockwise scale count (", bwq.NumBlocks(),
            ") is smaller than block_count (", expected_block_count, ")"));
      }
      if (info.type == Type::kI8) {
        defined_dtype = xnn_datatype_qcint8;
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                sg, defined_dtype, /*zero_point=*/0, warm_dummy_f32_scales_,
                dims.size(), /*channel_dim=*/0, dims.data(), data_ptr,
                external_id, value.flags, &value.id,
                /*channelwise_zero_point=*/nullptr))
            << "Could not define warm channelwise fallback tensor for "
               "blockwise INT8.";
      } else {
        defined_dtype = GetXnnpackType(value);
        const bool uses_sym_fallback =
            bwq.MayHaveFloatZeroPoints() ||
            bwq.HasCachedCpuSymmetricFallback() ||
            (bwq.zero_points.Data<float>().ok() &&
             !bwq.zero_points.Scalar<int32_t>().ok());
        const int32_t zero_point =
            uses_sym_fallback ? 0
                              : bwq.zero_points.Scalar<int32_t>().value_or(0);
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_blockwise_quantized_tensor_value_v2(
                sg, defined_dtype, zero_point, warm_dummy_bf16_scales_,
                dims.size(), bwq.quantized_dimension, bwq.block_size,
                dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                value.flags, xnn_datatype_bf16, &value.id))
            << "Could not define warm blockwise quantized tensor value.";
      }
    } else {
      bwq.EnsureMaterialized();
      if (bwq.scales.size() < expected_block_count) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name, ": blockwise scale count (", bwq.scales.size(),
            ") is smaller than block_count (", expected_block_count, ")"));
      }
      LRT_TENSOR_RETURN_IF_ERROR(
          bwq.zero_points.Validate<float>(bwq.scales.size()));
      const absl::StatusOr<int32_t> uniform_zp =
          bwq.zero_points.Scalar<int32_t>();
      // XNNPACK's `xnn_define_blockwise_quantized_tensor_value_v2` only accepts
      // a single scalar `int32_t zero_point` (no per-block affine zero points)
      // and does not support blockwise INT8 (`Q8_0`), so XNNPACK converts
      // `Q8_0` to per-channel symmetric `qcint8` and `Q4_1`/`Q4_K` to symmetric
      // `qbint4` via `GetOrCreateCpuSymmetricFallback`. For native per-block
      // affine zero points on CPU without symmetric re-quantization, prefer the
      // YNNPACK backend.
      if (info.type == Type::kI8) {
        // Other layouts are unsupported rather than invalid.
        if (dims.size() != 2 || value.data.data() == nullptr ||
            bwq.quantized_dimension != 0 ||
            dims[1] % static_cast<size_t>(bwq.block_size) != 0) {
          return absl::UnimplementedError(absl::StrCat(
              info.name,
              ": XNNPACK only supports blockwise INT8 tensors as 2D constant "
              "weights quantized along dimension 0"));
        }
        // TODO(b/577601974): Remove this transitional lossy fallback once the
        // lossless YNNPACK blockwise backend lands (cl/1014238277).
        ABSL_LOG_FIRST_N(WARNING, 1)
            << "XNNPACK does not natively support blockwise INT8 weights; "
               "re-quantizing '"
            << info.name
            << "' to per-channel INT8 via lossy CPU symmetric fallback.";
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto fallback,
            bwq.GetOrCreateCpuSymmetricFallback(
                Type::kI8, static_cast<int>(dims[0]), static_cast<int>(dims[1]),
                value.data.data(), value.data.size()));
        const float* scales_ptr = fallback->scales.data();
        data_ptr = fallback->data->data();
        defined_dtype = xnn_datatype_qcint8;
        graph_->cpu_fallback_refs().push_back(std::move(fallback));
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                sg, defined_dtype, /*zero_point=*/0, scales_ptr, dims.size(),
                /*channel_dim=*/0, dims.data(), data_ptr, external_id,
                value.flags, &value.id, /*channelwise_zero_point=*/nullptr))
            << "Could not define channelwise fallback tensor for blockwise "
               "INT8.";
      } else if (info.type == Type::kI4 &&
                 (bwq.HasCachedCpuSymmetricFallback() ||
                  (bwq.zero_points.Data<float>().ok() && !uniform_zp.ok()))) {
        if (dims.size() != 2 || value.data.data() == nullptr ||
            bwq.quantized_dimension != 0 ||
            dims[1] % static_cast<size_t>(bwq.block_size) != 0) {
          return absl::UnimplementedError(absl::StrCat(
              info.name,
              ": XNNPACK only supports affine blockwise INT4 tensors as 2D "
              "constant weights quantized along dimension 0"));
        }
        // TODO(b/577601974): Remove this transitional lossy fallback once the
        // lossless YNNPACK blockwise backend lands (cl/1014238277).
        ABSL_LOG_FIRST_N(WARNING, 1)
            << "XNNPACK does not natively support affine blockwise INT4 "
               "zero-points; re-quantizing '"
            << info.name
            << "' to symmetric Q4_0 via lossy CPU symmetric fallback.";
        LRT_TENSOR_ASSIGN_OR_RETURN(
            auto fallback,
            bwq.GetOrCreateCpuSymmetricFallback(
                Type::kI4, static_cast<int>(dims[0]), static_cast<int>(dims[1]),
                value.data.data(), value.data.size()));
        const void* scale_ptr = fallback->bf16_scales->data();
        data_ptr = fallback->data->data();
        defined_dtype = xnn_datatype_qbint4;
        graph_->cpu_fallback_refs().push_back(std::move(fallback));
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_blockwise_quantized_tensor_value_v2(
                sg, defined_dtype, /*zero_point=*/0, scale_ptr, dims.size(),
                bwq.quantized_dimension, bwq.block_size, dims.data(), data_ptr,
                external_id, value.flags, xnn_datatype_bf16, &value.id))
            << "Could not define symmetric blockwise INT4 fallback tensor.";
      } else {
        if (!bwq.zero_points.IsUniform()) {
          return absl::UnimplementedError(absl::StrCat(
              info.name,
              ": XNNPACK doesn't support blockwise quantization with different "
              "zero points"));
        }
        if (!uniform_zp.ok()) {
          return uniform_zp.status();
        }
        const int32_t zero_point = *uniform_zp;
        std::shared_ptr<const std::vector<bf16_t>> bf16_scales =
            bwq.GetOrCreateBf16Scales();
        const void* scale_ptr = bf16_scales->data();
        graph_->bf16_scale_refs().push_back(std::move(bf16_scales));
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_blockwise_quantized_tensor_value_v2(
                sg, defined_dtype, zero_point, scale_ptr, dims.size(),
                bwq.quantized_dimension, bwq.block_size,
                dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                value.flags, xnn_datatype_bf16, &value.id))
            << "Could not define a new blockwise quantized tensor value.";
      }
    }
  } else {
    return absl::UnimplementedError("Unsupported quantization type.");
  }

  if (!is_external && data_ptr != nullptr) {
    static_cast<XnnpackGraph*>(graph_.get())
        ->constant_fingerprints()
        .push_back(XnnConstantFingerprint{
            .data_ptr = data_ptr,
            .source_buffer = info.buffer.get(),
            .source_quantization = info.quantization.get(),
            .content_hash = ComputeConstantTensorHash(info, value.data.data(),
                                                      value.data.size()),
            .min_packed_bytes =
                ComputeMinPackedBytesForXnnType(defined_dtype, dims),
        });
  }
  return absl::OkStatus();
}

absl::StatusOr<uint32_t>
XnnpackBuildContext::DefinePerTensorQuantizedWeightsAsChannelwise(
    const graph::Tensor& weights) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(weights));
  NnpackValue value;
  value.info = info;
  if (value.info.quantization == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": missing quantization parameters"));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const PerChannelAffineQuantization& pcq,
      value.info.quantization->As<const PerChannelAffineQuantization>());
  if (pcq.scales.size() != 1 || value.info.shape.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        info.name, ": expected per-tensor quantized weights with a rank > 0"));
  }
  if (value.info.type != Type::kI8) {
    return absl::InvalidArgumentError(
        absl::StrCat(info.name, ": expected int8 weights"));
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      pcq.zero_points.Validate<int32_t>(pcq.scales.size()));
  LRT_TENSOR_ASSIGN_OR_RETURN(const int32_t zero_point,
                              pcq.zero_points.Scalar<int32_t>());
  if (value.info.buffer == nullptr) {
    return absl::UnimplementedError(absl::StrCat(
        info.name,
        ": XNNPACK only supports per-tensor quantized constant weights"));
  }

  if (absl::c_any_of(value.info.shape, [](int32_t v) { return v < 0; })) {
    return absl::InvalidArgumentError(absl::StrCat(
        info.name,
        ": negative dimensions are not supported in constant weights."));
  }
  std::vector<size_t> dims(value.info.shape.begin(), value.info.shape.end());
  value.data = value.info.buffer->Lock();
  graph_->keep_alive_buffers().push_back(value.info.buffer);
  graph_->dequantized_buffers().emplace_back(dims[0], pcq.scales[0]);
  const float* scales = graph_->dequantized_buffers().back().data();
  LRT_TENSOR_RETURN_IF_ERROR(xnn_define_channelwise_quantized_tensor_value_v3(
      subgraph(), xnn_datatype_qcint8, zero_point, scales, dims.size(),
      /*channel_dim=*/0, dims.data(), value.data.data(), XNN_INVALID_VALUE_ID,
      /*flags=*/0, &value.id, /*channelwise_zero_point=*/nullptr))
      << "Could not define per-tensor quantized weights as channelwise "
         "quantized weights.";
  if (value.data.data() != nullptr) {
    static_cast<XnnpackGraph*>(graph_.get())
        ->constant_fingerprints()
        .push_back(XnnConstantFingerprint{
            .data_ptr = value.data.data(),
            .source_buffer = value.info.buffer.get(),
            .source_quantization = value.info.quantization.get(),
            .content_hash = ComputeConstantTensorHash(
                value.info, value.data.data(), value.data.size()),
            .min_packed_bytes =
                ComputeMinPackedBytesForXnnType(xnn_datatype_qcint8, dims),
        });
  }
  // The value isn't in the tensor index: the other consumers of `weights` get
  // the per-tensor quantized value.
  graph_->mutable_values().push_back(std::move(value));
  return graph_->mutable_values().back().id;
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

  LRT_TENSOR_RETURN_IF_ERROR(XnnStatusToAbsl(
      xnn_define_tensor_value(subgraph(), xnn_type, shape.size(),
                              shape.empty() ? nullptr : shape.data(), data,
                              XNN_INVALID_VALUE_ID, /*flags=*/0, id),
      "xnn_define_tensor_value"));
  if (data != nullptr) {
    const uint64_t raw_bytes = ComputeMinPackedBytesForXnnType(xnn_type, shape);
    uint64_t h = kFnvOffsetBasis;
    FnvMixScalar(&h, static_cast<int32_t>(datatype));
    FnvMixScalar(&h, static_cast<uint64_t>(shape.size()));
    for (size_t dim : shape) {
      FnvMixScalar(&h, static_cast<int64_t>(dim));
    }
    FnvMixSampledBytes(&h, static_cast<const std::byte*>(data),
                       static_cast<size_t>(raw_bytes));
    static_cast<XnnpackGraph*>(graph_.get())
        ->constant_fingerprints()
        .push_back(XnnConstantFingerprint{
            .data_ptr = data,
            .content_hash = h == 0 ? 1ULL : h,
            .min_packed_bytes = raw_bytes,
        });
  }
  return absl::OkStatus();
}

void RegisterXnnWeightsCacheFingerprintSink(xnn_weights_cache_t cache,
                                            XnnConstantFingerprintSink sink) {
  if (cache == nullptr || sink == nullptr) return;
  FingerprintSinkRegistry& reg = GetFingerprintSinkRegistry();
  absl::MutexLock lock(reg.mu);
  reg.sinks[cache] = sink;
}

void UnregisterXnnWeightsCacheFingerprintSink(xnn_weights_cache_t cache) {
  if (cache == nullptr) return;
  FingerprintSinkRegistry& reg = GetFingerprintSinkRegistry();
  absl::MutexLock lock(reg.mu);
  reg.sinks.erase(cache);
}

void NotifyXnnWeightsCacheConstantFingerprints(
    xnn_weights_cache_t cache,
    absl::Span<const XnnConstantFingerprint> fingerprints) {
  if (cache == nullptr || fingerprints.empty()) return;
  XnnConstantFingerprintSink sink = nullptr;
  {
    FingerprintSinkRegistry& reg = GetFingerprintSinkRegistry();
    absl::MutexLock lock(reg.mu);
    auto it = reg.sinks.find(cache);
    if (it != reg.sinks.end()) {
      sink = it->second;
    }
  }
  if (sink != nullptr) {
    sink(cache->context, fingerprints);
  }
}

absl::Status XnnpackBuildContext::LowerOp(const graph::Operation& op) {
  auto op_ext = op.GetExtension<XnnpackOperation>();
  if (op_ext == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Operation ", op.GetName(), " does not implement XNNPACK operation."));
  }
  return op_ext->ToXnnpack(op, *this);
}

namespace {

enum class WarmFcCacheMode {
  kDisabled,
  kEnabled,
};

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphImpl(
    std::vector<TensorHandle> outputs, WarmFcCacheMode warm_fc_cache_mode) {
  const bool warm_fc_cache_only =
      (warm_fc_cache_mode == WarmFcCacheMode::kEnabled);
  uint32_t next_id = 0;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  absl::flat_hash_set<const Buffer*> ineligible_buffers;
  for (const TensorHandle& out : outputs) {
    const graph::Tensor raw = out.GetRaw();
    auto [it, inserted] = external_ids.insert({raw, next_id});
    next_id += inserted;
    if (warm_fc_cache_only && raw.group != nullptr) {
      for (const graph::TensorInformation& info : raw.group->tensor_infos) {
        if (info.buffer != nullptr) {
          ineligible_buffers.insert(info.buffer.get());
        }
      }
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(outputs));
  absl::flat_hash_set<const Buffer*> candidate_fc_buffers;
  size_t max_warm_rows = 0;
  size_t max_warm_blocks = 0;
  for (const graph::Operation* op : plan) {
    const bool is_fc =
        warm_fc_cache_only && op->As<graph::FullyConnectedOperation>().ok();
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      const graph::Tensor& t = op->inputs[i];
      if (auto info_or = graph::GetInfo(t);
          !info_or.ok() || info_or->buffer != nullptr) {
        if (warm_fc_cache_only && t.group != nullptr &&
            t.group->producer == nullptr) {
          if (t.group->tensor_infos.size() != 1) {
            for (const graph::TensorInformation& inf : t.group->tensor_infos) {
              if (inf.buffer != nullptr) {
                ineligible_buffers.insert(inf.buffer.get());
              }
            }
          } else if (info_or.ok() && info_or->buffer != nullptr) {
            bool eligible = false;
            size_t need_rows = 0;
            size_t need_blocks = 0;
            if (is_fc && i == 1 && op->inputs.size() >= 2 &&
                (info_or->buffer->HasDeterministicRawHash() ||
                 info_or->buffer->IsMaterialized()) &&
                info_or->shape.size() == 2 && info_or->shape[0] > 0 &&
                info_or->shape[1] > 0 &&
                static_cast<uint64_t>(info_or->shape[0]) *
                        static_cast<uint64_t>(info_or->shape[1]) >
                    1ULL) {
              auto act_info = graph::GetInfo(op->inputs[0]);
              const bool valid_act =
                  act_info.ok() && (act_info->type == Type::kFP32 ||
                                    act_info->type == Type::kFP16);
              const bool valid_bias = op->inputs.size() < 3 ||
                                      op->inputs[2].group == nullptr ||
                                      op->inputs[2].group->producer == nullptr;
              if (valid_act && valid_bias) {
                if (info_or->quantization == nullptr) {
                  // XNNPACK's mixed-precision FP32-input/FP16-weight FC
                  // operator dereferences `f16_kernel` before checking
                  // `weights_cache`, so unquantized weights are only warm-cache
                  // safe when activation and weight float types match.
                  eligible = (info_or->type == Type::kFP32 &&
                              act_info->type == Type::kFP32) ||
                             (info_or->type == Type::kFP16 &&
                              act_info->type == Type::kFP16);
                } else if (auto pcq = info_or->quantization
                                          ->As<PerChannelAffineQuantization>();
                           pcq.ok()) {
                  const int qdim = pcq->quantized_dimension;
                  eligible =
                      (info_or->type == Type::kI8 ||
                       info_or->type == Type::kI4 ||
                       info_or->type == Type::kI2) &&
                      pcq->scales.size() > 1 &&
                      pcq->zero_points.IsSymmetric() && qdim >= 0 &&
                      static_cast<size_t>(qdim) < info_or->shape.size() &&
                      pcq->scales.size() >=
                          static_cast<size_t>(info_or->shape[qdim]);
                } else if (auto bwq = info_or->quantization
                                          ->As<BlockwiseQuantization>();
                           bwq.ok()) {
                  const size_t rows = static_cast<size_t>(info_or->shape[0]);
                  const size_t cols = static_cast<size_t>(info_or->shape[1]);
                  if ((info_or->type == Type::kI8 ||
                       info_or->type == Type::kI4) &&
                      (info_or->buffer->HasDeterministicRawHash() ||
                       bwq->IsMaterialized()) &&
                      bwq->block_size > 0 &&
                      (rows * cols) % static_cast<size_t>(bwq->block_size) ==
                          0) {
                    const size_t blocks =
                        (rows * cols) / static_cast<size_t>(bwq->block_size);
                    if (bwq->NumBlocks() >= blocks) {
                      eligible = true;
                      if (info_or->type == Type::kI8) {
                        need_rows = rows;
                      } else {
                        need_blocks = blocks;
                      }
                    }
                  }
                }
              }
            }
            if (eligible) {
              candidate_fc_buffers.insert(info_or->buffer.get());
              max_warm_rows = std::max(max_warm_rows, need_rows);
              max_warm_blocks = std::max(max_warm_blocks, need_blocks);
            } else {
              ineligible_buffers.insert(info_or->buffer.get());
            }
          }
        }
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
  if (warm_fc_cache_only && !candidate_fc_buffers.empty()) {
    absl::flat_hash_set<const Buffer*> warm_fc_buffers;
    warm_fc_buffers.reserve(candidate_fc_buffers.size());
    for (const Buffer* buf : candidate_fc_buffers) {
      if (!ineligible_buffers.contains(buf)) {
        warm_fc_buffers.insert(buf);
      }
    }
    if (!warm_fc_buffers.empty()) {
      ctx.SetWarmFcBuffers(std::move(warm_fc_buffers), max_warm_rows,
                           max_warm_blocks);
    }
  }
  LRT_TENSOR_RETURN_IF_ERROR(BuildNnpackGraph(ctx));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto graph, ctx.Finalize());
  return std::unique_ptr<XnnpackGraph>(
      static_cast<XnnpackGraph*>(graph.release()));
}

}  // namespace

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs) {
  return BuildXnnpackGraphImpl(std::move(outputs), WarmFcCacheMode::kDisabled);
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphWithWarmFcCache(
    std::vector<TensorHandle> outputs) {
  return BuildXnnpackGraphImpl(std::move(outputs), WarmFcCacheMode::kEnabled);
}

}  // namespace litert::tensor
