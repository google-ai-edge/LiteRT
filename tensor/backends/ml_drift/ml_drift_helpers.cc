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

#include "tensor/backends/ml_drift/ml_drift_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tensor/backends/ml_drift/ml_drift_buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {

void MlDriftBuildContext::AddWeightPackStat(absl::string_view kind,
                                            absl::Duration time, size_t bytes,
                                            int count) {
  WeightPackStat& stat = weight_pack_stats[kind];
  stat.time += time;
  stat.bytes += bytes;
  stat.count += count;
}

static bool IsSub8Bit(ml_drift::DataType type) {
  return type == ml_drift::DataType::kInt4 ||
         type == ml_drift::DataType::kUint4 ||
         type == ml_drift::DataType::kInt2;
}

absl::StatusOr<ml_drift::BHWC> ShapeToBHWC(const std::vector<int>& shape,
                                           int fill_value) {
  switch (shape.size()) {
    case 0:
      return ml_drift::BHWC(fill_value, fill_value, fill_value, fill_value);
    case 1:
      return ml_drift::BHWC(fill_value, fill_value, fill_value, shape[0]);
    case 2:
      return ml_drift::BHWC(shape[0], fill_value, fill_value, shape[1]);
    case 3:
      return ml_drift::BHWC(fill_value, shape[0], shape[1], shape[2]);
    case 4:
      return ml_drift::BHWC(shape[0], shape[1], shape[2], shape[3]);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot convert shape ", ShapeStr(shape), " to MlDrift BHWC."));
  }
}

absl::StatusOr<ml_drift::DataType> GetMlDriftDataType(
    Type ty, ml_drift::DataType ml_drift_float_type) {
  switch (ty) {
    case Type::kBOOL:
      return ml_drift::DataType::kBool;
    case Type::kFP32:
      return ml_drift_float_type;
    case Type::kFP16:
      return ml_drift::DataType::kFloat16;
    case Type::kI8:
      return ml_drift::DataType::kInt8;
    case Type::kU8:
      return ml_drift::DataType::kUint8;
    case Type::kI4:
      return ml_drift::DataType::kInt4;
    case Type::kU4:
      return ml_drift::DataType::kUint4;
    case Type::kI16:
      return ml_drift::DataType::kInt16;
    case Type::kU16:
      return ml_drift::DataType::kUint16;
    case Type::kI32:
      return ml_drift::DataType::kInt32;
    case Type::kU32:
      return ml_drift::DataType::kUint32;
    case Type::kI2:
      return ml_drift::DataType::kInt2;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type: ", ToString(ty)));
  }
}

// Returns the descriptor, without data, of the constant tensor `info` as an ML
// Drift constant tensor of type `mld_data_type`.
static absl::StatusOr<ml_drift::TensorDescriptor> GetConstantTensorDescriptor(
    MlDriftBuildContext& ctx, const graph::TensorInformation& info,
    ml_drift::DataType mld_data_type, bool create_linear_tensor,
    bool force_buffer_storage,
    ml_drift::TensorDescriptor::PhysicalLayout1D physical_layout_1d) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ShapeToBHWC(info.shape));
  auto locked_buffer = info.buffer->Lock();
  int64_t element_count = 1;
  for (int dim : info.shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Negative dimension in constant tensor shape: ",
                       ShapeStr(info.shape)));
    }
    element_count *= dim;
  }
  const size_t expected_bytes =
      BufferSize(info.type, static_cast<size_t>(element_count));
  if (locked_buffer.size() < expected_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Constant buffer too small for shape/type. Expected >= ",
                     expected_bytes, " bytes, got ", locked_buffer.size()));
  }
  ml_drift::TensorStorageType storage_type;
  ml_drift::Layout layout;
  ml_drift::BHWC bhwc_shape;
  if (create_linear_tensor) {
    storage_type = ml_drift::TensorStorageType::kBuffer;
    if (ctx.gpu_model_builder->gpu_info().SupportsImageBuffer() &&
        !IsSub8Bit(mld_data_type)) {
      storage_type = ml_drift::TensorStorageType::kImageBuffer;
    }
    layout = ml_drift::Layout::kLinear;
    bhwc_shape = ml_drift::BHWC(1, 1, 1, info.shape[0]);
  } else {
    storage_type = ctx.gpu_model_builder->default_storage();
    if (force_buffer_storage) {
      storage_type = ml_drift::TensorStorageType::kBuffer;
      if (ctx.gpu_model_builder->gpu_info().SupportsImageBuffer() &&
          !IsSub8Bit(mld_data_type)) {
        storage_type = ml_drift::TensorStorageType::kImageBuffer;
      }
    }
    layout = shape.b == 1 ? ml_drift::Layout::kHWC : ml_drift::Layout::kBHWC;
    bhwc_shape = shape;
  }
  ml_drift::TensorDescriptor desc(mld_data_type, storage_type, layout,
                                  physical_layout_1d);
  if (!create_linear_tensor && !force_buffer_storage) {
    desc.UpdateToSupportedStorageType(ctx.gpu_model_builder->gpu_info(),
                                      bhwc_shape)
        .IgnoreError();
  }
  desc.SetBHWCShape(bhwc_shape);
  return desc;
}

// Uploads the data of the constant tensor `info` to `desc`.
static void UploadConstantData(const graph::TensorInformation& info,
                               ml_drift::TensorDescriptor& desc) {
  auto locked_buffer = info.buffer->Lock();
  const void* raw_data = locked_buffer.data();
  switch (info.type) {
    case Type::kFP32:
      desc.UploadData(reinterpret_cast<const float*>(raw_data));
      break;
    case Type::kI32:
      desc.UploadData(reinterpret_cast<const int32_t*>(raw_data));
      break;
    case Type::kU32:
      desc.UploadData(reinterpret_cast<const uint32_t*>(raw_data));
      break;
    case Type::kI8:
      desc.UploadData(reinterpret_cast<const int8_t*>(raw_data));
      break;
    case Type::kU8:
      desc.UploadData(reinterpret_cast<const uint8_t*>(raw_data));
      break;
    case Type::kBOOL:
      desc.UploadData(reinterpret_cast<const bool*>(raw_data));
      break;
    default: {
      const auto* bytes = reinterpret_cast<const uint8_t*>(raw_data);
      absl::Span<const uint8_t> data(bytes, locked_buffer.size());
      desc.UploadDataRaw<uint8_t>(data);
      break;
    }
  }
}

// Uploads the constant tensor `info` as an ML Drift constant tensor of type
// `mld_data_type`, without caching its handle.
static absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
AddConstantTensorHandle(
    MlDriftBuildContext& ctx, const graph::TensorInformation& info,
    ml_drift::DataType mld_data_type, bool create_linear_tensor,
    bool force_buffer_storage,
    ml_drift::TensorDescriptor::PhysicalLayout1D physical_layout_1d) {
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto desc, GetConstantTensorDescriptor(
                     ctx, info, mld_data_type, create_linear_tensor,
                     force_buffer_storage, physical_layout_1d));
  UploadConstantData(info, desc);
  return ctx.gpu_model_builder->AddConstantTensor(std::move(desc));
}

absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle> GetTensorHandle(
    MlDriftBuildContext& ctx, const graph::Tensor& t, bool create_if_not_found,
    bool create_linear_tensor, bool force_buffer_storage,
    ml_drift::TensorDescriptor::PhysicalLayout1D physical_layout_1d) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, graph::GetInfo(t));
  if (info.name == "model.embed_tokens.weight") {
    force_buffer_storage = true;
  }
  auto it = ctx.graph_tensor_to_handle.find(t);
  if (it != ctx.graph_tensor_to_handle.end()) {
    return it->second;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto mld_data_type,
      GetMlDriftDataType(info.type, ctx.default_float_type));

  if (ctx.external_weight_names.contains(info.name)) {
    LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ShapeToBHWC(info.shape));
    ml_drift::TensorStorageType storage_type =
        ctx.gpu_model_builder->default_storage();
    if (force_buffer_storage) {
      storage_type = ml_drift::TensorStorageType::kBuffer;
      if (ctx.gpu_model_builder->gpu_info().SupportsImageBuffer() &&
          !IsSub8Bit(mld_data_type)) {
        storage_type = ml_drift::TensorStorageType::kImageBuffer;
      }
    }
    ml_drift::Layout layout =
        shape.b == 1 ? ml_drift::Layout::kHWC : ml_drift::Layout::kBHWC;
    ml_drift::TensorDescriptor desc(mld_data_type, storage_type, layout,
                                    physical_layout_1d);
    desc.SetBHWCShape(shape);
    auto handle = ctx.gpu_model_builder->AddTensor(desc);
    ctx.graph_tensor_to_handle.insert({t, handle});
    ctx.weight_name_to_handle.insert({info.name, handle});
    return handle;
  }

  if (info.buffer) {
    if (auto mld_buffer_or = info.buffer->As<MlDriftBuffer>();
        mld_buffer_or.ok()) {
      ml_drift::GpuModelBuilder::TensorHandle handle;
      handle.id = mld_buffer_or->value_id();
      ctx.graph_tensor_to_handle.insert({t, handle});
      return handle;
    }
    // This is a constant tensor.
    LRT_TENSOR_ASSIGN_OR_RETURN(
        auto desc, GetConstantTensorDescriptor(
                       ctx, info, mld_data_type, create_linear_tensor,
                       force_buffer_storage, physical_layout_1d));
    if (!info.name.empty() && ctx.shared_weight_descriptors != nullptr) {
      auto [shared_it, inserted] =
          ctx.shared_weight_descriptors->try_emplace(info.name, desc);
      if (inserted) {
        UploadConstantData(info, shared_it->second);
      }
      if (shared_it->second == desc &&
          shared_it->second.GetBHWDCShape() == desc.GetBHWDCShape()) {
        auto handle = ctx.gpu_model_builder->AddConstantTensor(
            CopyDescriptorWithoutData(shared_it->second));
        ctx.graph_tensor_to_handle.insert({t, handle});
        ctx.weight_name_to_handle.insert({info.name, handle});
        return handle;
      }
    }
    UploadConstantData(info, desc);
    auto handle = ctx.gpu_model_builder->AddConstantTensor(std::move(desc));
    ctx.graph_tensor_to_handle.insert({t, handle});
    if (!info.name.empty()) {
      ctx.weight_name_to_handle.insert({info.name, handle});
    }
    return handle;
  }

  if (!create_if_not_found) {
    return absl::NotFoundError(absl::StrCat(
        "Tensor ", absl::HashOf(t), " not found in graph. Total entries: ",
        ctx.graph_tensor_to_handle.size()));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ShapeToBHWC(info.shape));

  ml_drift::GpuModelBuilder::TensorHandle handle;
  if (create_linear_tensor) {
    auto storage_type = ml_drift::TensorStorageType::kBuffer;
    if (ctx.gpu_model_builder->gpu_info().SupportsImageBuffer() &&
        !IsSub8Bit(mld_data_type)) {
      storage_type = ml_drift::TensorStorageType::kImageBuffer;
    }
    ml_drift::TensorDescriptor desc(mld_data_type, storage_type,
                                    ml_drift::Layout::kLinear);
    desc.SetBHWCShape(ml_drift::BHWC(1, 1, 1, info.shape[0]));
    handle = ctx.gpu_model_builder->AddTensor(desc);
  } else {
    auto storage_type = ctx.gpu_model_builder->default_storage();
    if (force_buffer_storage) {
      storage_type = ml_drift::TensorStorageType::kBuffer;
      if (ctx.gpu_model_builder->gpu_info().SupportsImageBuffer() &&
          !IsSub8Bit(mld_data_type)) {
        storage_type = ml_drift::TensorStorageType::kImageBuffer;
      }
    }
    ml_drift::TensorDescriptor desc(
        mld_data_type, storage_type,
        shape.b == 1 ? ml_drift::Layout::kHWC : ml_drift::Layout::kBHWC,
        physical_layout_1d);
    if (!force_buffer_storage) {
      desc.UpdateToSupportedStorageType(ctx.gpu_model_builder->gpu_info(),
                                        shape)
          .IgnoreError();
    }
    desc.SetBHWCShape(shape);
    handle = ctx.gpu_model_builder->AddTensor(desc);
  }
  auto [new_it, _] = ctx.graph_tensor_to_handle.insert({t, handle});
  return new_it->second;
}

absl::StatusOr<ml_drift::GpuModelBuilder::TensorHandle>
GetConstantTensorHandleInTensorLayout(MlDriftBuildContext& ctx,
                                      const graph::Tensor& t) {
  LRT_TENSOR_ASSIGN_OR_RETURN(const auto& info, graph::GetInfo(t));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto mld_data_type,
      GetMlDriftDataType(info.type, ctx.default_float_type));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto shape, ShapeToBHWC(info.shape));
  if (auto it = ctx.graph_tensor_to_handle.find(t);
      it != ctx.graph_tensor_to_handle.end()) {
    const ml_drift::TensorDescriptor& desc = it->second.tensor_desc;
    if (desc.GetDataType() == mld_data_type && desc.GetBHWCShape() == shape &&
        desc.GetLayout() != ml_drift::Layout::kLinear) {
      return it->second;
    }
  }
  if (auto it = ctx.constant_handles_in_tensor_layout.find(t);
      it != ctx.constant_handles_in_tensor_layout.end()) {
    return it->second;
  }
  if (info.buffer == nullptr || info.buffer->As<MlDriftBuffer>().ok()) {
    return absl::UnimplementedError(absl::StrCat(
        "Tensor ", info.name, " isn't a constant that can be uploaded."));
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto handle, AddConstantTensorHandle(
                       ctx, info, mld_data_type, /*create_linear_tensor=*/false,
                       /*force_buffer_storage=*/false,
                       ml_drift::TensorDescriptor::PhysicalLayout1D::kDCHWBC4));
  ctx.constant_handles_in_tensor_layout.insert({t, handle});
  return handle;
}

std::optional<bool> ParseBoolEnvValue(absl::string_view env_value) {
  if (env_value.empty()) return std::nullopt;
  bool value = false;
  return absl::SimpleAtob(env_value, &value) ? std::optional<bool>(value)
                                             : std::nullopt;
}

std::optional<bool> BoolFromEnv(const char* env_var_name) {
  if (env_var_name == nullptr) return std::nullopt;
  const char* raw_env = std::getenv(env_var_name);
  if (raw_env == nullptr) return std::nullopt;
  return ParseBoolEnvValue(raw_env);
}

absl::StatusOr<std::vector<float>> AdditiveZeroPoints(
    const graph::PerChannelAffineQuantization& pcq, int num_channels) {
  if (num_channels < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Negative number of channels: ", num_channels));
  }
  const auto one_or_per_channel = [&](size_t count) {
    return count == 1 || count == static_cast<size_t>(num_channels);
  };
  // Integer or float zero points: none, one, or one per channel.
  if (!one_or_per_channel(pcq.scales.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        pcq.scales.size(), " scales don't match ", num_channels, " channels"));
  }
  LRT_TENSOR_RETURN_IF_ERROR(
      pcq.zero_points.Validate<float>(static_cast<size_t>(num_channels)));
  std::vector<float> values(num_channels, 0.0f);
  LRT_TENSOR_RETURN_IF_ERROR(
      pcq.zero_points.CopyTo<float>(absl::MakeSpan(values)));
  for (int o = 0; o < num_channels; ++o) {
    const float scale = pcq.scales[pcq.scales.size() == 1 ? 0 : o];
    values[o] *= -scale;
  }
  return values;
}

namespace {

template <typename Fn>
void ParallelForRanges(size_t total, size_t min_grain, Fn&& fn,
                       size_t max_threads = 8) {
  if (total == 0) return;
  const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
  const size_t workers = std::min<size_t>(
      {max_threads, hw, (total + std::max<size_t>(1, min_grain) - 1) /
                            std::max<size_t>(1, min_grain)});
  if (workers <= 1) {
    fn(0, total);
    return;
  }
  const size_t chunk = (total + workers - 1) / workers;
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (size_t t = 1; t < workers; ++t) {
    const size_t begin = t * chunk;
    if (begin >= total) break;
    const size_t end = std::min(total, begin + chunk);
    threads.emplace_back([&, begin, end]() { fn(begin, end); });
  }
  fn(0, std::min(total, chunk));
  for (std::thread& t : threads) t.join();
}

}  // namespace

absl::StatusOr<std::vector<int32_t>> ComputePackedInt2WeightsSumI(
    absl::Span<const uint8_t> src, const ml_drift::OHWI& shape) {
  const int out_channels = shape.o;
  if (out_channels <= 0 || shape.h <= 0 || shape.w <= 0 || shape.i <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-positive INT2 weight dimensions: [", shape.o, ", ",
                     shape.h, ", ", shape.w, ", ", shape.i, "]"));
  }
  const size_t row_elems = static_cast<size_t>(shape.h) *
                           static_cast<size_t>(shape.w) *
                           static_cast<size_t>(shape.i);
  const size_t total_elems = static_cast<size_t>(out_channels) * row_elems;
  const size_t expected_bytes = (total_elems + 3) / 4;
  if (src.size() < expected_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Packed INT2 weight buffer too small: got ", src.size(),
                     " bytes, expected at least ", expected_bytes, " bytes"));
  }
  std::vector<int32_t> sum_i(out_channels, 0);
  const uint8_t* raw = src.data();
  const bool aligned_row = (row_elems & 3u) == 0;
  const size_t row_bytes = row_elems >> 2;
  auto sum_row = [&](int o) {
    if (aligned_row) {
      const uint8_t* row = raw + static_cast<size_t>(o) * row_bytes;
      int64_t unsigned_sum = 0;
      for (size_t j = 0; j < row_bytes; ++j) {
        const uint8_t u = row[j] ^ 0xAAu;
        const uint8_t pair_sum =
            static_cast<uint8_t>((u & 0x33u) + ((u >> 2) & 0x33u));
        unsigned_sum += static_cast<int64_t>(pair_sum & 0x0Fu) +
                        static_cast<int64_t>(pair_sum >> 4);
      }
      sum_i[o] = static_cast<int32_t>(unsigned_sum -
                                      2 * static_cast<int64_t>(row_elems));
    } else {
      int32_t row_sum = 0;
      const size_t base = static_cast<size_t>(o) * row_elems;
      for (size_t k = 0; k < row_elems; ++k) {
        const size_t idx = base + k;
        const uint8_t crumb = (raw[idx >> 2] >> ((idx & 3u) * 2)) & 0x03u;
        row_sum += static_cast<int32_t>(crumb ^ 0x02u) - 2;
      }
      sum_i[o] = row_sum;
    }
  };
  if (total_elems < (512ULL << 10)) {
    for (int o = 0; o < out_channels; ++o) sum_row(o);
  } else {
    const size_t min_grain_rows =
        std::max<size_t>(1, ((64ULL << 10) + row_elems - 1) / row_elems);
    ParallelForRanges(
        static_cast<size_t>(out_channels), min_grain_rows,
        [&](size_t begin, size_t end) {
          for (size_t o = begin; o < end; ++o) {
            sum_row(static_cast<int>(o));
          }
        },
        /*max_threads=*/8);
  }
  return sum_i;
}

absl::StatusOr<std::vector<int32_t>> ComputeInt8WeightsSumI(
    absl::Span<const int8_t> src, const ml_drift::OHWI& shape) {
  const int out_channels = shape.o;
  if (out_channels <= 0 || shape.h <= 0 || shape.w <= 0 || shape.i <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-positive INT8 weight dimensions: [", shape.o, ", ",
                     shape.h, ", ", shape.w, ", ", shape.i, "]"));
  }
  const size_t row_elems = static_cast<size_t>(shape.h) *
                           static_cast<size_t>(shape.w) *
                           static_cast<size_t>(shape.i);
  const size_t total_elems = static_cast<size_t>(out_channels) * row_elems;
  if (src.size() < total_elems) {
    return absl::InvalidArgumentError(
        absl::StrCat("INT8 weight buffer too small: got ", src.size(),
                     " bytes, expected at least ", total_elems, " bytes"));
  }
  std::vector<int32_t> sum_i(out_channels, 0);
  const int8_t* raw = src.data();
  auto sum_row = [&](int o) {
    const int8_t* row = raw + static_cast<size_t>(o) * row_elems;
    int64_t row_sum = 0;
    for (size_t j = 0; j < row_elems; ++j) {
      row_sum += static_cast<int64_t>(row[j]);
    }
    sum_i[o] = static_cast<int32_t>(row_sum);
  };
  if (total_elems < (512ULL << 10)) {
    for (int o = 0; o < out_channels; ++o) sum_row(o);
  } else {
    const size_t min_grain_rows =
        std::max<size_t>(1, ((64ULL << 10) + row_elems - 1) / row_elems);
    ParallelForRanges(
        static_cast<size_t>(out_channels), min_grain_rows,
        [&](size_t begin, size_t end) {
          for (size_t o = begin; o < end; ++o) {
            sum_row(static_cast<int>(o));
          }
        },
        /*max_threads=*/8);
  }
  return sum_i;
}

}  // namespace litert::tensor
