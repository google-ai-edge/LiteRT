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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
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
    desc.SetBHWCShape(bhwc_shape);
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
    desc.SetBHWCShape(shape);
    handle = ctx.gpu_model_builder->AddTensor(desc);
  }
  auto [new_it, _] = ctx.graph_tensor_to_handle.insert({t, handle});
  return new_it->second;
}

}  // namespace litert::tensor
