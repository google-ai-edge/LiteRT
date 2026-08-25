// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/composite/experts_remap_builder.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_apple_mpp.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_generic.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_wave_matrix.h"  // from @ml_drift
#include "ml_drift/common/kernels/conv_wave_memory.h"  // from @ml_drift
#include "ml_drift/common/kernels/fully_connected.h"  // from @ml_drift
#include "ml_drift/common/kernels/google/custom/experts_remap.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift

namespace litert::ml_drift {

std::vector<::ml_drift::GpuModelBuilder::TensorHandle> CreateExpertsRemap(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& indices, int num_experts) {
  const ::ml_drift::BHWC& indices_shape = indices.tensor_desc.GetBHWCShape();
  auto experts_remap = builder.AddTensor(
      ::ml_drift::BHWC(1, num_experts, indices_shape.w, 2),
      ::ml_drift::DataType::INT32);
  auto experts_remap_op = ::ml_drift::CreateExpertsRemapOp(
      indices.tensor_desc, experts_remap.tensor_desc);

  ::ml_drift::TensorDescriptor dst_count_desc = ::ml_drift::TensorDescriptor(
      ::ml_drift::DataType::INT32, ::ml_drift::TensorStorageType::BUFFER,
      ::ml_drift::Layout::HWC);
  dst_count_desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, num_experts));
  auto experts_count = builder.AddTensor(dst_count_desc);

  builder.AddGpuOperation({indices}, {experts_remap, experts_count},
                          std::move(experts_remap_op), "create_experts_remap");

  auto experts_offsets = builder.AddTensor(dst_count_desc);
  auto experts_offsets_op = ::ml_drift::CreateOffsetsOp();
  builder.AddGpuOperation({experts_count}, {experts_offsets},
                          std::move(experts_offsets_op),
                          "create_experts_offsets");

  auto experts_remap_packed = builder.AddTensor(
      ::ml_drift::BHWC(1, 1, indices_shape.c * indices_shape.w, 2),
      ::ml_drift::DataType::INT32);
  auto experts_remap_packed_op = ::ml_drift::CreateLinearizeMapOp(
      experts_remap.tensor_desc, experts_remap_packed.tensor_desc);
  experts_remap_packed_op->read_size_ =
      experts_remap_packed.tensor_desc.GetMemorySizeInBytes();
  builder.AddGpuOperation(
      {experts_remap, experts_count, experts_offsets}, {experts_remap_packed},
      std::move(experts_remap_packed_op), "linearize_experts_remap");

  auto storage = builder.default_storage();
  builder.SetDefaultStorage(::ml_drift::TensorStorageType::BUFFER);
  auto experts_params = builder.Concat({experts_count, experts_offsets},
                                       ::ml_drift::Axis::CHANNELS);
  builder.SetDefaultStorage(storage);

  return {experts_remap, experts_params, experts_remap_packed};
}

::ml_drift::GpuModelBuilder::TensorHandle ExpertsRemapTo(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts) {
  const ::ml_drift::BHWC& src_shape = src.tensor_desc.GetBHWCShape();
  auto remapped_src = builder.AddTensor(
      ::ml_drift::BHWC(src_shape.b, 1, src_shape.w * num_active_experts,
                       src_shape.c),
      src.tensor_desc.GetDataType());
  auto remap_to = ::ml_drift::CreateExpertsRemapToOp(
      src.tensor_desc, experts_remap.tensor_desc, remapped_src.tensor_desc);
  builder.AddGpuOperation({src, experts_remap}, {remapped_src},
                          std::move(remap_to), "experts_remap_to");
  return remapped_src;
}

::ml_drift::GpuModelBuilder::TensorHandle ExpertsRemapFrom(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts) {
  const auto shape = src.tensor_desc.GetBHWCShape();
  auto remapped_dst = builder.AddTensor(
      ::ml_drift::BHWC(shape.b, num_active_experts,
                       shape.w / num_active_experts, shape.c),
      src.tensor_desc.GetDataType());
  auto remap_from = ::ml_drift::CreateExpertsRemapFromOp(
      src.tensor_desc, experts_remap.tensor_desc, remapped_dst.tensor_desc);
  builder.AddGpuOperation({src, experts_remap}, {remapped_dst},
                          std::move(remap_from), "experts_remap_from");
  return remapped_dst;
}

absl::StatusOr<::ml_drift::GpuModelBuilder::TensorHandle>
MakeConvWithPackedGroups(
    ::ml_drift::GpuModelBuilder& builder,
    const ::ml_drift::GpuModelBuilder::TensorHandle& src,
    const ::ml_drift::GpuModelBuilder::TensorHandle& params,
    const ::ml_drift::GpuModelBuilder::Weights& weights,
    int num_active_experts) {
  const auto precision =
      builder.GetConvPrecision(src.tensor_desc.GetDataType());
  ::ml_drift::BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  auto dst = builder.AddTensor(dst_shape, src.tensor_desc.GetDataType());

  ::ml_drift::OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  ::ml_drift::ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = weights.shape.h;
  packed_groups.max_group_size =
      src.tensor_desc.GetBHWCShape().w / num_active_experts;
  int average_task_size = ::ml_drift::DivideRoundUp(
      src.tensor_desc.GetBHWCShape().w, packed_groups.num_groups);
  ::ml_drift::ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;

  std::unique_ptr<::ml_drift::GPUOperation> conv;
  ::ml_drift::ExternalWeights external_weights;
  external_weights.desc = weights.desc;
  external_weights.shape = weights.shape;
  if (weights.scale) {
    external_weights.scale_zp_shape = weights.scale_zp_shape;
    external_weights.scale = &(weights.scale->tensor_desc);
  }
  if (weights.zero_point) {
    external_weights.zero_point = &(weights.zero_point->tensor_desc);
  }

  ::ml_drift::Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = ::ml_drift::HW(0, 0);
  conv_attr.padding.appended = ::ml_drift::HW(0, 0);
  conv_attr.strides = ::ml_drift::HW(1, 1);
  conv_attr.dilations = ::ml_drift::HW(1, 1);
  auto& conv_attr_weights = conv_attr.weights.emplace<
      ::ml_drift::Tensor<::ml_drift::OHWI, ::ml_drift::DataType::FLOAT32>>();
  conv_attr_weights.shape = weights.shape;

  std::vector<::ml_drift::GpuModelBuilder::TensorHandle> src_ids;
  src_ids.push_back(src);
  bool weights_conversion = false;
  if (average_task_size >= 32 &&
      ::ml_drift::SupportsConvAppleMPP(builder.gpu_info(), external_weights) &&
      precision == ::ml_drift::CalculationsPrecision::F16) {
    auto conv_apple_mpp = ::ml_drift::CreateConvAppleMPPExternalWeights(
        src.tensor_desc, dst.tensor_desc, external_weights,
        /*bias=*/nullptr, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true, runtime_check, &dst_shape);
    conv =
        std::make_unique<::ml_drift::ConvAppleMPP>(std::move(conv_apple_mpp));
  } else if (average_task_size >= 32 &&
             ::ml_drift::SupportsConvWaveMatrix(builder.gpu_info(), precision,
                                                external_weights)) {
    auto conv_wave_matrix = ::ml_drift::CreateConvWaveMatrixExternalWeights(
        op_def, precision, dst_shape, external_weights, builder.gpu_info(),
        /*bias=*/nullptr, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true, runtime_check);
    conv = std::make_unique<::ml_drift::ConvWaveMatrix>(
        std::move(conv_wave_matrix));
  } else if (average_task_size >= 32 &&
             ::ml_drift::SupportsConvGeneric(builder.gpu_info(), precision,
                                             external_weights)) {
    auto conv_generic = ::ml_drift::CreateConvGenericExternalWeights(
        builder.gpu_info(), op_def, precision, external_weights,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    conv = std::make_unique<::ml_drift::ConvGeneric>(std::move(conv_generic));
  } else if (average_task_size >= 64 &&
             ::ml_drift::IsConvWaveMemorySupported(builder.gpu_info())) {
    auto conv_wave_memory = ::ml_drift::CreateConvWaveMemoryExternalWeights(
        builder.gpu_info(), op_def, precision, conv_attr,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    auto conv_weights_desc = conv_wave_memory.GetWeightsDescription();
    conv = std::make_unique<::ml_drift::ConvWaveMemory>(
        std::move(conv_wave_memory));
    if (!(conv_weights_desc == weights.desc)) {
      weights_conversion = true;
      const ::ml_drift::GpuModelBuilder::TensorHandle* weights_scale =
          weights.scale ? &(*weights.scale) : nullptr;
      const ::ml_drift::GpuModelBuilder::TensorHandle* weights_zero_point =
          weights.zero_point ? &(*weights.zero_point) : nullptr;
      auto weights_tensors = builder.WeightsConversion(
          weights.weights, weights_scale, weights_zero_point, weights.desc,
          conv_weights_desc, weights.shape);
      src_ids.push_back(weights_tensors[0]);
    }
  } else {
    ABSL_ASSIGN_OR_RETURN(
        auto conv_fc,
        ::ml_drift::CreateFullyConnectedExternalWeights(
            builder.gpu_info(), precision, op_def.src_tensors[0],
            op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
            &dst_shape, /*src_exp=*/nullptr, runtime_check));
    conv = std::make_unique<::ml_drift::FullyConnected>(std::move(conv_fc));
  }
  if (!weights_conversion) {
    src_ids.push_back(weights.weights);
    if (weights.scale) {
      src_ids.push_back(*weights.scale);
    }
    if (weights.zero_point) {
      src_ids.push_back(*weights.zero_point);
    }
  }
  src_ids.push_back(params);
  conv->flops_ = dst_shape.DimensionsProduct() * weights.shape.i * 2;
  builder.AddGpuOperation(src_ids, {dst}, std::move(conv),
                          "conv_packed_groups_" +
                              ::ml_drift::ToString(weights.desc.type));
  return dst;
}

}  // namespace litert::ml_drift
