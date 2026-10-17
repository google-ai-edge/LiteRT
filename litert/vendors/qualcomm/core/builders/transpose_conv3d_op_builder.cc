// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/transpose_conv3d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "QnnOpDef.h"         // from @qairt
#include "QnnTypes.h"         // from @qairt
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kFilterIndex = 1;
constexpr size_t kInputIndex = 2;
constexpr size_t kBiasIndex = 3;
constexpr size_t kNumInputsBias = 4;
constexpr size_t kOutputIndex = 0;
constexpr size_t kSpatialRank = 3;
constexpr size_t kTensorRank = 5;
constexpr size_t kDepthIndex = 1;
constexpr size_t kHeightIndex = 2;
constexpr size_t kWidthIndex = 3;
constexpr size_t kFilterDepthIndex = 0;
constexpr size_t kFilterHeightIndex = 1;
constexpr size_t kFilterWidthIndex = 2;
constexpr size_t kFilterChannelOutIndex = 3;
constexpr size_t kFilterChannelInIndex = 4;
}  // namespace

std::vector<OpWrapper> BuildTransposeConv3dOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, std::uint32_t stride_d,
    std::uint32_t stride_h, std::uint32_t stride_w, std::uint32_t dilation_d,
    std::uint32_t dilation_h, std::uint32_t dilation_w,
    PaddingType padding_type) {
  std::vector<OpWrapper> ops;
  if (inputs.size() <= kInputIndex || outputs.empty()) {
    QNN_LOG_ERROR("TransposeConv3d requires at least 3 inputs and 1 output.");
    return {};
  }

  TensorWrapper& input_tensor = inputs[kInputIndex];
  TensorWrapper& filter_tensor = inputs[kFilterIndex];
  TensorWrapper& output_tensor = outputs[kOutputIndex];

  if (input_tensor.GetRank() != kTensorRank ||
      filter_tensor.GetRank() != kTensorRank ||
      output_tensor.GetRank() != kTensorRank) {
    QNN_LOG_ERROR("TransposeConv3d requires rank 5 input, filter and output.");
    return {};
  }

  if (!filter_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("TransposeConv3d requires a static filter.");
    return {};
  }

  const std::vector<std::uint32_t>& filter_dims = filter_tensor.GetDimensions();
  const std::vector<std::uint32_t> qnn_filter_dims{
      filter_dims[kFilterDepthIndex], filter_dims[kFilterHeightIndex],
      filter_dims[kFilterWidthIndex], filter_dims[kFilterChannelInIndex],
      filter_dims[kFilterChannelOutIndex]};
  QuantizeParamsWrapperVariant qnn_filter_quant_params =
      filter_tensor.GetQuantParams();
  if (auto* axis_quant_params =
          std::get_if<AxisScaleOffsetQuantizeParamsWrapper>(
              &qnn_filter_quant_params)) {
    const std::array<std::int32_t, kTensorRank> new_axis{
        kFilterDepthIndex, kFilterHeightIndex, kFilterWidthIndex,
        kFilterChannelInIndex, kFilterChannelOutIndex};
    axis_quant_params->SetAxis(new_axis[axis_quant_params->GetAxis()]);
  }
  TensorWrapper& transposed_filter_tensor = tensor_pool.CreateNativeTensor(
      filter_tensor.GetDataType(), qnn_filter_quant_params, qnn_filter_dims);
  const std::array<std::uint32_t, kTensorRank> permute_data{
      kFilterDepthIndex, kFilterHeightIndex, kFilterWidthIndex,
      kFilterChannelInIndex, kFilterChannelOutIndex};
  const std::vector<std::uint32_t> permute_shape{kTensorRank};
  TensorWrapper& permute_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, permute_shape,
      sizeof(permute_data[0]) * permute_data.size(), permute_data.data());
  ops.emplace_back(CreateTransposeOp(filter_tensor, transposed_filter_tensor,
                                     permute_tensor));

  // stride param
  const std::array<std::uint32_t, kSpatialRank> stride_data{stride_d, stride_h,
                                                            stride_w};
  const std::vector<std::uint32_t> stride_shape{kSpatialRank};
  TensorWrapper& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(stride_data[0]) * stride_data.size(), stride_data.data());

  const std::array<std::uint32_t, kSpatialRank> dilation_data{
      dilation_d, dilation_h, dilation_w};
  const std::vector<std::uint32_t> dilation_shape{kSpatialRank};
  TensorWrapper& dilation_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, dilation_shape,
      sizeof(dilation_data[0]) * dilation_data.size(), dilation_data.data());

  // padding param
  const auto [padding_before_depth, padding_after_depth] =
      ComputePaddingBeforeAfter(output_tensor.GetDimension(kDepthIndex),
                                qnn_filter_dims[kFilterDepthIndex], stride_d,
                                dilation_data[kFilterDepthIndex], padding_type);
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(output_tensor.GetDimension(kHeightIndex),
                                qnn_filter_dims[kFilterHeightIndex], stride_h,
                                dilation_data[kFilterHeightIndex], padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(output_tensor.GetDimension(kWidthIndex),
                                qnn_filter_dims[kFilterWidthIndex], stride_w,
                                dilation_data[kFilterWidthIndex], padding_type);
  const std::array<std::uint32_t, kSpatialRank * 2> padding_data{
      padding_before_depth, padding_after_depth,  padding_before_height,
      padding_after_height, padding_before_width, padding_after_width};
  const std::vector<std::uint32_t> padding_shape{kSpatialRank, 2};
  TensorWrapper& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(padding_data[0]) * padding_data.size(), padding_data.data());

  TensorWrapper* bias_tensor = nullptr;
  if (inputs.size() >= kNumInputsBias) {
    bias_tensor = &(inputs[kBiasIndex].get());
  }

  ops.emplace_back(CreateTransposeConv3dOp(
      input_tensor, transposed_filter_tensor, bias_tensor, output_tensor,
      stride_tensor, padding_tensor, dilation_tensor));
  return ops;
}

OpWrapper CreateTransposeConv3dOp(const TensorWrapper& input,
                                  const TensorWrapper& filter,
                                  const TensorWrapper* bias,
                                  const TensorWrapper& output,
                                  const TensorWrapper& stride,
                                  const TensorWrapper& pad_amount,
                                  const TensorWrapper& dilation) {
  OpWrapper op(GetUniqueOpName(QNN_OP_TRANSPOSE_CONV_3D),
               QNN_OP_TRANSPOSE_CONV_3D, QnnOpCode::kTransposeConv3d);
  op.AddInputTensor(input);
  op.AddInputTensor(filter);
  if (bias != nullptr) {
    op.AddInputTensor(*bias);
  }
  op.AddOutputTensor(output);
  op.AddTensorParam(QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE, stride);
  op.AddTensorParam(QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT, pad_amount);
  op.AddTensorParam(QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION, dilation);
  return op;
}

}  // namespace qnn
