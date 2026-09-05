// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/transpose_conv3d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "QnnOpDef.h"         // from @qairt
#include "QnnTypes.h"         // from @qairt
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
// TFLite passes the output shape as input 0; the activation is input 2.
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
    const std::vector<TensorWrapperRef>& outputs, const std::uint32_t stride_d,
    const std::uint32_t stride_h, const std::uint32_t stride_w,
    const std::uint32_t dilation_d, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type) {
  if (inputs.size() <= kInputIndex || outputs.empty()) {
    QNN_LOG_ERROR("TransposeConv3d got too few tensors.");
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

  // A runtime filter would need a runtime layout swap and dilation expansion,
  // which the compile-time lowering below cannot express; returning {} leaves
  // the op on CPU.
  if (!filter_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("TransposeConv3d requires a static filter.");
    return {};
  }

  const std::vector<std::uint32_t>& filter_dims = filter_tensor.GetDimensions();
  std::vector<std::uint32_t> qnn_filter_dims{
      filter_dims[kFilterDepthIndex], filter_dims[kFilterHeightIndex],
      filter_dims[kFilterWidthIndex], filter_dims[kFilterChannelInIndex],
      filter_dims[kFilterChannelOutIndex]};

  // The TFLite kernel accepts float filters only (conv3d_transpose.cc rejects
  // every other type in Prepare), so a float read covers all reachable models.
  auto filter_data = filter_tensor.GetTensorData<float>();
  if (!filter_data.has_value()) {
    QNN_LOG_ERROR("TransposeConv3d failed to read filter data.");
    return {};
  }
  std::vector<float> qnn_filter_data;
  TransposeFromDHWOIToDHWIO(filter_data.value(), filter_dims,
                             qnn_filter_data);

  const std::array<std::uint32_t, kSpatialRank> requested_dilation{
      dilation_d, dilation_h, dilation_w};
  const std::array<std::uint32_t, kSpatialRank> qnn_dilation{1, 1, 1};
  if (requested_dilation != qnn_dilation) {
    std::vector<std::uint32_t> dilated_filter_dims;
    std::vector<float> dilated_filter_data;
    DilateDHWIO(absl::Span<const float>(qnn_filter_data), qnn_filter_dims,
                requested_dilation, dilated_filter_dims, dilated_filter_data);
    qnn_filter_dims.swap(dilated_filter_dims);
    qnn_filter_data.swap(dilated_filter_data);
  }

  // Fold dilation into static weights so the emitted QNN operation always uses
  // unit dilation.
  TensorWrapper& transposed_filter_tensor = tensor_pool.CreateStaticTensor(
      filter_tensor.GetDataType(), filter_tensor.GetQuantParams(),
      qnn_filter_dims, sizeof(decltype(qnn_filter_data)::value_type) *
                           qnn_filter_data.size(),
      qnn_filter_data.data());

  // stride param
  const std::array<std::uint32_t, kSpatialRank> stride_data{stride_d, stride_h,
                                                            stride_w};
  const std::vector<std::uint32_t> stride_shape{kSpatialRank};
  TensorWrapper& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(stride_data[0]) * stride_data.size(), stride_data.data());

  const auto dilation_data = qnn_dilation;
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

  return MakeVector(CreateTransposeConv3dOp(
      input_tensor, transposed_filter_tensor, bias_tensor, output_tensor,
      stride_tensor, padding_tensor, dilation_tensor));
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
