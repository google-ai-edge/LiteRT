// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/conv3d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kBiasIndex = 2;
constexpr size_t kNumInputsBias = 3;
constexpr size_t kOutputIndex = 0;
constexpr size_t kDepthIndex = 1;
constexpr size_t kHeightIndex = 2;
constexpr size_t kWidthIndex = 3;
constexpr size_t kFilterDepthIndex = 0;
constexpr size_t kFilterHeightIndex = 1;
constexpr size_t kFilterWidthIndex = 2;

}  // namespace

std::vector<OpWrapper> BuildConv3dOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::uint32_t stride_d,
    const std::uint32_t stride_h, const std::uint32_t stride_w,
    const std::uint32_t dilation_d, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const PaddingType padding_type) {
  std::vector<OpWrapper> res;

  // conv
  OpWrapper& conv_op = CreateOpWrapper(res, QNN_OP_CONV_3D);
  TensorWrapper& input_tensor = inputs[kInputIndex];
  TensorWrapper& filter_tensor = inputs[kFilterIndex];
  conv_op.AddInputTensor(input_tensor);
  conv_op.AddInputTensor(filter_tensor);
  if (inputs.size() == kNumInputsBias) {
    conv_op.AddInputTensor(inputs[kBiasIndex]);
  }
  conv_op.AddOutputTensor(outputs[kOutputIndex]);

  // stride param
  const std::array<std::uint32_t, 3> stride_data{stride_d, stride_h, stride_w};
  const std::vector<std::uint32_t> stride_shape{3};
  auto& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(stride_data[0]) * stride_data.size(), stride_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_3D_PARAM_STRIDE, stride_tensor);

  // dilation param
  const std::array<std::uint32_t, 3> dilation_data{dilation_d, dilation_h,
                                                   dilation_w};
  const std::vector<std::uint32_t> dilation_shape{3};
  auto& dilation_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, dilation_shape,
      sizeof(dilation_data[0]) * dilation_data.size(), dilation_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_3D_PARAM_DILATION, dilation_tensor);

  // padding param
  const auto [padding_before_depth, padding_after_depth] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kDepthIndex),
                                filter_tensor.GetDim(kFilterDepthIndex),
                                stride_d, dilation_d, padding_type);
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kHeightIndex),
                                filter_tensor.GetDim(kFilterHeightIndex),
                                stride_h, dilation_h, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kWidthIndex),
                                filter_tensor.GetDim(kFilterWidthIndex),
                                stride_w, dilation_w, padding_type);
  const std::array<std::uint32_t, 6> padding_data = {
      padding_before_depth, padding_after_depth,  padding_before_height,
      padding_after_height, padding_before_width, padding_after_width};
  const std::vector<std::uint32_t> padding_shape{3, 2};
  auto& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(padding_data[0]) * padding_data.size(), padding_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_3D_PARAM_PAD_AMOUNT, padding_tensor);

  const std::vector<uint32_t>& input_dims = input_tensor.GetDims();
  const std::vector<uint32_t>& filters_dims = filter_tensor.GetDims();

  // group param
  const std::uint32_t input_channel = input_dims[4];
  const std::uint32_t filter_input_channel = filters_dims[3];
  std::uint32_t groups = input_channel / filter_input_channel;
  // TODO: Runtime not supports group yet. However, keep this for future.
  if (groups > 1) {
    // Remove this once conv3d supports group other than default value
    QNN_LOG_WARNING("Conv3d only supports group==1.");
    conv_op.AddScalarParam<std::uint32_t>(QNN_OP_CONV_3D_PARAM_GROUP,
                                          QNN_DATATYPE_UINT_32, groups);
  }
  if (input_channel % filter_input_channel != 0) {
    QNN_LOG_WARNING(
        "Filter input channel cannot be a factor of channels of "
        "input (grouped conv) or equals (normal conv). Input "
        "channel is %d and filter input channel is %d.",
        input_channel, filter_input_channel);
  }

  return res;
}

}  // namespace qnn
