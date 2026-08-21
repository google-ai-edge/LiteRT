// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <variant>
#include <vector>

#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr int kBiasIdx = 2;

std::vector<OpWrapper> BuildFullyConnectedOpImp(
    TensorPool& tensor_pool, const TensorWrapper& input_tensor,
    const TensorWrapper& weight_tensor, const TensorWrapper* bias_tensor,
    const TensorWrapper& output_tensor, bool keep_num_dims) {
  std::vector<OpWrapper> res;

  OpWrapper& fully_connected_op = CreateOpWrapper(res, QNN_OP_FULLY_CONNECTED);
  fully_connected_op.AddInputTensor(input_tensor);
  fully_connected_op.AddInputTensor(weight_tensor);
  if (bias_tensor != nullptr) {
    fully_connected_op.AddInputTensor(*bias_tensor);
  }

  if (keep_num_dims) {
    auto& input_dims = input_tensor.GetDimensions();
    std::uint32_t input_size = std::accumulate(
        input_dims.begin(), input_dims.end(), 1, std::multiplies<>());
    const std::uint32_t num_units = weight_tensor.GetDimension(0);
    const std::uint32_t num_input_elem = weight_tensor.GetDimension(1);

    // input_size must be divisible by num_input_elem. This should be validated
    // by QNN.
    const std::uint32_t batch_size = input_size / num_input_elem;
    // QNN output should always be rank 2
    TensorWrapper& fully_connected_out = tensor_pool.CloneNativeTensorFrom(
        output_tensor, {batch_size, num_units});
    fully_connected_op.AddOutputTensor(fully_connected_out);

    OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op.AddInputTensor(fully_connected_out);
    reshape_op.AddOutputTensor(output_tensor);
  } else {
    fully_connected_op.AddOutputTensor(output_tensor);
  }

  return res;
}

std::vector<OpWrapper> BuildFp16FullyConnectedOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_num_dims) {
  std::vector<OpWrapper> res;

  constexpr std::int32_t kWeightChannelAxis = 0;
  TensorWrapper& weight_tensor = inputs[1];
  if (weight_tensor.IsPerTensorQuant()) {
    const auto& weight_quant = std::get<ScaleOffsetQuantizeParamsWrapper>(
        weight_tensor.GetQuantParams());
    QuantizeParamsWrapperVariant per_channel_quant{
        std::in_place_type<AxisScaleOffsetQuantizeParamsWrapper>,
        kWeightChannelAxis, weight_tensor.GetDimension(kWeightChannelAxis),
        weight_quant.GetScale(), weight_quant.GetZeroPoint()};
    weight_tensor.SetQuantParams(per_channel_quant);
  } else if (!weight_tensor.IsPerChannelQuant()) {
    QNN_LOG_ERROR(
        "FP16 FullyConnected with int8 weight only supports per-channel "
        "quantized weight.");
    return {};
  }

  const TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& input_f16 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
      input_tensor.GetDimensions());
  res.emplace_back(CreateCastOp(input_tensor, input_f16));

  const TensorWrapper* bias_f16 = nullptr;
  if (kBiasIdx < inputs.size()) {
    const TensorWrapper& bias_tensor = inputs[kBiasIdx];
    TensorWrapper& bias_f16_tensor = tensor_pool.CreateNativeTensor(
        QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
        bias_tensor.GetDimensions());
    res.emplace_back(CreateCastOp(bias_tensor, bias_f16_tensor));
    bias_f16 = &bias_f16_tensor;
  }

  const TensorWrapper& output_tensor = outputs[0];
  TensorWrapper& output_f16 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_16, QuantizeParamsWrapperVariant{},
      output_tensor.GetDimensions());

  auto fc_ops = BuildFullyConnectedOpImp(tensor_pool, input_f16, weight_tensor,
                                         bias_f16, output_f16, keep_num_dims);
  std::move(fc_ops.begin(), fc_ops.end(), std::back_inserter(res));

  res.emplace_back(CreateCastOp(output_f16, output_tensor));
  return res;
}
}  // namespace

std::vector<OpWrapper> BuildFullyConnectedOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, bool keep_num_dims,
    bool use_int64_bias_as_int32, SdkVersion sdk_version) {
  if (inputs.size() < 2 || outputs.empty()) {
    return {};
  }

  const bool has_bias = kBiasIdx < inputs.size();
  if (inputs[0].get().IsF32() && inputs[1].get().IsQuantI8() &&
      outputs[0].get().IsF32() &&
      (!has_bias || inputs[kBiasIdx].get().IsF32())) {
    return BuildFp16FullyConnectedOp(tensor_pool, inputs, outputs,
                                     keep_num_dims);
  }

  const TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& weight_tensor = inputs[1];
  // Treat a8w2 as a8w4 if sdk version < 2.47.0.
  if (input_tensor.IsQuantI8() && weight_tensor.IsQuantI8() &&
      weight_tensor.IsQuantBitwidth(kQuantBitWidth2) &&
      sdk_version < SdkVersion{2, 47, 0}) {
    QNN_LOG_WARNING(
        "Aggressively convert the a8w2 Fully Connected Op to a8w4.");
    weight_tensor.SetQuantBitwidth(kQuantBitWidth4);
  }

  const TensorWrapper* bias_tensor = nullptr;
  if (has_bias) {
    bias_tensor = &inputs[kBiasIdx].get();
    if (use_int64_bias_as_int32 && bias_tensor->IsTensorStatic() &&
        bias_tensor->GetDataType() == QNN_DATATYPE_INT_64) {
      auto* converted_bias_tensor =
          tensor_pool.ConvertStaticTensorFrom<std::int32_t>(*bias_tensor);
      if (converted_bias_tensor == nullptr) {
        return {};
      }
      bias_tensor = converted_bias_tensor;
      QNN_LOG_WARNING(
          "Convert bias tensor in fully connected op from int64 to int32.");
    }
  }

  return BuildFullyConnectedOpImp(tensor_pool, input_tensor, weight_tensor,
                                  bias_tensor, outputs[0], keep_num_dims);
}

}  // namespace qnn
