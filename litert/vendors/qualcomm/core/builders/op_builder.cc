// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/op_builder.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::pair<std::uint32_t, std::uint32_t> ComputePaddingBeforeAfter(
    const std::uint32_t input_size, const std::uint32_t filter_size,
    const std::uint32_t stride, const std::uint32_t dilation_rate,
    const PaddingType padding_type) {
  // padding_before, padding_after
  std::pair<std::uint32_t, std::uint32_t> result{0, 0};
  if (stride == 0) {
    QNN_LOG_ERROR("Stride is 0");
    return result;
  }

  std::uint32_t output_size{};
  std::uint32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;

  switch (padding_type) {
    case PaddingType::Same:
      output_size = (input_size + stride - 1) / stride;
      break;
    case PaddingType::Valid:
      output_size = (input_size + stride - effective_filter_size) / stride;
      break;
    default:  // PaddingType::Unknown
      QNN_LOG_ERROR("Unknown padding type");
      return result;
  }

  std::int32_t total_padding =
      (output_size - 1) * stride + effective_filter_size - input_size;
  total_padding = total_padding > 0 ? total_padding : 0;
  result.first = total_padding / 2;
  result.second = result.first + total_padding % 2;
  return result;
}

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type) {
  const auto op_count = ops.size();
  const auto name = "op_type_" + std::string(op_type) + "_op_count_" +
                    std::to_string(op_count);
  return ops.emplace_back(std::move(name), op_type);
}

OpWrapper& CreateSimpleActivationOp(std::vector<OpWrapper>& ops,
                                    const char* op_type,
                                    const TensorWrapper& input_tensor,
                                    const TensorWrapper& output_tensor) {
  auto& ret = CreateOpWrapper(ops, op_type);
  ret.AddInputTensor(input_tensor);
  ret.AddOutputTensor(output_tensor);
  return ret;
}

TensorWrapper& ReplaceOutputTensorForFusedActivation(
    TensorPool& tensor_pool, const uint32_t fused_activation_function,
    std::vector<TensorWrapperRef>& output_tensors) {
  if (fused_activation_function == FusedActivationNone) {
    return output_tensors[0];
  }

  if (output_tensors.size() != 1) {
    QNN_LOG_WARNING(
        "Fused activation function: %d is not None but the size of output "
        "tensors is not 1.",
        fused_activation_function);
  }

  TensorWrapper& activation_input =
      tensor_pool.CloneNativeTensorFrom(output_tensors[0]);
  TensorWrapper& activation_output = output_tensors[0].get();
  output_tensors[0] = TensorWrapperRef(activation_input);
  return activation_output;
}

void AddFusedActivationNode(std::vector<OpWrapper>& res,
                            const uint32_t fused_activation_function,
                            const TensorWrapper& input_tensor,
                            const TensorWrapper& output_tensor) {
  switch (fused_activation_function) {
    case FusedActivationNone: {
      break;
    }
    case FusedActivationRelu: {
      CreateSimpleActivationOp(res, QNN_OP_RELU, input_tensor, output_tensor);
      break;
    }
    case FusedActivationReluN1To1: {
      auto& activation_op = CreateOpWrapper(res, QNN_OP_RELU_MIN_MAX);
      activation_op.AddInputTensor(input_tensor);
      activation_op.AddOutputTensor(output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          -1);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          1);
      break;
    }
    case FusedActivationRelu6: {
      auto& activation_op = CreateOpWrapper(res, QNN_OP_RELU_MIN_MAX);
      activation_op.AddInputTensor(input_tensor);
      activation_op.AddOutputTensor(output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          0);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          6);
      break;
    }
    case FusedActivationTanh: {
      CreateSimpleActivationOp(res, QNN_OP_TANH, input_tensor, output_tensor);
      break;
    }
    default: {
      QNN_LOG_WARNING("Unsupported fused activation function: %d",
                      fused_activation_function);
      break;
    }
  }
}

}  // namespace qnn
