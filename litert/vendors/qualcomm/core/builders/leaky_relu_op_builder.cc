// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/builders/leaky_relu_op_builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
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
constexpr size_t kOutputIndex = 0;

}  // namespace
std::vector<OpWrapper> BuildLeakyReluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float alpha) {
  std::vector<OpWrapper> res;

  OpWrapper& leaky_relu_op = CreateOpWrapper(res, QNN_OP_PRELU);
  TensorWrapper& input_tensor = inputs[kInputIndex];
  leaky_relu_op.AddInputTensor(input_tensor);
  leaky_relu_op.AddOutputTensor(outputs[kOutputIndex]);
  TensorWrapper* alpha_tensor = nullptr;
  if (std::holds_alternative<UndefinedQuantizeParamsWrapper>(
          input_tensor.GetQuantParams())) {
    alpha_tensor = tensor_pool.CreateStaticTensorWithValue(
        input_tensor.GetDataType(), input_tensor.GetQuantParams(), {1}, alpha);
  } else if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
                 input_tensor.GetQuantParams())) {
    QuantizeParamsWrapperVariant quant_param;
    quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(std::max(alpha, 0.0f),
                                                          0);

    switch (input_tensor.GetDataType()) {
      case QNN_DATATYPE_UFIXED_POINT_8:
      case QNN_DATATYPE_SFIXED_POINT_8:
      case QNN_DATATYPE_UFIXED_POINT_16:
      case QNN_DATATYPE_SFIXED_POINT_16: {
        alpha_tensor = tensor_pool.CreateStaticTensorWithValue(
            input_tensor.GetDataType(), quant_param, {1}, alpha);
        break;
      }
      default:
        QNN_LOG_ERROR(
            "Unsupported QNN data type when creating LeakyRelu alpha tensor in "
            "per-tensor quantization.");
        return {};
    }
  } else {
    QNN_LOG_ERROR(
        "Unsupported quantization type when creating LeakyRelu alpha tensor.");
    return {};
  }
  if (alpha_tensor == nullptr) {
    QNN_LOG_ERROR("Failed to create alpha tensor for LeakyRelu op.");
    return res;
  }
  leaky_relu_op.AddInputTensor(*alpha_tensor);
  return res;
}

}  // namespace qnn
