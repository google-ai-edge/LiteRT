// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"

#include <vector>
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::vector<OpWrapper> BuildMatmulOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool adj_x,
    const bool adj_y) {
  std::vector<OpWrapper> res;
  // SA8295 workaournd
  if (inputs[0].get().GetRank() == 4 && inputs[1].get().GetRank() == 4 &&
      adj_x == 0 && adj_y == 1) {
    QNN_LOG_INFO("[MatMul SA8295 Workaournd]");
    // Reshape in[0]
    const std::vector<uint32_t> reshape_in0_dims = {
        1, inputs[0].get().GetDim(2), 1, inputs[0].get().GetDim(3)};
    auto& reshape_in0_out = tensor_pool.CreateNativeTensor(
        inputs[0].get().GetDataType(), inputs[0].get().GetQuantParams(),
        reshape_in0_dims);
    auto& reshape_in0 = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_in0.AddInputTensor(inputs[0]);
    reshape_in0.AddOutputTensor(reshape_in0_out);
    // Elementwise Mul
    const std::vector<uint32_t> elementwise_mul_dims = {
        1, inputs[0].get().GetDim(2), inputs[1].get().GetDim(2),
        inputs[0].get().GetDim(3)};
    auto& elementwise_mul_out = tensor_pool.CreateNativeTensor(
        outputs[0].get().GetDataType(), outputs[0].get().GetQuantParams(),
        elementwise_mul_dims);
    auto& elementwise_mul = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_MULTIPLY);
    elementwise_mul.AddInputTensor(reshape_in0_out);
    elementwise_mul.AddInputTensor(inputs[1]);
    elementwise_mul.AddOutputTensor(elementwise_mul_out);
    // Reduce Sum
    const std::vector<uint32_t> adjusted_axis_data = {3};
    auto& adjusted_axis_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, {},
        {static_cast<const std::uint32_t>(adjusted_axis_data.size())},
        sizeof(std::uint32_t) * adjusted_axis_data.size(),
        adjusted_axis_data.data());
    const std::vector<uint32_t> reduce_sum_dims = {
        1, inputs[0].get().GetDim(2), inputs[1].get().GetDim(2), 1};
    auto& reduce_sum_out = tensor_pool.CreateNativeTensor(
        outputs[0].get().GetDataType(), outputs[0].get().GetQuantParams(),
        reduce_sum_dims);
    auto& reduce_sum = CreateOpWrapper(res, QNN_OP_REDUCE_SUM);
    reduce_sum.AddInputTensor(elementwise_mul_out);
    reduce_sum.AddOutputTensor(reduce_sum_out);
    reduce_sum.AddTensorParam(QNN_OP_REDUCE_SUM_PARAM_AXES,
                              adjusted_axis_tensor);
    reduce_sum.AddScalarParam<bool>(QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS, true);
    // Reshape Out[0]
    auto& reshape_out0 = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_out0.AddInputTensor(reduce_sum_out);
    reshape_out0.AddOutputTensor(outputs[0]);
    return res;
  }
  QNN_LOG_INFO("[MatMul dims] In[0]");
  for (std::uint32_t i = 0; i < inputs[0].get().GetRank(); ++i) {
    QNN_LOG_INFO("Dim %u: %u", i, inputs[0].get().GetDim(i));
  }
  QNN_LOG_INFO("[MatMul dims] In[1]");
  for (std::uint32_t i = 0; i < inputs[1].get().GetRank(); ++i) {
    QNN_LOG_INFO("Dim %u: %u", i, inputs[1].get().GetDim(i));
  }
  auto& matmul_op = CreateOpWrapper(res, QNN_OP_MAT_MUL);
  for (const auto& input : inputs) {
    matmul_op.AddInputTensor(input);
  }
  matmul_op.AddOutputTensor(outputs[0]);
  matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, adj_x);
  matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, adj_y);

  return res;
}

}  // namespace qnn
