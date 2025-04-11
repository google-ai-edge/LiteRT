// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

constexpr size_t kZero = 0;
constexpr size_t kOne = 1;

namespace qnn {

std::vector<OpWrapper> BuildMatmulOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool adj_x,
    const bool adj_y) {
  std::vector<OpWrapper> res;

  auto dequant_input_0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, {}, inputs[kZero].get().GetDims());

  auto dequant_input_1 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, {}, inputs[kOne].get().GetDims());

  auto dequant_matmul_output = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, {}, outputs[kZero].get().GetDims());

  auto& dequant_op_0 = CreateOpWrapper(res, QNN_OP_DEQUANTIZE);
  dequant_op_0.AddInputTensor(inputs[kZero]);
  // dequant_op_0.AddInputTensor(temp_out);
  dequant_op_0.AddOutputTensor(dequant_input_0);

  auto& dequant_op_1 = CreateOpWrapper(res, QNN_OP_DEQUANTIZE);
  dequant_op_1.AddInputTensor(inputs[kOne]);
  dequant_op_1.AddOutputTensor(dequant_input_1);

  auto& matmul_op = CreateOpWrapper(res, QNN_OP_MAT_MUL);

  matmul_op.AddInputTensor(dequant_input_0);
  matmul_op.AddInputTensor(dequant_input_1);
  matmul_op.AddOutputTensor(dequant_matmul_output);

  matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, adj_x);
  matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, adj_y);

  auto& quantize_op = CreateOpWrapper(res, QNN_OP_QUANTIZE);
  quantize_op.AddInputTensor(dequant_matmul_output);
  quantize_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
