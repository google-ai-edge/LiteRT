// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/graph_to_graph.h"

#include <gtest/gtest.h>

#include <cstring>

#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "QnnTypes.h"  // from @qairt

namespace qnn {
namespace {
TEST(MatMulConvertTest, Gemma3Prefill) {
  // G2G Test case: 2 MatMuls with 1 Convert
  //
  // ----- Before -----
  //   In1   In0   In2
  //    |  /     \  |
  //    | /       \ |
  // MatMul0     MatMul1
  //    |           |
  // Convert       Out1
  //    |
  //   Out0
  //
  // ----- After -----
  //   In1   In0   In2
  //    |  /     \  |
  //    | /       \ |
  // MatMul0     MatMul1
  //    |           |
  //   Out0        Out1
  //

  TensorPool tensor_pool;

  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});

  auto& matmul_to_concat = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});

  auto& input2 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 128, 256});
  auto& output1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                 quant_param, {1, 1, 512, 128});

  QuantizeParamsWrapperVariant convert_quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-3f, 0);
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, convert_quant_param, {1, 1, 512, 1280});

  std::vector<OpWrapper> op_wrappers;

  // MatMul0: In0, In1, matmul_to_concat
  std::vector<::qnn::TensorWrapperRef> matmul0_inputs;
  matmul0_inputs.emplace_back(input0);
  matmul0_inputs.emplace_back(input1);
  std::vector<::qnn::TensorWrapperRef> matmul0_outputs;
  matmul0_outputs.emplace_back(matmul_to_concat);
  auto matmul0 =
      BuildMatmulOp(tensor_pool, matmul0_inputs, matmul0_outputs, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // MatMul1: In1, In2, Out1
  std::vector<::qnn::TensorWrapperRef> matmul1_inputs;
  matmul1_inputs.emplace_back(input0);
  matmul1_inputs.emplace_back(input2);
  std::vector<::qnn::TensorWrapperRef> matmul1_outputs;
  matmul1_outputs.emplace_back(output1);
  auto matmul1 =
      BuildMatmulOp(tensor_pool, matmul1_inputs, matmul1_outputs, false, true);
  std::move(matmul1.begin(), matmul1.end(), std::back_inserter(op_wrappers));

  // Convert: matmul_to_concat, Out0
  std::vector<::qnn::TensorWrapperRef> convert_inputs;
  convert_inputs.emplace_back(matmul_to_concat);
  std::vector<::qnn::TensorWrapperRef> convert_outputs;
  convert_outputs.emplace_back(output0);
  auto convert = BuildQuantizeOp(tensor_pool, convert_inputs, convert_outputs);
  std::move(convert.begin(), convert.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 3);

  GraphToGraphTransform(op_wrappers);

  ASSERT_EQ(op_wrappers.size(), 2);
  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kMatMul), true);
  ASSERT_EQ(op_wrappers[1].IsOpCode(QnnOpCode::kMatMul), true);
}

TEST(MatMulConvertTest, Gemma3Decode) {
  // G2G Test case: 2 MatMuls with 1 Convert
  //
  // ----- Before -----
  //   In0 In1
  //    |  /
  //    | /
  // MatMul0
  //    |
  // Convert
  //    |
  //   Out0
  //
  // ----- After -----
  //   In0 In1
  //    |  /
  //    | /
  // MatMul0
  //    |
  //   Out0
  //

  TensorPool tensor_pool;

  QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-4f, 0);

  auto& input0 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 512, 256});
  auto& input1 = tensor_pool.CreateNativeTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant_param, {1, 1, 1280, 256});

  auto& matmul_to_concat = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, {1, 1, 512, 1280});

  QuantizeParamsWrapperVariant convert_quant_param;
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(1e-3f, 0);
  auto& output0 = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_SFIXED_POINT_16, convert_quant_param, {1, 1, 512, 1280});

  std::vector<OpWrapper> op_wrappers;

  // MatMul0: In0, In1, matmul_to_concat
  std::vector<::qnn::TensorWrapperRef> matmul0_inputs;
  matmul0_inputs.emplace_back(input0);
  matmul0_inputs.emplace_back(input1);
  std::vector<::qnn::TensorWrapperRef> matmul0_outputs;
  matmul0_outputs.emplace_back(matmul_to_concat);
  auto matmul0 =
      BuildMatmulOp(tensor_pool, matmul0_inputs, matmul0_outputs, false, true);
  std::move(matmul0.begin(), matmul0.end(), std::back_inserter(op_wrappers));

  // Convert: matmul_to_concat, Out0
  std::vector<::qnn::TensorWrapperRef> convert_inputs;
  convert_inputs.emplace_back(matmul_to_concat);
  std::vector<::qnn::TensorWrapperRef> convert_outputs;
  convert_outputs.emplace_back(output0);
  auto convert = BuildQuantizeOp(tensor_pool, convert_inputs, convert_outputs);
  std::move(convert.begin(), convert.end(), std::back_inserter(op_wrappers));

  ASSERT_EQ(op_wrappers.size(), 2);

  GraphToGraphTransform(op_wrappers);

  ASSERT_EQ(op_wrappers.size(), 1);
  ASSERT_EQ(op_wrappers[0].IsOpCode(QnnOpCode::kMatMul), true);
}
}  // namespace
}  // namespace qnn
