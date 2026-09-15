// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

void TestBlockwiseMatmul(::qnn::QnnModel& qnn_model,
                         ::qnn::TensorPool& tensor_pool,
                         std::uint32_t bitwidth) {
  const std::string suffix = std::to_string(bitwidth);
  const std::vector<std::int8_t> weights(64, 1);
  const std::vector<float> scales(4, 0.5f);
  auto& input = tensor_pool.CreateInputTensorWithName(
      "input_" + suffix, QNN_DATATYPE_FLOAT_32, {}, {1, 1, 1, 32});
  auto& input_fp16 =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_16, {}, {1, 1, 1, 32});
  ::qnn::BwFloatBlockQuantizeParamsWrapper weight_quant(bitwidth, {1, 1, 16, 1},
                                                        scales);
  const auto packed_weights = PackLowBitWeights(bitwidth, weights);
  auto& weight = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, weight_quant, {1, 1, 32, 2},
      packed_weights.size(), packed_weights.data());
  auto& output_fp16 =
      tensor_pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_16, {}, {1, 1, 1, 2});
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output_" + suffix, QNN_DATATYPE_FLOAT_32, {}, {1, 1, 1, 2});

  qnn_model.MoveOpToGraph(::qnn::CreateCastOp(input, input_fp16));
  auto matmul_op =
      ::qnn::CreateMatmulOp(input_fp16, weight, output_fp16, false, false);
  const auto& qnn_quant =
      matmul_op.GetInputTensor(1).GetQnnTensor().v2.quantizeParams;
  EXPECT_EQ(qnn_quant.quantizationEncoding,
            QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
  EXPECT_EQ(qnn_quant.bwFloatBlockEncoding.bitwidth, bitwidth);
  qnn_model.MoveOpToGraph(std::move(matmul_op));
  qnn_model.MoveOpToGraph(::qnn::CreateCastOp(output_fp16, output));

  ASSERT_TRUE(qnn_model.ValidateOpConfig());
  ASSERT_TRUE(qnn_model.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "Execution requires an Android device with a Qualcomm HTP.";
#else
  const auto input_idx = qnn_model.AddInputTensor(input);
  const auto output_idx = qnn_model.AddOutputTensor(output);
  ASSERT_TRUE(
      qnn_model.SetInputData<float>(input_idx, std::vector<float>(32, 1.0f)));
  ASSERT_TRUE(qnn_model.Execute());
  const auto output_data = qnn_model.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              ::testing::Pointwise(::testing::FloatNear(1e-2f),
                                   std::vector<float>{16.0f, 16.0f}));
#endif
}

TEST_P(QnnModelTest, BlockwiseW2Fp16) {
  TestBlockwiseMatmul(qnn_model_, tensor_pool_, 2);
}

TEST_P(QnnModelTest, BlockwiseW4Fp16) {
  TestBlockwiseMatmul(qnn_model_, tensor_pool_, 4);
}

TEST_P(QnnModelTest, BlockwiseW8Fp16) {
  TestBlockwiseMatmul(qnn_model_, tensor_pool_, 8);
}

}  // namespace
}  // namespace litert::qnn
