// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/layer_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, LayerNormQuantI16BypassesValidationAndExecutesAccurately) {
  static constexpr std::uint32_t kChannels = 4;
  static constexpr float kEpsilon = 1e-6f;
  static constexpr float kQuantScale = 0.01f;
  static constexpr std::array<std::int16_t, kChannels> kGamma{100, 100, 100,
                                                              100};
  static constexpr std::array<std::int16_t, kChannels> kBeta{0, 0, 0, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{kQuantScale, 0};
  const std::vector<std::uint32_t> kDims{1, 2, kChannels};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_16, quant, kDims);
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_16, quant, kDims);
  auto& gamma = tensor_pool_.CreateStaticTensor(QNN_DATATYPE_SFIXED_POINT_16,
                                                quant, {kChannels},
                                                sizeof(kGamma), kGamma.data());
  auto& beta =
      tensor_pool_.CreateStaticTensor(QNN_DATATYPE_SFIXED_POINT_16, quant,
                                      {kChannels}, sizeof(kBeta), kBeta.data());

  auto ops = ::qnn::BuildLayerNormOp(tensor_pool_, {input, gamma, beta},
                                     {output}, kEpsilon);

  ASSERT_EQ(ops.size(), 1u);
  ASSERT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kLayerNorm));
  EXPECT_EQ(qnn_manager_ptr_->ValidateOp(*qnn_backend_ptr_, ops[0]),
            kLiteRtStatusOk);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

  const auto input_idx = qnn_model_.AddInputTensor(input);
  const auto output_idx = qnn_model_.AddOutputTensor(output);
  ASSERT_TRUE(qnn_model_.SetInputData<std::int16_t>(
      input_idx, {100, 200, 300, 400, 400, 300, 200, 100}));

  ASSERT_TRUE(qnn_model_.Execute());

  const auto output_data = qnn_model_.GetOutputData<std::int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 8u);
  std::vector<float> dequantized_output;
  ::qnn::DequantizeInto(output_data.value(), quant.GetScale(),
                        quant.GetZeroPoint(), dequantized_output);
  EXPECT_THAT(dequantized_output,
              Pointwise(FloatNear(kQuantScale / 2),
                        {-1.34164f, -0.44721f, 0.44721f, 1.34164f, 1.34164f,
                         0.44721f, -0.44721f, -1.34164f}));
}

}  // namespace
}  // namespace litert::qnn
