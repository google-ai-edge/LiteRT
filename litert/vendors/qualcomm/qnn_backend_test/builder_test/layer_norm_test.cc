// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/layer_norm_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, LayerNormQuantI16InsertsConvertOpsAndValidatesOnHtp) {
  static constexpr std::uint32_t kChannels = 4;
  static constexpr float kEpsilon = 1e-6f;
  static constexpr std::array<std::int16_t, kChannels> kGamma{100, 100, 100,
                                                              100};
  static constexpr std::array<std::int16_t, kChannels> kBeta{0, 0, 0, 0};
  const ::qnn::ScaleOffsetQuantizeParamsWrapper quant{0.01f, 0};
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

  ASSERT_EQ(ops.size(), 3u);
  EXPECT_TRUE(ops[0].IsOpCode(::qnn::QnnOpCode::kConvert));
  EXPECT_TRUE(ops[1].IsOpCode(::qnn::QnnOpCode::kLayerNorm));
  EXPECT_TRUE(ops[2].IsOpCode(::qnn::QnnOpCode::kConvert));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());
}

}  // namespace
}  // namespace litert::qnn
