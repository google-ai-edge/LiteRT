// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/squeeze_op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::FloatNear;
using testing::Pointwise;

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

//   Input shape: {1, 2, 1, 3}, axes=[0, 2]
//   Output shape: {2, 3}
TEST_P(QnnModelTest, SqueezeRemovesMultipleDims) {
  static constexpr std::array<std::uint32_t, 4> kInputDims{1, 2, 1, 3};
  static constexpr std::array<std::uint32_t, 2> kOutputDims{2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {},
      {kInputDims.begin(), kInputDims.end()});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {},
      {kOutputDims.begin(), kOutputDims.end()});

  auto ops = ::qnn::BuildSqueezeOp(tensor_pool_, {input_tensor}, {output_tensor},
                                   {0, 2});
  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx,
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Verifies that Squeeze accepts quantized (int8) tensors
TEST_P(QnnModelTest, SqueezeQuantizedInt8) {
  static constexpr std::array<std::uint32_t, 2> kInputDims{1, 4};
  static constexpr std::array<std::uint32_t, 1> kOutputDims{4};
  const ::qnn::QuantizeParamsWrapperVariant kQuant{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>,
      /*scale=*/0.1f, /*zero_point=*/0};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_SFIXED_POINT_8, kQuant,
      {kInputDims.begin(), kInputDims.end()});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_SFIXED_POINT_8, kQuant,
      {kOutputDims.begin(), kOutputDims.end()});

  auto ops = ::qnn::BuildSqueezeOp(tensor_pool_, {input_tensor}, {output_tensor},
                                   {0});
  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  // Quantized values (scale=0.1, zero_point=0): {10, 20, 30, 40} → {1.0, 2.0,
  // 3.0, 4.0} in float.
  qnn_model_.SetInputData<std::int8_t>(input_idx, {10, 20, 30, 40});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<std::int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(testing::Eq(), std::vector<std::int8_t>{10, 20, 30, 40}));
}

// Verifies that negative squeeze axes (e.g. -4, -2 on a rank-4 tensor) are
// correctly resolved to positive indices inside BuildSqueezeOp.
//   Input shape: {1, 2, 1, 3}, axes=[-4, -2] → resolved=[0, 2]
//   Output shape: {2, 3}
TEST_P(QnnModelTest, SqueezeNegativeDims) {
  static constexpr std::array<std::uint32_t, 4> kInputDims{1, 2, 1, 3};
  static constexpr std::array<std::uint32_t, 2> kOutputDims{2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {},
      {kInputDims.begin(), kInputDims.end()});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {},
      {kOutputDims.begin(), kOutputDims.end()});

  auto ops = ::qnn::BuildSqueezeOp(tensor_pool_, {input_tensor}, {output_tensor},
                                   {-4, -2});
  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx,
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Verifies that empty squeeze_dims omits the axes param, causing QNN to squeeze
// all size-1 dimensions.
//   Input shape: {1, 2, 1, 3}, axes=[]
//   Output shape: {2, 3}
TEST_P(QnnModelTest, SqueezeEmptyDims) {
  static constexpr std::array<std::uint32_t, 4> kInputDims{1, 2, 1, 3};
  static constexpr std::array<std::uint32_t, 2> kOutputDims{2, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {},
      {kInputDims.begin(), kInputDims.end()});
  auto& output_tensor = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {},
      {kOutputDims.begin(), kOutputDims.end()});

  auto ops = ::qnn::BuildSqueezeOp(tensor_pool_, {input_tensor}, {output_tensor},
                                   {});
  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto output_idx = qnn_model_.AddOutputTensor(output_tensor);
  qnn_model_.SetInputData<float>(input_idx,
                                 {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

}  // namespace
}  // namespace litert::qnn
