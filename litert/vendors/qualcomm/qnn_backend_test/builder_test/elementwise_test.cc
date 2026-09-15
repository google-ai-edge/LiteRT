// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "absl/types/span.h"  // from @com_google_absl
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/gelu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/logistic_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu6_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_0to1_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_n1to1_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/tanh_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {
using testing::ElementsAre;       // NOLINT
using testing::ElementsAreArray;  // NOLINT
using testing::FloatNear;         // NOLINT
using testing::Pointwise;         // NOLINT

using UnaryBuilder = std::function<std::vector<::qnn::OpWrapper>(
    ::qnn::TensorPool&, const std::vector<::qnn::TensorWrapperRef>&,
    const std::vector<::qnn::TensorWrapperRef>&)>;

template <typename InputType, typename OutputType>
void RunUnaryTest(
    ::qnn::QnnModel& qnn_model, ::qnn::TensorPool& tensor_pool,
    UnaryBuilder builder, ::qnn::QnnOpCode expected_op_code,
    Qnn_DataType_t input_data_type, Qnn_DataType_t output_data_type,
    absl::Span<const InputType> input_values,
    absl::Span<const OutputType> expected_values, float tolerance = 1e-3f,
    const ::qnn::QuantizeParamsWrapperVariant& input_quant = {},
    const ::qnn::QuantizeParamsWrapperVariant& output_quant = {}) {
  const std::vector<std::uint32_t> kDims{
      1, 1, static_cast<std::uint32_t>(input_values.size()), 1};
  auto& input = tensor_pool.CreateInputTensorWithName("input", input_data_type,
                                                      input_quant, kDims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", output_data_type, output_quant, kDims);

  auto ops = builder(tensor_pool, {input}, {output});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), expected_op_code);
  qnn_model.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model.ValidateOpConfig());
  ASSERT_TRUE(qnn_model.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The runtime portion of this test requires an Android HTP "
                  "target.";
#else
  const auto input_index = qnn_model.AddInputTensor(input);
  const auto output_index = qnn_model.AddOutputTensor(output);
  ASSERT_TRUE(qnn_model.SetInputData<InputType>(input_index, input_values));
  ASSERT_TRUE(qnn_model.Execute());

  const auto output_values = qnn_model.GetOutputData<OutputType>(output_index);
  ASSERT_TRUE(output_values);
  if constexpr (std::is_floating_point_v<OutputType>) {
    ASSERT_THAT(output_values.value(),
                Pointwise(FloatNear(tolerance), expected_values));
  } else {
    ASSERT_EQ(tolerance, 0.0f);
    ASSERT_THAT(output_values.value(), ElementsAreArray(expected_values));
  }
#endif
}

using BinaryBuilder = std::function<std::vector<::qnn::OpWrapper>(
    ::qnn::TensorPool&, const std::vector<::qnn::TensorWrapperRef>&,
    const std::vector<::qnn::TensorWrapperRef>&)>;

template <typename InputType, typename OutputType>
void RunBinaryTest(
    ::qnn::QnnModel& qnn_model, ::qnn::TensorPool& tensor_pool,
    BinaryBuilder builder, ::qnn::QnnOpCode expected_op_code,
    Qnn_DataType_t input_data_type, Qnn_DataType_t output_data_type,
    absl::Span<const InputType> input_0_values,
    absl::Span<const InputType> input_1_values,
    absl::Span<const OutputType> expected_values, float tolerance = 1e-3f,
    const ::qnn::QuantizeParamsWrapperVariant& input_quant = {},
    const ::qnn::QuantizeParamsWrapperVariant& output_quant = {}) {
  ASSERT_EQ(input_0_values.size(), input_1_values.size());
  const std::vector<std::uint32_t> kDims{
      1, 1, static_cast<std::uint32_t>(input_0_values.size()), 1};
  auto& input_0 = tensor_pool.CreateInputTensorWithName(
      "input_0", input_data_type, input_quant, kDims);
  auto& input_1 = tensor_pool.CreateInputTensorWithName(
      "input_1", input_data_type, input_quant, kDims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "output", output_data_type, output_quant, kDims);

  auto ops = builder(tensor_pool, {input_0, input_1}, {output});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), expected_op_code);
  qnn_model.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model.ValidateOpConfig());
  ASSERT_TRUE(qnn_model.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The runtime portion of this test requires an Android HTP "
                  "target.";
#else
  const auto input_0_index = qnn_model.AddInputTensor(input_0);
  const auto input_1_index = qnn_model.AddInputTensor(input_1);
  const auto output_index = qnn_model.AddOutputTensor(output);
  ASSERT_TRUE(qnn_model.SetInputData<InputType>(input_0_index, input_0_values));
  ASSERT_TRUE(qnn_model.SetInputData<InputType>(input_1_index, input_1_values));
  ASSERT_TRUE(qnn_model.Execute());

  const auto output_values = qnn_model.GetOutputData<OutputType>(output_index);
  ASSERT_TRUE(output_values);
  if constexpr (std::is_floating_point_v<OutputType>) {
    ASSERT_THAT(output_values.value(),
                Pointwise(FloatNear(tolerance), expected_values));
  } else {
    ASSERT_EQ(tolerance, 0.0f);
    ASSERT_THAT(output_values.value(), ElementsAreArray(expected_values));
  }
#endif
}

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

std::vector<::qnn::OpWrapper> BuildAddForTest(
    ::qnn::TensorPool&, const std::vector<::qnn::TensorWrapperRef>& inputs,
    const std::vector<::qnn::TensorWrapperRef>& outputs) {
  return ::qnn::MakeVector(
      ::qnn::CreateElementWiseAddOp(inputs[0], inputs[1], outputs[0]));
}

TEST_P(QnnModelTest, ElementWiseAddInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, BuildAddForTest,
      ::qnn::QnnOpCode::kElementWiseAdd, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {-2.0f, 0.2f, 0.7f, 0.8f},
      {0.1f, 0.2f, 0.3f, 0.5f}, {-1.9f, 0.4f, 1.0f, 1.3f});
}

std::vector<::qnn::OpWrapper> BuildMultiplyForTest(
    ::qnn::TensorPool&, const std::vector<::qnn::TensorWrapperRef>& inputs,
    const std::vector<::qnn::TensorWrapperRef>& outputs) {
  return ::qnn::MakeVector(
      ::qnn::CreateElementWiseMulOp(inputs[0], inputs[1], outputs[0]));
}

TEST_P(QnnModelTest, ElementWiseMultiplyInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, BuildMultiplyForTest,
      ::qnn::QnnOpCode::kElementWiseMultiply, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {-0.8f, 0.2f, 0.9f, 0.7f},
      {0.6f, 0.4f, 0.9f, 0.8f}, {-0.48f, 0.08f, 0.81f, 0.56f});
}

TEST_P(QnnModelTest, ElementWiseSubtractInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSubOp,
      ::qnn::QnnOpCode::kElementWiseSubtract, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 2.0f, 3.0f, 4.0f}, {4.0f, 3.0f, 2.0f, 1.0f},
      {-3.0f, -1.0f, 1.0f, 3.0f});
}

TEST_P(QnnModelTest, ElementWiseDivideInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseDivOp,
      ::qnn::QnnOpCode::kElementWiseDivide, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {6.0f, 8.0f, 5.0f, -9.0f},
      {2.0f, 4.0f, 0.5f, 3.0f}, {3.0f, 2.0f, 10.0f, -3.0f});
}

TEST_P(QnnModelTest, ElementWiseMaximumInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseMaximumOp,
      ::qnn::QnnOpCode::kElementWiseMaximum, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 0.0f, 2.0f, 11.0f, 2.0f, 23.0f},
      {0.0f, 0.0f, 1.0f, 12.0f, 255.0f, 1.0f},
      {1.0f, 0.0f, 2.0f, 12.0f, 255.0f, 23.0f});
}

TEST_P(QnnModelTest, ElementWiseMinimumInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseMinimumOp,
      ::qnn::QnnOpCode::kElementWiseMinimum, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 0.0f, 2.0f, 11.0f, 2.0f, 23.0f},
      {0.0f, 0.0f, 1.0f, 12.0f, 255.0f, 1.0f},
      {0.0f, 0.0f, 1.0f, 11.0f, 2.0f, 1.0f});
}

TEST_P(QnnModelTest, ElementWisePowerInputOutput) {
  RunBinaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwisePowerOp,
      ::qnn::QnnOpCode::kElementWisePower, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 2.0f, 0.5f, -3.0f},
      {2.0f, 3.0f, 4.0f, 1.0f}, {1.0f, 8.0f, 0.0625f, -3.0f});
}

TEST_P(QnnModelTest, ElementWiseLogicalAndInputOutput) {
  RunBinaryTest<bool, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseAndOp,
      ::qnn::QnnOpCode::kElementWiseAnd, QNN_DATATYPE_BOOL_8,
      QNN_DATATYPE_BOOL_8, {true, true, false, false},
      {true, false, false, true}, {true, false, false, false}, 0.0f);
}

TEST_P(QnnModelTest, ElementWiseLogicalOrInputOutput) {
  RunBinaryTest<bool, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseOrOp,
      ::qnn::QnnOpCode::kElementWiseOr, QNN_DATATYPE_BOOL_8,
      QNN_DATATYPE_BOOL_8, {true, false, true, false},
      {true, true, false, false}, {true, true, true, false}, 0.0f);
}

std::vector<::qnn::OpWrapper> BuildNotForTest(
    ::qnn::TensorPool&, const std::vector<::qnn::TensorWrapperRef>& inputs,
    const std::vector<::qnn::TensorWrapperRef>& outputs) {
  return ::qnn::MakeVector(
      ::qnn::CreateElementWiseNotOp(inputs[0], outputs[0]));
}

TEST_P(QnnModelTest, ElementWiseLogicalNotInputOutput) {
  RunUnaryTest<bool, bool>(qnn_model_, tensor_pool_, BuildNotForTest,
                           ::qnn::QnnOpCode::kElementWiseNot,
                           QNN_DATATYPE_BOOL_8, QNN_DATATYPE_BOOL_8,
                           {true, false, true, false},
                           {false, true, false, true}, 0.0f);
}

TEST_P(QnnModelTest, ElementWiseCeilInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseCeilOp,
      ::qnn::QnnOpCode::kElementWiseCeil, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {8.5f, 0.0f}, {9.0f, 0.0f});
}

TEST_P(QnnModelTest, ElementWiseCosInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseCosOp,
      ::qnn::QnnOpCode::kElementWiseCos, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {0.0f, 3.1415926f, -3.1415926f, 1.0f},
      {1.0f, -1.0f, -1.0f, 0.54030f}, 1e-2f);
}

TEST_P(QnnModelTest, ElementWiseRsqrtInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseRsqrtOp,
      ::qnn::QnnOpCode::kElementWiseRsqrt, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 2.0f, 4.0f, 9.0f},
      {1.0f, 0.7071f, 0.5f, 0.33333f}, 1e-2f);
}

TEST_P(QnnModelTest, ElementWiseRoundInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseRoundOp,
      ::qnn::QnnOpCode::kElementWiseRound, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {8.1f, 0.0f, 3.9f, 4.2f, -3.1f, -4.2f},
      {8.0f, 0.0f, 4.0f, 4.0f, -3.0f, -4.0f});
}

TEST_P(QnnModelTest, ElementWiseSignInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSignOp,
      ::qnn::QnnOpCode::kElementWiseSign, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32,
      {0.0f, -7.0f, 6.0f, -5.0f, 4.0f, -3.0f, 2.0f, 1.0f},
      {0.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f});
}

TEST_P(QnnModelTest, ElementWiseSquareInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSquareOp,
      ::qnn::QnnOpCode::kElementWiseMultiply, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {1.0f, 2.0f, 0.5f, -3.0f},
      {1.0f, 4.0f, 0.25f, 9.0f});
}

TEST_P(QnnModelTest, ElementWiseEluInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseEluOp,
      ::qnn::QnnOpCode::kElu, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32,
      {0.0f, -6.0f, 2.0f, -4.0f, 3.0f, -2.0f, 10.0f, -0.1f},
      {0.0f, -0.997521f, 2.0f, -0.981684f, 3.0f, -0.864665f, 10.0f,
       -0.0951626f},
      1e-2f);
}

TEST_P(QnnModelTest, ElementWiseHardSwishInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseHardSwishOp,
      ::qnn::QnnOpCode::kHardSwish, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32, {-4.0f, -3.0f, 0.0f, 3.0f, 6.0f},
      {0.0f, 0.0f, 0.0f, 3.0f, 6.0f}, 1e-2f);
}

TEST_P(QnnModelTest, GeluInputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildGeluOp, ::qnn::QnnOpCode::kGelu,
      QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32,
      {0.0f, 1.0f, 3.0f, 1.0f, -1.0f, -2.0f},
      {0.0f, 0.841345f, 2.99595f, 0.841345f, -0.158655f, -0.0455003f},
      1e-2f);
}

TEST_P(QnnModelTest, Relu0To1InputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildRelu0To1Op,
      ::qnn::QnnOpCode::kReluMinMax, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32,
      {0.0f, -0.6f, 0.2f, -0.4f, 0.3f, -2.0f, 1.1f, -0.1f},
      {0.0f, 0.0f, 0.2f, 0.0f, 0.3f, 0.0f, 1.0f, 0.0f});
}

TEST_P(QnnModelTest, ReluN1To1InputOutput) {
  RunUnaryTest<float, float>(
      qnn_model_, tensor_pool_, ::qnn::BuildReluN1To1Op,
      ::qnn::QnnOpCode::kReluMinMax, QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_FLOAT_32,
      {0.0f, -0.6f, 0.2f, -0.4f, 0.3f, -2.0f, 1.1f, -0.1f},
      {0.0f, -0.6f, 0.2f, -0.4f, 0.3f, -1.0f, 1.0f, -0.1f});
}

TEST_P(QnnModelTest, Relu6InputOutput) {
  RunUnaryTest<float, float>(qnn_model_, tensor_pool_, ::qnn::BuildRelu6Op,
                             ::qnn::QnnOpCode::kReluMinMax,
                             QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32,
                             {-1.0f, 0.0f, 3.0f, 7.0f},
                             {0.0f, 0.0f, 3.0f, 6.0f});
}

TEST_P(QnnModelTest, LogisticInputOutput) {
  RunUnaryTest<float, float>(qnn_model_, tensor_pool_, ::qnn::BuildLogisticOp,
                             ::qnn::QnnOpCode::kSigmoid, QNN_DATATYPE_FLOAT_32,
                             QNN_DATATYPE_FLOAT_32, {-2.0f, 0.0f, 2.0f},
                             {0.119203f, 0.5f, 0.880797f}, 1e-2f);
}

TEST_P(QnnModelTest, TanhInputOutput) {
  RunUnaryTest<float, float>(qnn_model_, tensor_pool_, ::qnn::BuildTanhOp,
                             ::qnn::QnnOpCode::kTanh, QNN_DATATYPE_FLOAT_32,
                             QNN_DATATYPE_FLOAT_32, {-1.0f, 0.0f, 1.0f},
                             {-0.761594f, 0.0f, 0.761594f}, 1e-2f);
}

::qnn::QuantizeParamsWrapperVariant Int8Quantization(float scale) {
  return ::qnn::QuantizeParamsWrapperVariant(
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, scale, 0);
}

::qnn::QuantizeParamsWrapperVariant Int16Quantization(float scale) {
  return ::qnn::QuantizeParamsWrapperVariant(
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, scale, 0);
}

TEST_P(QnnModelTest, ElementWiseQuantizedMultiplyInt8InputOutput) {
  RunBinaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, BuildMultiplyForTest,
      ::qnn::QnnOpCode::kElementWiseMultiply, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {-1, 2, 3, -4}, {3, 3, -2, -2},
      {-3, 6, -6, 8}, 0.0f, Int8Quantization(1.0f),
      Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedSubtractInt16InputOutput) {
  RunBinaryTest<std::int16_t, std::int16_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSubOp,
      ::qnn::QnnOpCode::kElementWiseSubtract, QNN_DATATYPE_SFIXED_POINT_16,
      QNN_DATATYPE_SFIXED_POINT_16, {1, 2, 3, 4}, {4, 3, 2, 1},
      {-3, -1, 1, 3}, 0.0f, Int16Quantization(1.0f),
      Int16Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedRsqrtInt16InputOutput) {
  RunUnaryTest<std::int16_t, std::int16_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseRsqrtOp,
      ::qnn::QnnOpCode::kElementWiseRsqrt, QNN_DATATYPE_SFIXED_POINT_16,
      QNN_DATATYPE_SFIXED_POINT_16, {1, 4, 16}, {4, 2, 1},
      0.0f, Int16Quantization(1.0f), Int16Quantization(0.25f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedSignInt8InputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSignOp,
      ::qnn::QnnOpCode::kElementWiseSign, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {0, -7, 6, -5, 4, -3, 2, 1},
      {0, -1, 1, -1, 1, -1, 1, 1}, 0.0f, Int8Quantization(1.0f),
      Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedSinInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSinOp,
      ::qnn::QnnOpCode::kElementWiseSin, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {0, 1, -1}, {0, 1, -1},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedSqrtInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSqrtOp,
      ::qnn::QnnOpCode::kElementWiseSquareRoot, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {0, 1, 4, 9}, {0, 1, 2, 3},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedSquaredDifferenceInputOutput) {
  RunBinaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseSquaredDifferenceOp,
      ::qnn::QnnOpCode::kElementWiseSquaredDifference,
      QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8, {1, 2, 3, 4},
      {4, 3, 2, 1}, {9, 1, 1, 9}, 0.0f, Int8Quantization(1.0f),
      Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedLessInputOutput) {
  RunBinaryTest<std::int8_t, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseLessOp,
      ::qnn::QnnOpCode::kElementWiseLess, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_BOOL_8, {1, 2, 3, 4}, {4, 3, 2, 1},
      {true, true, false, false}, 0.0f, Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedGreaterInputOutput) {
  RunBinaryTest<std::int8_t, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseGreaterOp,
      ::qnn::QnnOpCode::kElementWiseGreater, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_BOOL_8, {1, 2, 3, 4}, {4, 3, 2, 1},
      {false, false, true, true}, 0.0f, Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedFloorInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseFloorOp,
      ::qnn::QnnOpCode::kElementWiseFloor, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {-2, -1, 0, 3}, {-2, -1, 0, 3},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedFloorDivInputOutput) {
  RunBinaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseFloorDivOp,
      ::qnn::QnnOpCode::kElementWiseFloorDiv, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {7, -7, 7, -7}, {3, 3, -3, -3},
      {2, -3, -3, 2}, 0.0f, Int8Quantization(1.0f),
      Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedLessEqualInputOutput) {
  RunBinaryTest<std::int8_t, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseLessEqualOp,
      ::qnn::QnnOpCode::kElementWiseLessEqual, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_BOOL_8, {1, 2, 3, 4}, {1, 3, 2, 4},
      {true, true, false, true}, 0.0f, Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedGreaterEqualInputOutput) {
  RunBinaryTest<std::int8_t, bool>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseGreaterEqualOp,
      ::qnn::QnnOpCode::kElementWiseGreaterEqual, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_BOOL_8, {1, 2, 3, 4}, {1, 3, 2, 4},
      {true, false, true, true}, 0.0f, Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedExpInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseExpOp,
      ::qnn::QnnOpCode::kElementWiseExp, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {0, 1, 2}, {1, 3, 7},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedLogInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseLogOp,
      ::qnn::QnnOpCode::kElementWiseLog, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {1, 3, 7}, {0, 1, 2},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

TEST_P(QnnModelTest, ElementWiseQuantizedAbsInputOutput) {
  RunUnaryTest<std::int8_t, std::int8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseAbsOp,
      ::qnn::QnnOpCode::kElementWiseAbs, QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8, {0, -1, 2, -3}, {0, 1, 2, 3},
      0.0f, Int8Quantization(1.0f), Int8Quantization(1.0f));
}

::qnn::QuantizeParamsWrapperVariant UInt8Quantization() {
  return ::qnn::QuantizeParamsWrapperVariant(
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 1.0f, 128);
}

TEST_P(QnnModelTest, ElementWiseQuantizedMultiplyUInt8InputOutput) {
  RunBinaryTest<std::uint8_t, std::uint8_t>(
      qnn_model_, tensor_pool_, BuildMultiplyForTest,
      ::qnn::QnnOpCode::kElementWiseMultiply, QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_UFIXED_POINT_8, {127, 130, 126, 129},
      {131, 131, 129, 124}, {125, 134, 126, 124},
      0.0f, UInt8Quantization(), UInt8Quantization());
}

TEST_P(QnnModelTest, ElementWiseQuantizedMaximumUInt8InputOutput) {
  RunBinaryTest<std::uint8_t, std::uint8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseMaximumOp,
      ::qnn::QnnOpCode::kElementWiseMaximum, QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_UFIXED_POINT_8, {1, 0, 2, 11, 2, 23},
      {0, 0, 1, 12, 255, 1}, {1, 0, 2, 12, 255, 23},
      0.0f, UInt8Quantization(), UInt8Quantization());
}

TEST_P(QnnModelTest, ElementWiseQuantizedMinimumUInt8InputOutput) {
  RunBinaryTest<std::uint8_t, std::uint8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseMinimumOp,
      ::qnn::QnnOpCode::kElementWiseMinimum, QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_UFIXED_POINT_8, {1, 0, 2, 11, 2, 23},
      {0, 0, 1, 12, 255, 1}, {0, 0, 1, 11, 2, 1},
      0.0f, UInt8Quantization(), UInt8Quantization());
}

TEST_P(QnnModelTest, ElementWiseQuantizedNegInputOutput) {
  RunUnaryTest<std::uint8_t, std::uint8_t>(
      qnn_model_, tensor_pool_, ::qnn::BuildElementwiseNegOp,
      ::qnn::QnnOpCode::kElementWiseNeg, QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_UFIXED_POINT_8, {128, 127, 130, 125}, {128, 129, 126, 131},
      0.0f, UInt8Quantization(), UInt8Quantization());
}

TEST_P(QnnModelTest, SingleElementWiseDivide) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  ::qnn::QuantizeParamsWrapperVariant quant_param_0{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000031f,
      0};
  ::qnn::QuantizeParamsWrapperVariant quant_param_1{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000101f,
      0};
  ::qnn::QuantizeParamsWrapperVariant quant_param_2{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.000030f,
      0};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param_0, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_16, quant_param_1, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param_2, kDims);
  auto ops = ::qnn::BuildElementwiseDivOp(tensor_pool_, {input_0, input_1},
                                          {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO (chunhsue-qti): Uncomment the line below when QAIRT 2.42 releases.
  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx = qnn_model_.AddInputTensor(input_0);
  auto input_idx1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx, {1, 1, 1, 1});
  qnn_model_.SetInputData<int16_t>(input_idx1, {1, 1, 1, 1});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  const float output_scale =
      std::get<::qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param_2)
          .GetScale();
  const std::int32_t output_zero_point =
      std::get<::qnn::ScaleOffsetQuantizeParamsWrapper>(quant_param_2)
          .GetZeroPoint();
  std::vector<float> dequant_output;
  ::qnn::DequantizeInto(output_data.value(), output_scale, output_zero_point,
                        dequant_output);
  ASSERT_THAT(dequant_output,
              Pointwise(FloatNear(1e-2), {0.306f, 0.306f, 0.306f, 0.306f}));
#endif
}

TEST_P(QnnModelTest, SingleElementWiseMax) {
  ::qnn::QuantizeParamsWrapperVariant quant_param{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.00015f, 0};

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims);
  auto ops = ::qnn::BuildElementwiseMaximumOp(tensor_pool_, {input_0, input_1},
                                              {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // TODO (chunhsue-qti): Uncomment the line below when QAIRT 2.42 releases.
  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx_0, {-20000, 0, 10000, 20000});
  qnn_model_.SetInputData<int16_t>(input_idx_1,
                                   {-17204, -17204, -17204, -17204});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  // Only check quant value in this test since this op is only a data mover.
  ASSERT_THAT(output_data.value(), ElementsAre(-17204, 0, 10000, 20000));
#endif
}

TEST_P(QnnModelTest, SingleElementWiseBinaryMulAsymmetricQuant) {
  const std::vector<std::uint32_t> kDims{1, 1, 1, 4};
  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.005f,
          -97},
      kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.09f,
          8},
      kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_SFIXED_POINT_8,
      ::qnn::QuantizeParamsWrapperVariant{
          std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.06f,
          -68},
      kDims);

  qnn_model_.MoveOpToGraph(
      ::qnn::CreateElementWiseMulOp(input_0, input_1, output_0));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int8_t>(input_idx_0, {-100, -50, 0, 50});
  qnn_model_.SetInputData<int8_t>(input_idx_1, {-100, -50, 0, 50});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int8_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ElementsAre(-66, -88, -74, -22));
#endif
}

// FloorMod follows the sign of the divisor.
TEST_P(QnnModelTest, FloorModNegativeValue) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_INT_32, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_INT_32, {}, kDims);
  auto ops = ::qnn::BuildElementwiseFloorModOp(tensor_pool_, {input_0, input_1},
                                               {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else

  auto input_idx_0 = qnn_model_.AddInputTensor(input_0);
  auto input_idx_1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int32_t>(input_idx_0, {10, -9, -11, 7});
  qnn_model_.SetInputData<int32_t>(input_idx_1, {2, 2, -3, -4});

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int32_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), ElementsAre(0, 1, -2, -1));
#endif
}

// HTP only supports QNN_OP_ELEMENT_WISE_MOD on INT32
TEST_P(QnnModelTest, FloorModRejectsNonInt32) {
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};

  auto& input_0 = tensor_pool_.CreateInputTensorWithName(
      "in_0", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& input_1 = tensor_pool_.CreateInputTensorWithName(
      "in_1", QNN_DATATYPE_FLOAT_32, {}, kDims);
  auto& output_0 = tensor_pool_.CreateOutputTensorWithName(
      "out_0", QNN_DATATYPE_FLOAT_32, {}, kDims);

  auto ops = ::qnn::BuildElementwiseFloorModOp(tensor_pool_, {input_0, input_1},
                                               {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_FALSE(qnn_model_.ValidateOpConfig());
}
}  // namespace
}  // namespace litert::qnn
