// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/pad_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"
#include "QnnOpDef.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert::qnn {
namespace {

class PadOpBuilderQnnModelTest : public QnnModelTest {};

INSTANTIATE_TEST_SUITE_P(, PadOpBuilderQnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

constexpr std::uint32_t kInputHeight = 2;
constexpr std::uint32_t kInputWidth = 2;
constexpr std::uint32_t kOutputHeight = 4;
constexpr std::uint32_t kOutputWidth = 4;

std::vector<::qnn::OpWrapper> BuildFloatPadOp(::qnn::TensorPool& tensor_pool) {
  const std::vector<std::uint32_t> input_dims{1, kInputHeight, kInputWidth, 1};
  const std::vector<std::uint32_t> output_dims{1, kOutputHeight, kOutputWidth,
                                               1};
  const std::vector<std::uint32_t> pad_dims{4, 2};

  auto& input = tensor_pool.CreateInputTensorWithName(
      "in", QNN_DATATYPE_FLOAT_32, {}, input_dims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "out", QNN_DATATYPE_FLOAT_32, {}, output_dims);

  const std::vector<std::uint32_t> pad_data = {0, 0, 1, 1, 1, 1, 0, 0};
  auto& pad_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, pad_dims,
      pad_data.size() * sizeof(std::uint32_t), pad_data.data());

  return ::qnn::BuildConstantPadOp(tensor_pool, {input, pad_tensor}, {output});
}

std::vector<::qnn::OpWrapper> BuildQuantizedPadDefaultOp(
    ::qnn::TensorPool& tensor_pool) {
  const std::vector<std::uint32_t> input_dims{1, kInputHeight, kInputWidth, 1};
  const std::vector<std::uint32_t> output_dims{1, kOutputHeight, kOutputWidth,
                                               1};
  const std::vector<std::uint32_t> pad_dims{4, 2};
  const ::qnn::QuantizeParamsWrapperVariant quant_params{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.5f, 10};

  auto& input = tensor_pool.CreateInputTensorWithName(
      "in", QNN_DATATYPE_SFIXED_POINT_8, quant_params, input_dims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "out", QNN_DATATYPE_SFIXED_POINT_8, quant_params, output_dims);

  const std::vector<std::uint32_t> pad_data = {0, 0, 1, 1, 1, 1, 0, 0};
  auto& pad_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, pad_dims,
      pad_data.size() * sizeof(std::uint32_t), pad_data.data());

  return ::qnn::BuildConstantPadOp(tensor_pool, {input, pad_tensor}, {output});
}

template <typename T>
std::vector<::qnn::OpWrapper> BuildQuantizedPadV2Op(
    ::qnn::TensorPool& tensor_pool, Qnn_DataType_t data_type, T const_value) {
  const std::vector<std::uint32_t> input_dims{1, kInputHeight, kInputWidth, 1};
  const std::vector<std::uint32_t> output_dims{1, kOutputHeight, kOutputWidth,
                                               1};
  const std::vector<std::uint32_t> pad_dims{4, 2};
  const ::qnn::QuantizeParamsWrapperVariant quant_params{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.25f, 0};

  auto& input = tensor_pool.CreateInputTensorWithName("in", data_type,
                                                     quant_params, input_dims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "out", data_type, quant_params, output_dims);

  const std::vector<std::uint32_t> pad_data = {0, 0, 1, 1, 1, 1, 0, 0};
  auto& pad_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, pad_dims,
      pad_data.size() * sizeof(std::uint32_t), pad_data.data());
  auto& const_tensor = tensor_pool.CreateStaticTensor(
      data_type, quant_params, {1}, sizeof(T), &const_value);

  return ::qnn::BuildConstantPadOp(tensor_pool,
                                   {input, pad_tensor, const_tensor}, {output});
}

std::vector<::qnn::OpWrapper> BuildQuantizedPadV2MismatchedInt32ConstOp(
    ::qnn::TensorPool& tensor_pool, std::int32_t const_value) {
  const std::vector<std::uint32_t> input_dims{1, kInputHeight, kInputWidth, 1};
  const std::vector<std::uint32_t> output_dims{1, kOutputHeight, kOutputWidth,
                                               1};
  const std::vector<std::uint32_t> pad_dims{4, 2};
  const ::qnn::QuantizeParamsWrapperVariant quant_params{
      std::in_place_type<::qnn::ScaleOffsetQuantizeParamsWrapper>, 0.25f, 0};

  auto& input = tensor_pool.CreateInputTensorWithName(
      "in", QNN_DATATYPE_SFIXED_POINT_8, quant_params, input_dims);
  auto& output = tensor_pool.CreateOutputTensorWithName(
      "out", QNN_DATATYPE_SFIXED_POINT_8, quant_params, output_dims);

  const std::vector<std::uint32_t> pad_data = {0, 0, 1, 1, 1, 1, 0, 0};
  auto& pad_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, {}, pad_dims,
      pad_data.size() * sizeof(std::uint32_t), pad_data.data());
  auto& const_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, {1}, sizeof(std::int32_t), &const_value);

  return ::qnn::BuildConstantPadOp(tensor_pool,
                                   {input, pad_tensor, const_tensor}, {output});
}

::testing::AssertionResult ValidateQnnPadModel(
    ::qnn::QnnModel& qnn_model, std::vector<::qnn::OpWrapper> ops) {
  if (ops.empty()) {
    return ::testing::AssertionFailure() << "Pad builder returned no ops";
  }

  qnn_model.MoveOpsToGraph(std::move(ops));
  if (!qnn_model.ValidateOpConfig()) {
    return ::testing::AssertionFailure()
           << "QNN rejected the Pad op configuration";
  }
  if (!qnn_model.Finalize()) {
    return ::testing::AssertionFailure() << "QNN failed to finalize Pad model";
  }

  return ::testing::AssertionSuccess();
}

std::optional<std::int32_t> GetPadConstantValue(const ::qnn::OpWrapper& op) {
  constexpr size_t kPadConstantValueScalarIndex = 1;
  const auto scalar_param = op.GetScalarParam(kPadConstantValueScalarIndex);
  if (!scalar_param.has_value()) {
    return std::nullopt;
  }

  Qnn_Param_t qnn_param = QNN_PARAM_INIT;
  scalar_param->CloneTo(qnn_param);
  if (qnn_param.name == nullptr ||
      std::string_view(qnn_param.name) !=
          QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE ||
      qnn_param.paramType != QNN_PARAMTYPE_SCALAR ||
      qnn_param.scalarParam.dataType != QNN_DATATYPE_INT_32) {
    return std::nullopt;
  }

  return qnn_param.scalarParam.int32Value;
}

template <typename T>
void ExpectQuantizedPadV2ConstantValue(Qnn_DataType_t data_type, T pad_value) {
  ::qnn::TensorPool tensor_pool;
  auto ops = BuildQuantizedPadV2Op<T>(tensor_pool, data_type, pad_value);

  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops.front().IsOpCode(::qnn::QnnOpCode::kPad));
  const auto pad_constant_value = GetPadConstantValue(ops.front());
  ASSERT_TRUE(pad_constant_value.has_value());
  EXPECT_EQ(*pad_constant_value, static_cast<std::int32_t>(pad_value));
}

TEST_P(PadOpBuilderQnnModelTest, FloatPadValidatesWithQnn) {
  EXPECT_TRUE(ValidateQnnPadModel(qnn_model_,
                                  BuildFloatPadOp(tensor_pool_)));
}

TEST_P(PadOpBuilderQnnModelTest,
       QuantizedPadDefaultConstValueValidatesWithQnn) {
  EXPECT_TRUE(ValidateQnnPadModel(qnn_model_,
                                  BuildQuantizedPadDefaultOp(tensor_pool_)));
}

TEST_P(PadOpBuilderQnnModelTest, QuantizedPadV2Int8ConstValueValidatesWithQnn) {
  EXPECT_TRUE(ValidateQnnPadModel(
      qnn_model_, BuildQuantizedPadV2Op<std::int8_t>(
                      tensor_pool_, QNN_DATATYPE_SFIXED_POINT_8, -5)));
}

TEST_P(PadOpBuilderQnnModelTest,
       QuantizedPadV2Uint8ConstValueValidatesWithQnn) {
  EXPECT_TRUE(ValidateQnnPadModel(
      qnn_model_, BuildQuantizedPadV2Op<std::uint8_t>(
                      tensor_pool_, QNN_DATATYPE_UFIXED_POINT_8, 5)));
}

TEST(PadOpBuilderTest, QuantizedPadV2ReadsPadConstTensorDataType) {
  ExpectQuantizedPadV2ConstantValue<std::int8_t>(QNN_DATATYPE_SFIXED_POINT_8,
                                                -7);
  ExpectQuantizedPadV2ConstantValue<std::uint8_t>(QNN_DATATYPE_UFIXED_POINT_8,
                                                 13);
  ExpectQuantizedPadV2ConstantValue<std::int16_t>(QNN_DATATYPE_SFIXED_POINT_16,
                                                 -257);
  ExpectQuantizedPadV2ConstantValue<std::uint16_t>(
      QNN_DATATYPE_UFIXED_POINT_16, 513);
}

TEST(PadOpBuilderTest, QuantizedPadUsesDefaultZeroPointConstValue) {
  ::qnn::TensorPool tensor_pool;
  auto ops = BuildQuantizedPadDefaultOp(tensor_pool);

  ASSERT_EQ(ops.size(), 1);
  const auto pad_constant_value = GetPadConstantValue(ops.front());
  ASSERT_TRUE(pad_constant_value.has_value());
  EXPECT_EQ(*pad_constant_value, 10);
}

TEST(PadOpBuilderTest, QuantizedPadV2RejectsMismatchedPadConstTensorType) {
  ::qnn::TensorPool tensor_pool;
  auto ops = BuildQuantizedPadV2MismatchedInt32ConstOp(tensor_pool, -100);

  EXPECT_TRUE(ops.empty());
}

}  // namespace
}  // namespace litert::qnn
