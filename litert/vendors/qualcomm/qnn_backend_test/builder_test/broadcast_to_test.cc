// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/builders/broadcast_to_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

using ::testing::Each;
using ::testing::Eq;
#if defined(__ANDROID__)
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;
#endif

TEST_P(QnnModelTest, BroadcastToFloat16) {
  const std::vector<std::uint32_t> kInputDims{1, 3};
  const std::vector<std::uint32_t> kOutputDims{2, 3};
  const std::vector<std::uint32_t> kExpectedStaticDims{2, 1};

  auto& input_f16 = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_16, {}, kInputDims);
  auto& output_f16 = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_16, {}, kOutputDims);

  auto ops = ::qnn::BuildBroadcastToOp(tensor_pool_, {input_f16}, {output_f16});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kElementWiseAdd);
  ASSERT_EQ(ops[0].GetInputCount(), 2u);
  EXPECT_TRUE(ops[0].GetInputTensor(0).IsF16());

  const ::qnn::TensorWrapper& static_tensor = ops[0].GetInputTensor(1);
  EXPECT_TRUE(static_tensor.IsTensorStatic());
  EXPECT_TRUE(static_tensor.IsF16());
  EXPECT_EQ(static_tensor.GetDimensions(), kExpectedStaticDims);
  ASSERT_EQ(static_tensor.GetTensorBytes(), 2 * sizeof(std::uint16_t));
  const auto* raw_data = static_cast<const std::uint16_t*>(
      static_tensor.GetQnnTensor().v2.clientBuf.data);
  ASSERT_NE(raw_data, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(raw_data, 2), Each(Eq(0u)));

  EXPECT_TRUE(ops[0].GetOutputTensor(0).IsF16());
  EXPECT_EQ(ops[0].GetOutputTensor(0).GetDimensions(), kOutputDims);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());
}

TEST_P(QnnModelTest, BroadcastToFloat16WithFp32CastsExecutes) {
  const std::vector<std::uint32_t> kInputDims{1, 3};
  const std::vector<std::uint32_t> kOutputDims{2, 3};

  auto& input_f32 = tensor_pool_.CreateInputTensorWithName(
      "input_f32", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& input_f16 = tensor_pool_.CreateNativeTensor(QNN_DATATYPE_FLOAT_16, {},
                                                    kInputDims);
  auto& output_f16 = tensor_pool_.CreateNativeTensor(QNN_DATATYPE_FLOAT_16, {},
                                                     kOutputDims);
  auto& output_f32 = tensor_pool_.CreateOutputTensorWithName(
      "output_f32", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  std::vector<::qnn::OpWrapper> ops;
  ops.emplace_back(::qnn::CreateCastOp(input_f32, input_f16));
  auto broadcast_ops =
      ::qnn::BuildBroadcastToOp(tensor_pool_, {input_f16}, {output_f16});
  ASSERT_EQ(broadcast_ops.size(), 1u);
  ops.emplace_back(std::move(broadcast_ops[0]));
  ops.emplace_back(::qnn::CreateCastOp(output_f16, output_f32));

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input_f32);
  auto output_idx = qnn_model_.AddOutputTensor(output_f32);
  ASSERT_TRUE(qnn_model_.SetInputData<float>(input_idx, {1.5f, -2.0f, 3.25f}));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(
      output_data.value(),
      Pointwise(FloatNear(1e-2f), {1.5f, -2.0f, 3.25f, 1.5f, -2.0f, 3.25f}));
#endif
}

TEST_P(QnnModelTest, BroadcastToFloat32) {
  const std::vector<std::uint32_t> kInputDims{1, 3};
  const std::vector<std::uint32_t> kOutputDims{2, 3};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_FLOAT_32, {}, kInputDims);
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_FLOAT_32, {}, kOutputDims);

  auto ops = ::qnn::BuildBroadcastToOp(tensor_pool_, {input}, {output});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kElementWiseAdd);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  ASSERT_TRUE(qnn_model_.SetInputData<float>(input_idx, {1.0f, 2.0f, 3.0f}));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(
      output_data.value(),
      Pointwise(FloatNear(1e-3f), {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}));
#endif
}

TEST_P(QnnModelTest, BroadcastToBool8) {
  const std::vector<std::uint32_t> kInputDims{1, 3};
  const std::vector<std::uint32_t> kOutputDims{2, 3};

  auto& input = tensor_pool_.CreateInputTensorWithName(
      "input", QNN_DATATYPE_BOOL_8, {}, kInputDims);
  auto& output = tensor_pool_.CreateOutputTensorWithName(
      "output", QNN_DATATYPE_BOOL_8, {}, kOutputDims);

  auto ops = ::qnn::BuildBroadcastToOp(tensor_pool_, {input}, {output});
  ASSERT_EQ(ops.size(), 1u);
  EXPECT_EQ(ops[0].GetOpCode(), ::qnn::QnnOpCode::kElementWiseOr);

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#else
  auto input_idx = qnn_model_.AddInputTensor(input);
  auto output_idx = qnn_model_.AddOutputTensor(output);
  ASSERT_TRUE(qnn_model_.SetInputData<bool>(input_idx, {true, false, true}));

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<bool>(output_idx);
  ASSERT_TRUE(output_data);
  EXPECT_THAT(output_data.value(),
              ElementsAre(true, false, true, true, false, true));
#endif
}

}  // namespace
}  // namespace litert::qnn
