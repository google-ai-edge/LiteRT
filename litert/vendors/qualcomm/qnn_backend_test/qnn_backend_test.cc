// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "QnnTypes.h"  // from @qairt
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

class QnnModelTest : public testing::Test {
 protected:
  QnnManager::Ptr qnn_manager_ptr_;
  QnnManager::ContextHandle context_handle_;
  ::qnn::QnnModel qnn_model_;
  ::qnn::TensorPool tensor_pool_;

  void SetUpQnnModel(const ::qnn::Options& options,
                     std::string_view soc_model_name) {
    // TODO (chunhsue-qti) get rid of QnnManager and move to core/
    auto qnn_manager = QnnManager::Create(options, std::nullopt,
                                          ::qnn::FindSocModel(soc_model_name));
    ASSERT_TRUE(qnn_manager) << "Failed to create QnnManager";
    auto context_configs = QnnManager::DefaultContextConfigs();
    auto context_handle =
        (**qnn_manager)
            .CreateContextHandle(context_configs, options.GetProfiling());
    ASSERT_TRUE(context_handle) << "Failed to create Context Handle";

    std::swap(qnn_manager_ptr_, *qnn_manager);
    context_handle_ = std::move(context_handle.Value());

    auto qnn_model =
        ::qnn::QnnModel(qnn_manager_ptr_->BackendHandle(),
                        qnn_manager_ptr_->Api(), context_handle_.get());

    std::swap(qnn_model_, qnn_model);
  }
};

TEST_F(QnnModelTest, Sanity) {
  SetUpQnnModel(::qnn::Options(), "SM8650");
  EXPECT_NE(qnn_manager_ptr_->BackendHandle(), nullptr);
  EXPECT_NE(context_handle_.get(), nullptr);
  EXPECT_NE(qnn_manager_ptr_->Api(), nullptr);
  EXPECT_TRUE(qnn_model_.ValidateOpConfig());
  // no nodes to finalize
  EXPECT_FALSE(qnn_model_.Finalize());
}

TEST_F(QnnModelTest, SingleRelu) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto ops = ::qnn::BuildReluOp(tensor_pool_, {input_0}, {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);  // NOLINT
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<float>(input_idx, {-1.f, 0.f, 1.f, 2.f});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<float>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), Pointwise(FloatNear(1e-3), {0, 0, 1, 2}));
}

TEST_F(QnnModelTest, SingleElementWiseDivide) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  ::qnn::QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<::qnn::ScaleOffsetQuantizeParamsWrapper>(0.1f, 0);

  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& input_1 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto ops = ::qnn::BuildElementwiseDivOp(tensor_pool_, {input_0, input_1},
                                          {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  // #if !defined(__ANDROID__)
  //   GTEST_SKIP() << "The rest of this test is specific to Android devices
  //   with a "
  //                   "Qualcomm HTP";
  // #endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);  // NOLINT
  auto input_idx1 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx, {40, 40, 40, 40});
  qnn_model_.SetInputData<int16_t>(input_idx1, {4, 4, 4, 2});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {100, 100, 100, 200}));
}

TEST_F(QnnModelTest, SingleElementWiseMax) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  ::qnn::QuantizeParamsWrapperVariant quant_param;
  quant_param.emplace<::qnn::ScaleOffsetQuantizeParamsWrapper>(0.1f, 0);
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& input_1 = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto& output_0 = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_SFIXED_POINT_16, quant_param, kDims, "");
  auto ops = ::qnn::BuildElementwiseMaximumOp(tensor_pool_, {input_0, input_1},
                                              {output_0});
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));

  // ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  ASSERT_TRUE(qnn_model_.Finalize());

  // #if !defined(__ANDROID__)
  //   GTEST_SKIP() << "The rest of this test is specific to Android devices
  //   with a "
  //                   "Qualcomm HTP";
  // #endif

  auto input_idx = qnn_model_.AddInputTensor(input_0);  // NOLINT
  auto input_idx2 = qnn_model_.AddInputTensor(input_1);
  auto output_idx = qnn_model_.AddOutputTensor(output_0);

  qnn_model_.SetInputData<int16_t>(input_idx, {-1, 2, 3, 4});
  qnn_model_.SetInputData<int16_t>(input_idx2, {2, 0, 5, 6});

  ASSERT_TRUE(qnn_model_.Execute());

  auto output_data = qnn_model_.GetOutputData<int16_t>(output_idx);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(), Pointwise(FloatNear(1e-3), {2, 2, 5, 6}));
}

}  // namespace
}  // namespace litert::qnn
