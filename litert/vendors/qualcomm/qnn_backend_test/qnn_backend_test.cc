// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/topk_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnTypes.h"  // from @qairt

using testing::ElementsAre;
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

TEST_F(QnnModelTest, SingleTopK) {
  SetUpQnnModel(::qnn::Options(), "SM8650");

  const std::vector<std::uint32_t> inputDims{1, 5};
  const uint32_t k_value = 3;
  const std::vector<std::uint32_t> outputDims{1, 3};

  auto& input_tensor = tensor_pool_.CreateInputTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, inputDims, "");
  auto& values_tensor = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, outputDims, "");
  auto& indices_tensor = tensor_pool_.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_UINT_32, {}, outputDims, "");

  auto ops = ::qnn::BuildTopKOp(tensor_pool_, {input_tensor},
                                {values_tensor, indices_tensor}, k_value);
  ASSERT_FALSE(ops.empty());

  qnn_model_.MoveOpsToGraph(std::move(ops));
  ASSERT_TRUE(qnn_model_.Finalize());

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_idx = qnn_model_.AddInputTensor(input_tensor);
  auto values_idx = qnn_model_.AddOutputTensor(values_tensor);
  auto indices_idx = qnn_model_.AddOutputTensor(indices_tensor);

  qnn_model_.SetInputData<float>(input_idx, {1.2f, 5.6f, 3.3f, 9.8f, 2.1f});
  ASSERT_TRUE(qnn_model_.Execute());
  auto values_data = qnn_model_.GetOutputData<float>(values_idx);
  auto indices_data = qnn_model_.GetOutputData<std::uint32_t>(indices_idx);

  ASSERT_TRUE(values_data);
  ASSERT_TRUE(indices_data);
  ASSERT_EQ(values_data->size(), 3);
  ASSERT_EQ(indices_data->size(), 3);
  ASSERT_THAT(values_data.value(),
              Pointwise(FloatNear(1e-2), {9.8f, 5.6f, 3.3f}));
  ASSERT_THAT(indices_data.value(), ElementsAre(3, 1, 2));
}

}  // namespace
}  // namespace litert::qnn
