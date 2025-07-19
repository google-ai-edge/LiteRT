#include <gtest/gtest.h>

#include <string_view>

#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "litert/vendors/qualcomm/qnn_model_test/utils.h"

namespace qnn {
namespace {

TEST(QnnModelTest, SingleRelu) {
  // Create QnnManager
  auto configs = litert::qnn::QnnManager::DefaultBackendConfigs();
  auto options = ::qnn::Options();
  auto qnn = litert::qnn::QnnManager::Create(configs, options, std::nullopt,
                                             FindSocModel("SM8650"));
  ASSERT_TRUE(qnn);

  // Create graph via qnn wrappers
  // This can be substitute into any graph wanted to be test directly in QNN
  TensorPool tensor_pool;
  const std::vector<std::uint32_t> kDims{1, 1, 128, 1408};
  auto& input_0 = tensor_pool.CreateInputTensorWithSuffix(QNN_DATATYPE_FLOAT_32,
                                                          {}, kDims, "");
  auto& output_0 = tensor_pool.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto ops = BuildReluOp(tensor_pool, {input_0}, {output_0});
  EXPECT_EQ(ops.size(), 1);

  // Check validation and compilation
  EXPECT_EQ(ValidateModel((**qnn), ops), true);
  EXPECT_EQ(CreateGraphAndCompile((**qnn), ops), true);
}

}  // namespace
}  // namespace qnn