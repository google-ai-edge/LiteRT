#include "litert/vendors/qualcomm/core/utils/qnn_model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "litert/vendors/qualcomm/core/builders/relu_op_builder.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/qnn_backend_creator.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

TEST(QnnModelTest, SingleRelu) {
  auto options = ::qnn::Options();
  QnnBackendCreator backend_creator(options, "SM8650");

  ::qnn::TensorPool tensor_pool;
  std::vector<::qnn::OpWrapper> ops_wrappers;
  const std::vector<std::uint32_t> kDims{1, 2, 2, 1};
  auto& input_0 = tensor_pool.CreateInputTensorWithSuffix(QNN_DATATYPE_FLOAT_32,
                                                          {}, kDims, "");
  auto& output_0 = tensor_pool.CreateOutpuTensorWithSuffix(
      QNN_DATATYPE_FLOAT_32, {}, kDims, "");
  auto ops = BuildReluOp(tensor_pool, {input_0}, {output_0});
  std::move(ops.begin(), ops.end(), std::back_inserter(ops_wrappers));

  ::qnn::QnnModel model(backend_creator.GetBackendHandle(),
                        backend_creator.GetContextHandle(),
                        backend_creator.GetQnnApi());
  EXPECT_TRUE(model.ValidateOpConfig(ops_wrappers));
  EXPECT_TRUE(model.Finalize(ops_wrappers));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_index = model.SetInputTensor(input_0);
  auto output_index = model.SetOutputTensor(output_0);

  model.SetInputData<float>(input_index, {0.f, -1.f, -1.f, 0.f});

  model.Execute();
  auto output_data = model.GetOutputData<float>(output_index);
  EXPECT_TRUE(output_data);
  EXPECT_EQ(output_data->size(), 4);
  EXPECT_THAT(output_data.value(), Pointwise(FloatNear(1e-3), {0, 0, 0, 0}));
}

}  // namespace
}  // namespace litert::qnn