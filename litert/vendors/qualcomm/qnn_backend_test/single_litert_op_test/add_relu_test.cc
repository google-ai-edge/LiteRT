#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "litert/c/litert_model.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/matchers.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/qnn_model.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/qnn_backend_test/qnn_backend_creator.h"
#include "litert/vendors/qualcomm/qnn_backend_test/utils.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

TEST(FromLiteRtOp, AddRelu) {
  // LiteRT model creation
  std::vector<float> cst_data = {1.f};
  testing::TensorDetails lhs = {{2, 2}, kLiteRtElementTypeFloat32, "lhs"};
  testing::TensorDetails rhs = {
      {},
      kLiteRtElementTypeFloat32,
      "cst",
      MakeBufferRef(cst_data.cbegin(), cst_data.cend())};
  testing::TensorDetails output = {{2, 2}, kLiteRtElementTypeFloat32, "output"};

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto litert_model,
      testing::SingleOpModel<kLiteRtOpCodeTflAdd>(
          {std::move(lhs), std::move(rhs)}, {std::move(output)},
          tflite::ActivationFunctionType_RELU, false));

  ASSERT_EQ(litert_model->NumSubgraphs(), 1);

  // QNN Conversion
  ::qnn::TensorPool tensor_pool;
  std::vector<::qnn::TensorWrapperRef> input_tensors;
  std::vector<::qnn::TensorWrapperRef> output_tensors;
  std::vector<::qnn::OpWrapper> op_wrappers;
  auto options = ::qnn::Options();

  ASSERT_TRUE(
      ConvertLiteRtSubGraph(litert::Subgraph(litert_model->MainSubgraph()),
                            tensor_pool, input_tensors, output_tensors,
                            op_wrappers, options.GetUseHtpPreference()));

  QnnBackendCreator backend_creator(options, "SM8650");
  ::qnn::QnnModel model(backend_creator.GetBackendHandle(),
                        backend_creator.GetContextHandle(),
                        backend_creator.GetQnnApi());

  ASSERT_TRUE(model.ValidateOpConfig(op_wrappers));
  ASSERT_TRUE(model.Finalize(op_wrappers));

#if !defined(__ANDROID__)
  GTEST_SKIP() << "The rest of this test is specific to Android devices with a "
                  "Qualcomm HTP";
#endif

  auto input_index_0 = model.SetInputTensor(input_tensors[0]);
  auto output_index_0 = model.SetOutputTensor(output_tensors[0]);

  ASSERT_TRUE(model.SetInputData<float>(input_index_0, {-2.f, -1.f, 0.f, 1.f}));

  ASSERT_TRUE(model.Execute());
  auto output_data = model.GetOutputData<float>(output_index_0);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {0.f, 0.f, 1.f, 2.f}));
}

}  // namespace
}  // namespace litert::qnn
