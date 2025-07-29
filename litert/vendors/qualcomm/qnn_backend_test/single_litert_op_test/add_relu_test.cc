#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdio>

#include "litert/core/model/model.h"
#include "litert/core/model/model_graph.h"
#include "litert/core/util/flatbuffer_tools.h"
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
  // LiteRT op creation
  static constexpr std::array kDims = {2, 2};
  LiteRtTensorT input1;
  input1.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input1.SetName("input1");

  LiteRtTensorT input2;
  input2.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  input2.SetName("input2");

  LiteRtTensorT output;
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32,
                                      absl::MakeConstSpan(kDims)));
  output.SetName("output");

  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflAdd);
  ::litert::internal::AttachInput(&input1, op);
  ::litert::internal::AttachInput(&input2, op);
  ::litert::internal::AttachOutput(&output, op);

  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  ::litert::internal::TflOptions tfl_opts;
  tfl_opts.Set(std::move(add_opts));
  litert::internal::SetTflOptions(op, std::move(tfl_opts));
  ::litert::Op Op(&op);

  // QNN Conversion
  ::qnn::TensorPool tensor_pool;
  std::vector<::qnn::TensorWrapperRef> input_tensors;
  std::vector<::qnn::TensorWrapperRef> output_tensors;
  std::vector<::qnn::OpWrapper> op_wrappers;
  auto options = ::qnn::Options();

  ASSERT_TRUE(ConvertLiteRtOp(Op, tensor_pool, input_tensors, output_tensors,
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
  auto input_index_1 = model.SetInputTensor(input_tensors[1]);
  auto output_index_0 = model.SetOutputTensor(output_tensors[0]);

  ASSERT_TRUE(model.SetInputData<float>(input_index_0, {0.f, -1.f, -1.f, 0.f}));
  ASSERT_TRUE(model.SetInputData<float>(input_index_1, {-1.f, 1.f, 2.f, 2.f}));

  ASSERT_TRUE(model.Execute());
  auto output_data = model.GetOutputData<float>(output_index_0);
  ASSERT_TRUE(output_data);
  ASSERT_EQ(output_data->size(), 4);
  ASSERT_THAT(output_data.value(),
              Pointwise(FloatNear(1e-3), {0.f, 0.f, 1.f, 2.f}));
}

}  // namespace
}  // namespace litert::qnn