// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/runtime/litert_interpreter_builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_options.h"
#include "tflite/kernels/register.h"
#include "tflite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatNear;

constexpr absl::string_view kSimpleModelFile = "simple_model.tflite";

TEST(LiteRtInterpreterBuilderTest, BuildFromInPlaceMutatedLiteRtModel) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtModelT::Ptr model,
      LoadModelFromFile(testing::GetTestFilePath(kSimpleModelFile)));

  ASSERT_NE(model->MainSubgraph(), nullptr);
  model->MainSubgraph()->SetName("mutated_main_subgraph");
  ASSERT_FALSE(model->MainSubgraph()->Ops().empty());
  LiteRtOpT* add_op = model->MainSubgraph()->Ops().front();
  ASSERT_EQ(add_op->OpCode(), kLiteRtOpCodeTflAdd);
  ASSERT_NE(add_op->FbOp(), nullptr);

  auto* add_opts = TakeTflOptions(*add_op).AsAddOptions();
  EXPECT_EQ(add_op->FbOp(), nullptr);
  ASSERT_NE(add_opts, nullptr);
  add_opts->fused_activation_function = tflite::ActivationFunctionType_RELU;

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions options;
  options.SetUseSignatureTensorNames(true);
  const auto* alloc =
      GetTflFlatbuffer(*model).FlatbufferModelPtr()
          ? GetTflFlatbuffer(*model).FlatbufferModel().allocation()
          : nullptr;
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      BuildInterpreterFromLiteRtModel(*model, resolver,
                                      /*error_reporter=*/nullptr, options,
                                      alloc, /*num_threads=*/1, &interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  EXPECT_EQ(interpreter->subgraph(0)->GetName(), "mutated_main_subgraph");

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  float* input0 = interpreter->typed_input_tensor<float>(0);
  float* input1 = interpreter->typed_input_tensor<float>(1);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  input0[0] = -5.0f;
  input0[1] = 3.0f;
  input1[0] = -2.0f;
  input1[1] = -1.0f;

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  const float* output = interpreter->typed_output_tensor<float>(0);
  ASSERT_NE(output, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(output, 2),
              ElementsAre(FloatNear(0.0f, 1e-5), FloatNear(2.0f, 1e-5)));
}

TEST(LiteRtInterpreterBuilderTest, BuildFromInMemoryConstructedLiteRtModel) {
  auto model = std::make_unique<LiteRtModelT>();
  auto& subgraph = model->EmplaceSubgraph();

  const int32_t dims[] = {1, 2};
  const int32_t weight_dims[] = {2, 2};
  TensorType tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeFloat32, dims);
  TensorType weight_type =
      MakeRankedTensorType(kLiteRtElementTypeFloat32, weight_dims);

  auto& input_tensor = subgraph.EmplaceTensor();
  input_tensor.SetName("input0");
  input_tensor.SetType(tensor_type);

  auto& const_tensor = subgraph.EmplaceTensor();
  const_tensor.SetName("const_weights");
  const_tensor.SetType(weight_type);
  const float weight_values[] = {1.0f, 0.0f, 0.0f, 10.0f};
  OwningBufferRef<uint8_t> weight_buf(
      reinterpret_cast<const uint8_t*>(weight_values), sizeof(weight_values));
  SetWeightsFromOwnedBuffer(const_tensor.Weights(), std::move(weight_buf));

  auto& output_tensor = subgraph.EmplaceTensor();
  output_tensor.SetName("output0");
  output_tensor.SetType(tensor_type);

  subgraph.Inputs().push_back(&input_tensor);
  subgraph.Outputs().push_back(&output_tensor);

  auto& fc_op = subgraph.EmplaceOp();
  fc_op.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  TflOptions opts;
  opts.type = tflite::BuiltinOptions_FullyConnectedOptions;
  opts.value = new tflite::FullyConnectedOptionsT();
  SetTflOptions(fc_op, std::move(opts));
  AttachInput(&input_tensor, fc_op);
  AttachInput(&const_tensor, fc_op);
  AttachInput(nullptr, fc_op);  // Optional bias input (kTfLiteOptionalTensor).
  AttachOutput(&output_tensor, fc_op);

  std::vector<std::string> input_names = {"sig_in0"};
  std::vector<LiteRtTensorT*> input_tensors = {&input_tensor};
  std::vector<std::string> output_names = {"sig_out0"};
  std::vector<LiteRtTensorT*> output_tensors = {&input_tensor};
  auto& sig = model->EmplaceSignature(
      &subgraph, std::move(input_names), std::move(input_tensors),
      std::move(output_names), std::move(output_tensors), "serving_default");
  sig.RemapOutputTensor(&input_tensor, &output_tensor);

  const char kMetaValue[] = "test_meta_payload";
  LITERT_ASSERT_OK(model->PushMetadata(
      "custom_meta",
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(kMetaValue),
                               sizeof(kMetaValue) - 1)));

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions options;
  options.SetUseSignatureTensorNames(true);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(BuildInterpreterFromLiteRtModel(*model, resolver,
                                            /*error_reporter=*/nullptr, options,
                                            /*allocation=*/nullptr,
                                            /*num_threads=*/1, &interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  auto* sig_runner = interpreter->GetSignatureRunner("serving_default");
  ASSERT_NE(sig_runner, nullptr);
  ASSERT_EQ(sig_runner->AllocateTensors(), kTfLiteOk);

  TfLiteTensor* in_t = sig_runner->input_tensor("sig_in0");
  ASSERT_NE(in_t, nullptr);
  in_t->data.f[0] = 1.5f;
  in_t->data.f[1] = 2.5f;

  ASSERT_EQ(sig_runner->Invoke(), kTfLiteOk);
  const TfLiteTensor* out_t = sig_runner->output_tensor("sig_out0");
  ASSERT_NE(out_t, nullptr);
  EXPECT_THAT(absl::MakeConstSpan(out_t->data.f, 2),
              ElementsAre(FloatNear(1.5f, 1e-5), FloatNear(25.0f, 1e-5)));
}

TEST(LiteRtInterpreterBuilderTest,
     RejectsReadOnlyQuantizedTensorWithExternalBufferWithoutLeaking) {
  auto model = std::make_unique<LiteRtModelT>();
  auto& subgraph = model->EmplaceSubgraph();

  const int32_t dims[] = {2};
  auto& tensor = subgraph.EmplaceTensor();
  tensor.SetName("bad_tensor");
  tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, dims));
  tensor.SetExternalBufferId(1);
  tensor.SetQarams(MakePerTensorQuantization(/*scale=*/0.5f, /*zero_point=*/0));
  const int8_t data[] = {1, 2};
  SetWeightsFromOwnedBuffer(
      tensor.Weights(),
      OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(data),
                               sizeof(data)));

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions options;
  std::unique_ptr<tflite::Interpreter> interpreter;
  EXPECT_EQ(BuildInterpreterFromLiteRtModel(*model, resolver,
                                            /*error_reporter=*/nullptr, options,
                                            /*allocation=*/nullptr,
                                            /*num_threads=*/1, &interpreter),
            kTfLiteError);
  EXPECT_EQ(interpreter, nullptr);
}

TEST(LiteRtInterpreterBuilderTest, QuantizationModesPerChannelAndBlockWise) {
  auto model = std::make_unique<LiteRtModelT>();
  auto& subgraph = model->EmplaceSubgraph();

  const int32_t dims[] = {2, 2};
  auto& pc_tensor = subgraph.EmplaceTensor();
  pc_tensor.SetName("per_channel_tensor");
  pc_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt8, dims));
  const float scales[] = {0.25f, 0.5f};
  const int64_t zero_points[] = {3, 3};
  pc_tensor.SetQarams(MakePerChannelQuantization(
      scales, zero_points, /*quantized_dimension=*/0,
      [&](auto s) { return pc_tensor.RequestScratchBuffer(s); }));

  const int32_t scale_dims[] = {1};
  auto& bw_scales = subgraph.EmplaceTensor();
  bw_scales.SetName("bw_scales");
  bw_scales.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, scale_dims));

  auto& bw_zps = subgraph.EmplaceTensor();
  bw_zps.SetName("bw_zps");
  bw_zps.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, scale_dims));

  auto& bw_tensor = subgraph.EmplaceTensor();
  bw_tensor.SetName("block_wise_tensor");
  bw_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, dims));
  Quantization bw_quant;
  bw_quant.first = kLiteRtQuantizationBlockWise;
  bw_quant.second.block_wise.scales = &bw_scales;
  bw_quant.second.block_wise.zero_points = &bw_zps;
  bw_quant.second.block_wise.block_size = 32;
  bw_tensor.SetQarams(bw_quant);

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions options;
  options.SetCompressQuantizationZeroPoints(true);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(BuildInterpreterFromLiteRtModel(*model, resolver,
                                            /*error_reporter=*/nullptr, options,
                                            /*allocation=*/nullptr,
                                            /*num_threads=*/1, &interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  const TfLiteTensor* t0 = interpreter->subgraph(0)->tensor(0);
  ASSERT_NE(t0, nullptr);
  EXPECT_EQ(t0->quantization.type, kTfLiteAffineQuantization);
  const auto* affine =
      static_cast<const TfLiteAffineQuantization*>(t0->quantization.params);
  ASSERT_NE(affine, nullptr);
  EXPECT_EQ(affine->scale->size, 2);
  // Compressed identical zero points [3, 3] -> size 1.
  EXPECT_EQ(affine->zero_point->size, 1);
  EXPECT_EQ(affine->zero_point->data[0], 3);

  const TfLiteTensor* t3 = interpreter->subgraph(0)->tensor(3);
  ASSERT_NE(t3, nullptr);
  EXPECT_EQ(t3->quantization.type, kTfLiteBlockwiseQuantization);
  const auto* bw =
      static_cast<const TfLiteBlockwiseQuantization*>(t3->quantization.params);
  ASSERT_NE(bw, nullptr);
  EXPECT_EQ(bw->scale, 1);
  EXPECT_EQ(bw->zero_point, 2);
  EXPECT_EQ(bw->blocksize, 32);
}

TEST(LiteRtInterpreterBuilderTest,
     ZeroSizeConstantBufferAndInMemoryCustomOpRegistration) {
  auto model = std::make_unique<LiteRtModelT>();
  auto& subgraph = model->EmplaceSubgraph();

  // 0-size constant tensor (e.g. shape=[0] target shape for scalar RESHAPE).
  const int32_t empty_shape_dims[] = {0};
  auto& zero_size_const = subgraph.EmplaceTensor();
  zero_size_const.SetName("empty_shape_const");
  zero_size_const.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, empty_shape_dims));
  const uint8_t dummy_non_null = 0;
  SetWeightsFromUnownedBuffer(
      zero_size_const.Weights(),
      BufferRef<uint8_t>(&dummy_non_null, /*size=*/0));

  const int32_t io_dims[] = {1};
  auto& out_tensor = subgraph.EmplaceTensor();
  out_tensor.SetName("out0");
  out_tensor.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, io_dims));
  subgraph.Outputs().push_back(&out_tensor);

  auto& dispatch_op = subgraph.EmplaceOp();
  MakeDispatchOp(dispatch_op);
  AttachInput(&zero_size_const, dispatch_op);
  AttachOutput(&out_tensor, dispatch_op);

  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterOptions options;
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(BuildInterpreterFromLiteRtModel(*model, resolver,
                                            /*error_reporter=*/nullptr, options,
                                            /*allocation=*/nullptr,
                                            /*num_threads=*/1, &interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  // Verify the 0-size constant tensor is registered as read-only
  // (kTfLiteMmapRo) rather than a runtime read-write tensor (kTfLiteArenaRw).
  const TfLiteTensor* t0 = interpreter->subgraph(0)->tensor(0);
  ASSERT_NE(t0, nullptr);
  EXPECT_EQ(t0->allocation_type, kTfLiteMmapRo);

  // Verify the unresolved custom op registration retains a valid non-dangling
  // custom_name matching kLiteRtDispatchOpCustomName.
  const auto* node_and_reg = interpreter->subgraph(0)->node_and_registration(0);
  ASSERT_NE(node_and_reg, nullptr);
  ASSERT_NE(node_and_reg->second.custom_name, nullptr);
  EXPECT_STREQ(node_and_reg->second.custom_name, "DISPATCH_OP");
}

}  // namespace
}  // namespace litert::internal
