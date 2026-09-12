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

#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "tflite/schema/schema_generated.h"

TEST(TensorRtGraphBuilderTest, LongContextSoftmaxSupport) {
  // Gemma 4 12B prefill_1024 at 32K depth. Only construct tensor metadata;
  // partition eligibility must not depend on allocating this large tensor.
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& input = graph.EmplaceTensor();
  auto& output = graph.EmplaceTensor();
  for (auto* tensor : {&input, &output}) {
    tensor->SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 34818}));
  }
  auto& op = graph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflSoftmax);
  litert::internal::AttachInput(&input, op);
  litert::internal::AttachOutput(&output, op);
  for (float beta : {1.0f, 0.5f}) {
    tflite::SoftmaxOptionsT softmax;
    softmax.beta = beta;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(softmax));
    litert::internal::SetTflOptions(op, std::move(options));
    EXPECT_EQ(litert::nvidia::IsTensorRtOpSupported(
                  litert::compiler::Op(LrtGetCompilerContext(), &op)),
              beta == 1.0f);
  }
}

TEST(TensorRtGraphBuilderTest, LongContextBatchMatmulSupport) {
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& query = graph.EmplaceTensor();
  query.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 512}));
  auto& key = graph.EmplaceTensor();
  key.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 34818, 512}));
  auto& scores = graph.EmplaceTensor();
  scores.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 34818}));
  auto& op = graph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  tflite::BatchMatMulOptionsT matmul;
  matmul.adj_y = true;
  tflite::BuiltinOptionsUnion options;
  options.Set(std::move(matmul));
  litert::internal::SetTflOptions(op, std::move(options));
  litert::internal::AttachInput(&query, op);
  litert::internal::AttachInput(&key, op);
  litert::internal::AttachOutput(&scores, op);
  EXPECT_TRUE(litert::nvidia::IsTensorRtOpSupported(
      litert::compiler::Op(LrtGetCompilerContext(), &op)));
  scores.SetType(
      MakeRankedTensorType(kLiteRtElementTypeInt32, {1, 1, 16384, 34818}));
  EXPECT_FALSE(litert::nvidia::IsTensorRtOpSupported(
      litert::compiler::Op(LrtGetCompilerContext(), &op)));
}

TEST(TensorRtGraphBuilderTest, LongContextRuntimeBatchMatmulSupport) {
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& query = graph.EmplaceTensor();
  query.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 512}));
  auto& key = graph.EmplaceTensor();
  key.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 34818, 512}));
  auto& positions = graph.EmplaceTensor();
  positions.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1024}));
  auto& scores = graph.EmplaceTensor();
  scores.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 34818}));
  auto& op = graph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  tflite::StableHLOCompositeOptionsT composite;
  composite.name = "odml.runtime_bmm";
  tflite::BuiltinOptions2Union options;
  options.Set(std::move(composite));
  litert::internal::SetTflOptions2(op, std::move(options));
  litert::internal::AttachInput(&query, op);
  litert::internal::AttachInput(&key, op);
  litert::internal::AttachInput(&positions, op);
  litert::internal::AttachOutput(&scores, op);
  EXPECT_TRUE(litert::nvidia::IsTensorRtOpSupported(
      litert::compiler::Op(LrtGetCompilerContext(), &op)));
  scores.SetType(
      MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 1, 16384, 34817}));
  EXPECT_FALSE(litert::nvidia::IsTensorRtOpSupported(
      litert::compiler::Op(LrtGetCompilerContext(), &op)));
}

TEST(TensorRtGraphBuilderTest, ConstantInt64ToHalfCast) {
  // Gemma 4 12B's FP16 cache decomposition casts an INT64 scalar. Also
  // exercise a vector, negative values, and the largest finite FP16 integer.
  for (bool scalar : {true, false}) {
    SCOPED_TRACE(scalar);
    const std::array<int64_t, 4> values = {0, 1, -1, 65504};
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& constant = graph.EmplaceTensor();
    const std::vector<int32_t> shape =
        scalar ? std::vector<int32_t>{} : std::vector<int32_t>{4};
    constant.SetType(MakeRankedTensorType(kLiteRtElementTypeInt64, shape));
    constant.SetName("constant");
    SetWeightsFromUnownedBuffer(
        constant.Weights(),
        litert::BufferRef<uint8_t>(
            reinterpret_cast<const uint8_t*>(values.data()),
            (scalar ? 1 : values.size()) * sizeof(int64_t)));
    auto& cast_output = graph.EmplaceTensor();
    cast_output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, shape));
    auto& cast = graph.EmplaceOp();
    cast.SetOpCode(kLiteRtOpCodeTflCast);
    litert::internal::AttachInput(&constant, cast);
    litert::internal::AttachOutput(&cast_output, cast);

    auto& input = graph.EmplaceTensor();
    input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));
    input.SetName("input");
    graph.Inputs().push_back(&input);
    auto& output = graph.EmplaceTensor();
    output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {4}));
    output.SetName("output");
    graph.Outputs().push_back(&output);
    auto& add = graph.EmplaceOp();
    add.SetOpCode(kLiteRtOpCodeTflAdd);
    tflite::BuiltinOptionsUnion add_options;
    add_options.Set(tflite::AddOptionsT{});
    litert::internal::SetTflOptions(add, std::move(add_options));
    litert::internal::AttachInput(&input, add);
    litert::internal::AttachInput(&cast_output, add);
    litert::internal::AttachOutput(&output, add);

    const auto* compiler_context = LrtGetCompilerContext();
    ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(compiler_context, &cast)));
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(compiler_context, &graph));
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    ASSERT_FALSE(built->engine.empty());
    ASSERT_FALSE(built->is_stripped_plan);

    litert::nvidia::TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(built->engine.data(),
                                       built->engine.size()));
    ASSERT_NE(engine, nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    const std::array<uint16_t, 4> zeros{};
    std::array<uint16_t, 4> actual{};
    void* device_input = nullptr;
    void* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_input, sizeof(zeros)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, sizeof(actual)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input, zeros.data(), sizeof(zeros),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(
        context->setTensorAddress(built->input_names[0].c_str(), device_input));
    ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                          device_output));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_output, sizeof(actual),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    const std::array<uint16_t, 4> expected =
        scalar ? zeros : std::array<uint16_t, 4>{0, 0x3c00, 0xbc00, 0x7bff};
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  }
}
