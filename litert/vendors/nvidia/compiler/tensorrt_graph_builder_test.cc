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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
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

namespace {

// Packs signed INT4 values two per byte, low nibble first (TFLite layout).
std::vector<uint8_t> PackInt4(const std::vector<int8_t>& values) {
  std::vector<uint8_t> packed((values.size() + 1) / 2);
  for (size_t i = 0; i < values.size(); ++i) {
    const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
    packed[i / 2] |= i % 2 == 0 ? nibble : nibble << 4;
  }
  return packed;
}

}  // namespace

TEST(TensorRtGraphBuilderTest, Int4FullyConnectedBlockScales) {
  // K=128 selects the block-scaled encoding (block 64); K=80 has no block
  // candidate and keeps the per-channel dequantize. Both must reproduce the
  // FP32 reference of the same per-channel INT4 weights.
  for (int32_t k : {128, 80}) {
    SCOPED_TRACE(k);
    constexpr int32_t kM = 16;
    constexpr int32_t kN = 32;
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& input = graph.EmplaceTensor();
    input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kM, k}));
    input.SetName("input");
    graph.Inputs().push_back(&input);

    std::vector<int8_t> values(static_cast<size_t>(kN) * k);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<int8_t>((i * 7 + i / k * 3) % 16) - 8;
    }
    const std::vector<uint8_t> packed = PackInt4(values);
    auto& weights = graph.EmplaceTensor();
    weights.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, {kN, k}));
    weights.SetName("weights");
    SetWeightsFromUnownedBuffer(
        weights.Weights(),
        litert::BufferRef<uint8_t>(packed.data(), packed.size()));
    std::vector<float> scales(kN);
    for (int32_t n = 0; n < kN; ++n) {
      scales[n] = 0.01f * (n + 1);
    }
    const std::vector<int64_t> zero_points(kN, 0);
    weights.SetQarams(MakePerChannelQuantization(scales, zero_points,
                                                 /*quantized_dim=*/0, weights));

    auto& output = graph.EmplaceTensor();
    output.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kM, kN}));
    output.SetName("output");
    graph.Outputs().push_back(&output);
    auto& fc = graph.EmplaceOp();
    fc.SetOpCode(kLiteRtOpCodeTflFullyConnected);
    tflite::FullyConnectedOptionsT fc_options;
    fc_options.keep_num_dims = true;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(fc_options));
    litert::internal::SetTflOptions(fc, std::move(options));
    litert::internal::AttachInput(&input, fc);
    litert::internal::AttachInput(&weights, fc);
    litert::internal::AttachOutput(&output, fc);

    const auto* compiler_context = LrtGetCompilerContext();
    ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(compiler_context, &fc)));
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(compiler_context, &graph));
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();

    litert::nvidia::TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(built->engine.data(),
                                       built->engine.size()));
    ASSERT_NE(engine, nullptr);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(
        engine->createEngineInspector());
    ASSERT_NE(inspector, nullptr);
    const std::string layers = inspector->getEngineInformation(
        nvinfer1::LayerInformationFormat::kONELINE);
    // Block scales fuse the dequantize into one Myelin GEMM kernel instead of
    // materializing the weights before a separate matmul layer.
    EXPECT_EQ(layers.find("[Matrix Multiply]") == std::string::npos, k == 128)
        << layers;

    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    // BF16-exact activations keep the reference error to output rounding.
    std::vector<float> activations(static_cast<size_t>(kM) * k);
    for (int32_t m = 0; m < kM; ++m) {
      for (int32_t i = 0; i < k; ++i) {
        activations[m * k + i] = 0.125f * (m + 1) * ((i % 5) - 2);
      }
    }
    std::vector<float> actual(static_cast<size_t>(kM) * kN);
    void* device_input = nullptr;
    void* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_input, activations.size() * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_output, actual.size() * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(device_input, activations.data(),
                   activations.size() * sizeof(float), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_TRUE(
        context->setTensorAddress(built->input_names[0].c_str(), device_input));
    ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                          device_output));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_output,
                         actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int32_t m = 0; m < kM; ++m) {
      for (int32_t n = 0; n < kN; ++n) {
        float expected = 0.0f;
        for (int32_t i = 0; i < k; ++i) {
          expected += activations[m * k + i] * values[n * k + i];
        }
        expected *= scales[n];
        EXPECT_NEAR(actual[m * kN + n], expected,
                    0.02f * std::fabs(expected) + 1e-3f)
            << "m=" << m << " n=" << n;
      }
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  }
}

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
