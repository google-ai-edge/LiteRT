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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/vendors/nvidia/cache_layout.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
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

uint16_t Fp16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half = mantissa >> shift;
    const uint32_t rest = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (half & 1))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exponent) << 10) |
                  (mantissa >> 13);
  const uint32_t rest = mantissa & 0x1FFF;
  if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) {
    ++half;
  }
  return static_cast<uint16_t>(half);
}

float Fp16ToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1F;
  const uint32_t mantissa = half & 0x3FF;
  uint32_t bits = 0;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (half & 0x8000) ? -magnitude : magnitude;
  }
  if (exponent == 31) {
    bits = sign | 0x7F800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

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
    input.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, kM, k}));
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
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
        built->engine.data(), built->engine.size()));
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
    ASSERT_EQ(cudaMemcpy(device_input, activations.data(),
                         activations.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                         device_input));
    ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                         device_output));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_output,
                         actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
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

TEST(TensorRtGraphBuilderTest, CudaSubbyteGemvGroupSharesOneLaunch) {
  // Two M=1 INT4 fully connected ops reading the same activation, as Gemma's
  // q/k/v projections do. With group fusion they must build one plugin layer
  // and still match the FP32 reference; without it, one plugin per op.
  constexpr int32_t kK = 64;
  const std::array<int32_t, 2> rows = {8, 16};
  for (bool fuse : {true, false}) {
    SCOPED_TRACE(fuse);
    setenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS", "cuda_gemv", 1);
    setenv("LITERT_NVIDIA_TENSORRT_FUSE_GEMV_GROUPS", fuse ? "1" : "0", 1);
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& input = graph.EmplaceTensor();
    input.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, kK}));
    input.SetName("input");
    graph.Inputs().push_back(&input);
    std::vector<std::vector<int8_t>> values(rows.size());
    std::vector<std::vector<uint8_t>> packed(rows.size());
    std::vector<std::vector<float>> scales(rows.size());
    std::vector<LiteRtTensorT*> outputs;
    for (size_t f = 0; f < rows.size(); ++f) {
      values[f].resize(static_cast<size_t>(rows[f]) * kK);
      for (size_t i = 0; i < values[f].size(); ++i) {
        values[f][i] = static_cast<int8_t>((i * 5 + f * 3) % 16) - 8;
      }
      packed[f] = PackInt4(values[f]);
      auto& weights = graph.EmplaceTensor();
      weights.SetType(
          MakeRankedTensorType(kLiteRtElementTypeInt4, {rows[f], kK}));
      weights.SetName("weights" + std::to_string(f));
      SetWeightsFromUnownedBuffer(
          weights.Weights(),
          litert::BufferRef<uint8_t>(packed[f].data(), packed[f].size()));
      scales[f].resize(rows[f]);
      for (int32_t n = 0; n < rows[f]; ++n) {
        scales[f][n] = 0.005f * (n + 1 + f);
      }
      const std::vector<int64_t> zero_points(rows[f], 0);
      weights.SetQarams(MakePerChannelQuantization(
          scales[f], zero_points, /*quantized_dim=*/0, weights));
      auto& output = graph.EmplaceTensor();
      output.SetType(
          MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1, rows[f]}));
      output.SetName("output" + std::to_string(f));
      graph.Outputs().push_back(&output);
      outputs.push_back(&output);
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
    }

    const auto* compiler_context = LrtGetCompilerContext();
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(compiler_context, &graph));
    unsetenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS");
    unsetenv("LITERT_NVIDIA_TENSORRT_FUSE_GEMV_GROUPS");
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    ASSERT_EQ(built->output_names.size(), rows.size());

    litert::nvidia::TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
        built->engine.data(), built->engine.size()));
    ASSERT_NE(engine, nullptr);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(
        engine->createEngineInspector());
    ASSERT_NE(inspector, nullptr);
    const std::string layers = inspector->getEngineInformation(
        nvinfer1::LayerInformationFormat::kONELINE);
    size_t plugin_layers = 0;
    for (size_t at = layers.find("PluginV3"); at != std::string::npos;
         at = layers.find("PluginV3", at + 1)) {
      ++plugin_layers;
    }
    EXPECT_EQ(plugin_layers, fuse ? 1 : rows.size()) << layers;

    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    std::vector<float> activations(kK);
    for (int32_t i = 0; i < kK; ++i) {
      activations[i] = 0.25f * ((i % 7) - 3);
    }
    void* device_input = nullptr;
    ASSERT_EQ(cudaMalloc(&device_input, activations.size() * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input, activations.data(),
                         activations.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                         device_input));
    std::vector<void*> device_outputs(rows.size(), nullptr);
    for (size_t f = 0; f < rows.size(); ++f) {
      ASSERT_EQ(cudaMalloc(&device_outputs[f], rows[f] * sizeof(float)),
                cudaSuccess);
      ASSERT_TRUE(context->setTensorAddress(built->output_names[f].c_str(),
                                           device_outputs[f]));
    }
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t f = 0; f < rows.size(); ++f) {
      std::vector<float> actual(rows[f]);
      ASSERT_EQ(cudaMemcpy(actual.data(), device_outputs[f],
                           actual.size() * sizeof(float),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (int32_t n = 0; n < rows[f]; ++n) {
        float expected = 0.0f;
        for (int32_t i = 0; i < kK; ++i) {
          expected += activations[i] * values[f][n * kK + i];
        }
        expected *= scales[f][n];
        EXPECT_NEAR(actual[n], expected, 0.02f * std::fabs(expected) + 1e-3f)
            << "fc=" << f << " n=" << n;
      }
      EXPECT_EQ(cudaFree(device_outputs[f]), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  }
}

TEST(TensorRtGraphBuilderTest, Fp16SoftCapAfterCudaSubbyteGemv) {
  // Gemma 4 12B's vocabulary soft cap: an FP16-typed div/tanh/mul chain
  // consuming the BF16-only CUDA GEMV plugin output next to Float16 scalar
  // constants. The lowering must reconcile the operand types itself.
  constexpr int32_t kK = 64;
  constexpr int32_t kN = 16;
  setenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS", "cuda_gemv", 1);
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& input = graph.EmplaceTensor();
  input.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, kK}));
  input.SetName("input");
  graph.Inputs().push_back(&input);
  std::vector<int8_t> values(static_cast<size_t>(kN) * kK);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int8_t>((i * 11) % 16) - 8;
  }
  const std::vector<uint8_t> packed = PackInt4(values);
  auto& weights = graph.EmplaceTensor();
  weights.SetType(MakeRankedTensorType(kLiteRtElementTypeInt4, {kN, kK}));
  weights.SetName("weights");
  SetWeightsFromUnownedBuffer(
      weights.Weights(),
      litert::BufferRef<uint8_t>(packed.data(), packed.size()));
  const std::vector<float> scales(kN, 0.5f);
  const std::vector<int64_t> zero_points(kN, 0);
  weights.SetQarams(MakePerChannelQuantization(scales, zero_points,
                                               /*quantized_dim=*/0, weights));
  auto& fc_output = graph.EmplaceTensor();
  fc_output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, kN}));
  auto& fc = graph.EmplaceOp();
  fc.SetOpCode(kLiteRtOpCodeTflFullyConnected);
  tflite::FullyConnectedOptionsT fc_options;
  fc_options.keep_num_dims = true;
  tflite::BuiltinOptionsUnion fc_union;
  fc_union.Set(std::move(fc_options));
  litert::internal::SetTflOptions(fc, std::move(fc_union));
  litert::internal::AttachInput(&input, fc);
  litert::internal::AttachInput(&weights, fc);
  litert::internal::AttachOutput(&fc_output, fc);

  const uint16_t kHalf30 = 0x4F80;  // 30.0 in FP16.
  auto& cap = graph.EmplaceTensor();
  cap.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {}));
  cap.SetName("cap");
  SetWeightsFromUnownedBuffer(
      cap.Weights(), litert::BufferRef<uint8_t>(
                         reinterpret_cast<const uint8_t*>(&kHalf30),
                         sizeof(kHalf30)));
  auto& scaled = graph.EmplaceTensor();
  scaled.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, kN}));
  auto& div = graph.EmplaceOp();
  div.SetOpCode(kLiteRtOpCodeTflDiv);
  tflite::BuiltinOptionsUnion div_union;
  div_union.Set(tflite::DivOptionsT{});
  litert::internal::SetTflOptions(div, std::move(div_union));
  litert::internal::AttachInput(&fc_output, div);
  litert::internal::AttachInput(&cap, div);
  litert::internal::AttachOutput(&scaled, div);
  auto& squashed = graph.EmplaceTensor();
  squashed.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, kN}));
  auto& tanh = graph.EmplaceOp();
  tanh.SetOpCode(kLiteRtOpCodeTflTanh);
  litert::internal::AttachInput(&scaled, tanh);
  litert::internal::AttachOutput(&squashed, tanh);
  auto& output = graph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, kN}));
  output.SetName("output");
  graph.Outputs().push_back(&output);
  auto& mul = graph.EmplaceOp();
  mul.SetOpCode(kLiteRtOpCodeTflMul);
  tflite::BuiltinOptionsUnion mul_union;
  mul_union.Set(tflite::MulOptionsT{});
  litert::internal::SetTflOptions(mul, std::move(mul_union));
  litert::internal::AttachInput(&squashed, mul);
  litert::internal::AttachInput(&cap, mul);
  litert::internal::AttachOutput(&output, mul);

  const auto* compiler_context = LrtGetCompilerContext();
  for (auto* op : {&fc, &div, &tanh, &mul}) {
    ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(compiler_context, op)));
  }
  auto built = litert::nvidia::BuildTensorRtEngine(
      litert::compiler::Subgraph(compiler_context, &graph));
  unsetenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS");
  ASSERT_TRUE(built.HasValue()) << built.Error().Message();

  litert::nvidia::TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
      built->engine.data(), built->engine.size()));
  ASSERT_NE(engine, nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  // Half-precision-exact activations: 0.5 * ((i % 5) - 2).
  std::array<uint16_t, kK> activations{};
  std::array<float, kK> activation_values{};
  for (int32_t i = 0; i < kK; ++i) {
    const float value = 0.5f * ((i % 5) - 2);
    activation_values[i] = value;
    activations[i] = static_cast<uint16_t>(
        value == 0.0f ? 0 : (value < 0 ? 0x8000 : 0) |
            (value == 1.0f || value == -1.0f ? 0x3C00 : 0x3800));
  }
  void* device_input = nullptr;
  void* device_output = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, sizeof(activations)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_output, kN * sizeof(uint16_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_input, activations.data(), sizeof(activations),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                       device_input));
  ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                       device_output));
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_TRUE(context->enqueueV3(stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::array<uint16_t, kN> actual{};
  ASSERT_EQ(cudaMemcpy(actual.data(), device_output, sizeof(actual),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int32_t n = 0; n < kN; ++n) {
    float logit = 0.0f;
    for (int32_t i = 0; i < kK; ++i) {
      logit += activation_values[i] * values[n * kK + i];
    }
    logit *= scales[n];
    const float expected = 30.0f * std::tanh(logit / 30.0f);
    // Decode FP16 bits.
    const uint16_t bits = actual[n];
    const int exponent = (bits >> 10) & 0x1F;
    const float mantissa = 1.0f + (bits & 0x3FF) / 1024.0f;
    const float magnitude =
        exponent == 0 ? (bits & 0x3FF) / 1024.0f * std::ldexp(1.0f, -14)
                      : mantissa * std::ldexp(1.0f, exponent - 15);
    const float value = (bits & 0x8000) ? -magnitude : magnitude;
    EXPECT_NEAR(value, expected, 0.02f * std::fabs(expected) + 0.05f)
        << "n=" << n;
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  EXPECT_EQ(cudaFree(device_output), cudaSuccess);
  EXPECT_EQ(cudaFree(device_input), cudaSuccess);
}

TEST(TensorRtGraphBuilderTest, InPlaceKvCacheUpdate) {
  // A subgraph-input cache updated by DynamicUpdateSlice into a subgraph
  // output lowers to TensorRT's in-place KVCacheUpdate: the output aliases
  // the input and the update lands at the runtime position. Covers the
  // [B, H, S, D] key layout (axis 2) and the transposed [B, H, D, S] value
  // layout (axis 3), which the engine either views as [B, H*D, S, 1]
  // (native layout) or holds as [B, H, S, D] (the default); with the feature
  // disabled the output is not aliased. Every path must clamp runtime starts
  // like TFLite, including multi-token updates at either end of the cache.
  struct Case {
    bool native;
    int axis;
    bool transposed;
    std::vector<int32_t> cache_dims;
    std::vector<int32_t> update_dims;
    std::vector<int32_t> engine_dims;
  };
  const std::vector<Case> cases = {
      {true, 2, false, {1, 2, 16, 8}, {1, 2, 1, 8}, {1, 2, 16, 8}},
      {true, 3, false, {1, 2, 8, 16}, {1, 2, 8, 1}, {1, 16, 16, 1}},
      {true, 3, true, {1, 2, 8, 16}, {1, 2, 8, 1}, {1, 2, 16, 8}},
      {false, 2, false, {1, 2, 16, 8}, {1, 2, 1, 8}, {1, 2, 16, 8}},
      {false, 3, false, {1, 2, 8, 16}, {1, 2, 8, 1}, {1, 2, 8, 16}},
      {true, 2, false, {1, 2, 16, 8}, {1, 2, 4, 8}, {1, 2, 16, 8}},
      {true, 3, false, {1, 2, 8, 16}, {1, 2, 8, 4}, {1, 16, 16, 1}},
      {true, 3, true, {1, 2, 8, 16}, {1, 2, 8, 4}, {1, 2, 16, 8}},
      {false, 2, false, {1, 2, 16, 8}, {1, 2, 4, 8}, {1, 2, 16, 8}},
      {false, 3, false, {1, 2, 8, 16}, {1, 2, 8, 4}, {1, 2, 8, 16}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "native=" << test_case.native << " axis=" << test_case.axis
                 << " transposed=" << test_case.transposed);
    setenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE",
           test_case.native ? "1" : "0", 1);
    setenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT",
           test_case.transposed ? "" : "native", 1);
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& cache = graph.EmplaceTensor();
    cache.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, test_case.cache_dims));
    cache.SetName("cache");
    graph.Inputs().push_back(&cache);
    auto& update = graph.EmplaceTensor();
    update.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        test_case.update_dims));
    update.SetName("update");
    graph.Inputs().push_back(&update);
    auto& start = graph.EmplaceTensor();
    start.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    start.SetName("start");
    graph.Inputs().push_back(&start);
    auto& output = graph.EmplaceTensor();
    output.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, test_case.cache_dims));
    output.SetName("cache_out");
    graph.Outputs().push_back(&output);
    auto& dus = graph.EmplaceOp();
    dus.SetOpCode(kLiteRtOpCodeTflDynamicUpdateSlice);
    litert::internal::AttachInput(&cache, dus);
    litert::internal::AttachInput(&update, dus);
    litert::internal::AttachInput(&start, dus);
    litert::internal::AttachOutput(&output, dus);

    const auto* compiler_context = LrtGetCompilerContext();
    ASSERT_TRUE(litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(compiler_context, &dus)));
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(compiler_context, &graph));
    unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE");
    unsetenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT");
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();

    litert::nvidia::TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
        built->engine.data(), built->engine.size()));
    ASSERT_NE(engine, nullptr);
    const char* aliased =
        engine->getAliasedInputTensor(built->output_names[0].c_str());
    if (test_case.native) {
      ASSERT_NE(aliased, nullptr);
      EXPECT_EQ(std::string(aliased), built->input_names[0]);
    } else {
      EXPECT_EQ(aliased, nullptr);
    }
    const auto engine_dims =
        engine->getTensorShape(built->input_names[0].c_str());
    ASSERT_EQ(engine_dims.nbDims, 4);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(engine_dims.d[i], test_case.engine_dims[i]) << "dim " << i;
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);

    const int seq_len = test_case.cache_dims[test_case.axis];
    const int depth = test_case.cache_dims[test_case.axis == 2 ? 3 : 2];
    const int update_length = test_case.update_dims[test_case.axis];
    const size_t cache_elements = 2 * seq_len * depth;
    const size_t update_elements = 2 * update_length * depth;
    std::vector<uint16_t> cache_host(cache_elements, 0);
    std::vector<uint16_t> update_host(update_elements);
    const auto update_value = [](int head, int row, int d) {
      return Fp16Bits(static_cast<float>(head * 100 + row * 10 + d + 1));
    };
    for (int h = 0; h < 2; ++h) {
      for (int row = 0; row < update_length; ++row) {
        for (int d = 0; d < depth; ++d) {
          const size_t index = test_case.axis == 2
                                   ? (h * update_length + row) * depth + d
                                   : (h * depth + d) * update_length + row;
          update_host[index] = update_value(h, row, d);
        }
      }
    }
    // Full-extent axes clamp to zero regardless of their supplied indices.
    std::array<int32_t, 4> start_host = {99, -99, 77, -77};
    void* device_cache = nullptr;
    void* device_update = nullptr;
    void* device_start = nullptr;
    void* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_cache, cache_elements * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_update, update_elements * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_start, sizeof(start_host)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_update, update_host.data(),
                         update_elements * 2, cudaMemcpyHostToDevice),
              cudaSuccess);
    if (test_case.native) {
      device_output = device_cache;  // in place, as LiteRT-LM binds it
    } else {
      ASSERT_EQ(cudaMalloc(&device_output, cache_elements * 2), cudaSuccess);
    }
    ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                         device_cache));
    ASSERT_TRUE(context->setTensorAddress(built->input_names[1].c_str(),
                                         device_update));
    ASSERT_TRUE(context->setTensorAddress(built->input_names[2].c_str(),
                                         device_start));
    ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                         device_output));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    std::vector<uint16_t> actual(cache_elements);
    // Elements are addressed by the engine layout: [B, H, S, D] for keys and
    // for values held transposed, [B, H, D, S] for values in the model layout.
    const bool seq_major = test_case.axis == 2 || test_case.transposed;
    const int max_start = seq_len - update_length;
    for (int32_t position : {std::numeric_limits<int32_t>::min(), -1, 0, 5,
                             max_start, max_start + 1,
                             std::numeric_limits<int32_t>::max()}) {
      SCOPED_TRACE(::testing::Message() << "position=" << position
                                       << " update_length=" << update_length);
      start_host[test_case.axis] = position;
      ASSERT_EQ(cudaMemcpy(device_cache, cache_host.data(), cache_elements * 2,
                             cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(device_start, start_host.data(), sizeof(start_host),
                             cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_TRUE(context->enqueueV3(stream));
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actual.data(), device_output, cache_elements * 2,
                             cudaMemcpyDeviceToHost), cudaSuccess);
      const int clamped_start = std::clamp(position, 0, max_start);
      for (int h = 0; h < 2; ++h) {
        for (int seq = 0; seq < seq_len; ++seq) {
          for (int d = 0; d < depth; ++d) {
            const uint16_t expected =
                seq >= clamped_start && seq < clamped_start + update_length
                    ? update_value(h, seq - clamped_start, d)
                    : 0;
            const size_t index = seq_major
                                     ? (h * seq_len + seq) * depth + d
                                     : (h * depth + d) * seq_len + seq;
            ASSERT_EQ(actual[index], expected)
                << "h=" << h << " seq=" << seq << " d=" << d;
          }
        }
      }
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    if (!test_case.native) {
      EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    }
    EXPECT_EQ(cudaFree(device_start), cudaSuccess);
    EXPECT_EQ(cudaFree(device_update), cudaSuccess);
    EXPECT_EQ(cudaFree(device_cache), cudaSuccess);
  }
}

TEST(TensorRtGraphBuilderTest, Fp16CacheUpdateSupportRequiresPrefillReaders) {
  enum class Reader { kNone, kOldBmm, kOldRuntimeBmm, kUpdatedBmm, kOldView };
  struct Case {
    int rows;
    Reader reader;
    bool value;
    bool supported;
  };
  const Case cases[] = {
      {1, Reader::kNone, false, false},
      {4, Reader::kNone, false, true},
      {4, Reader::kOldBmm, false, true},
      {4, Reader::kOldBmm, true, true},
      {4, Reader::kOldRuntimeBmm, false, true},
      {4, Reader::kOldRuntimeBmm, true, true},
      {4, Reader::kUpdatedBmm, false, false},
      {4, Reader::kUpdatedBmm, true, false},
      {4, Reader::kOldView, false, false},
      {4, Reader::kOldView, true, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(testing::Message()
                 << "rows=" << test_case.rows
                 << " reader=" << static_cast<int>(test_case.reader)
                 << " value=" << test_case.value);
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    const auto tensor = [&](LiteRtElementType type,
                            const std::vector<int32_t>& dims) -> LiteRtTensorT& {
      auto& value = graph.EmplaceTensor();
      value.SetType(MakeRankedTensorType(type, dims));
      return value;
    };
    const std::vector<int32_t> k_dims{1, 2, 8, 4};
    const std::vector<int32_t> v_dims{1, 2, 4, 8};
    auto& update_k = tensor(kLiteRtElementTypeFloat16, {1, 2, test_case.rows, 4});
    auto& update_v = tensor(kLiteRtElementTypeFloat16, {1, 2, test_case.rows, 4});
    auto& params = tensor(kLiteRtElementTypeInt32, {1, 1, 1, 7});
    auto& cache_k = tensor(kLiteRtElementTypeFloat16, k_dims);
    auto& cache_v = tensor(kLiteRtElementTypeFloat16, v_dims);
    auto& start_k = tensor(kLiteRtElementTypeInt32, {4});
    auto& start_v = tensor(kLiteRtElementTypeInt32, {4});
    graph.Inputs() = {&update_k, &update_v, &params, &cache_k,
                      &cache_v, &start_k, &start_v};
    auto& output_k = tensor(kLiteRtElementTypeFloat16, k_dims);
    auto& output_v = tensor(kLiteRtElementTypeFloat16, v_dims);
    graph.Outputs() = {&output_k, &output_v};
    auto& update = graph.EmplaceOp();
    update.SetOpCode(kLiteRtOpCodeShloComposite);
    tflite::StableHLOCompositeOptionsT composite;
    composite.name = "odml.cache_update";
    flexbuffers::Builder attributes;
    const auto map = attributes.StartMap();
    attributes.Bool("is_ring_buffer", true);
    attributes.Int("cache_size", 8);
    attributes.Int("head_size", 4);
    attributes.EndMap(map);
    attributes.Finish();
    composite.composite_attributes = attributes.GetBuffer();
    tflite::BuiltinOptions2Union options;
    options.Set(std::move(composite));
    litert::internal::SetTflOptions2(update, std::move(options));
    for (auto* input : graph.Inputs()) {
      litert::internal::AttachInput(input, update);
    }
    litert::internal::AttachOutput(&output_k, update);
    litert::internal::AttachOutput(&output_v, update);

    if (test_case.reader != Reader::kNone) {
      auto& old_cache = test_case.value ? cache_v : cache_k;
      auto& updated_cache = test_case.value ? output_v : output_k;
      auto& rhs =
          test_case.reader == Reader::kUpdatedBmm ? updated_cache : old_cache;
      auto& reader = graph.EmplaceOp();
      if (test_case.reader == Reader::kOldView) {
        // A reshape aliases old storage; it is not a completed cache read.
        const auto& dims = test_case.value ? v_dims : k_dims;
        auto& view = tensor(kLiteRtElementTypeFloat16, dims);
        reader.SetOpCode(kLiteRtOpCodeTflReshape);
        tflite::ReshapeOptionsT reshape;
        reshape.new_shape = dims;
        tflite::BuiltinOptionsUnion options;
        options.Set(std::move(reshape));
        litert::internal::SetTflOptions(reader, std::move(options));
        litert::internal::AttachInput(&rhs, reader);
        litert::internal::AttachOutput(&view, reader);
        graph.Outputs().push_back(&view);
      } else {
        auto& lhs = tensor(kLiteRtElementTypeFloat16,
                           {1, 2, 1, test_case.value ? 8 : 4});
        auto& result = tensor(kLiteRtElementTypeFloat16,
                             {1, 2, 1, test_case.value ? 4 : 8});
        graph.Inputs().push_back(&lhs);
        if (test_case.reader == Reader::kOldRuntimeBmm) {
          reader.SetOpCode(kLiteRtOpCodeShloComposite);
          tflite::StableHLOCompositeOptionsT composite;
          composite.name = "odml.runtime_bmm";
          tflite::BuiltinOptions2Union options;
          options.Set(std::move(composite));
          litert::internal::SetTflOptions2(reader, std::move(options));
        } else {
          reader.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
          tflite::BatchMatMulOptionsT matmul;
          matmul.adj_y = true;
          tflite::BuiltinOptionsUnion options;
          options.Set(std::move(matmul));
          litert::internal::SetTflOptions(reader, std::move(options));
        }
        litert::internal::AttachInput(&lhs, reader);
        litert::internal::AttachInput(&rhs, reader);
        if (test_case.reader == Reader::kOldRuntimeBmm) {
          litert::internal::AttachInput(&params, reader);
        }
        litert::internal::AttachOutput(&result, reader);
        graph.Outputs().push_back(&result);
      }
    }
    setenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES",
           "cache_update,runtime_bmm", 1);
    setenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE", "1", 1);
    const bool supported = litert::nvidia::IsTensorRtOpSupported(
        litert::compiler::Op(LrtGetCompilerContext(), &update));
    unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES");
    unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE");
    EXPECT_EQ(supported, test_case.supported);
  }
}

TEST(TensorRtGraphBuilderTest, Fp16CompositeCacheUpdatePreservesAttentionReaders) {
  constexpr int kHeads = 2;
  constexpr int kSeq = 8;
  constexpr int kDepth = 4;
  constexpr int kRows = 4;
  constexpr size_t kCacheElements = kHeads * kSeq * kDepth;
  constexpr size_t kUpdateElements = kHeads * kRows * kDepth;
  const std::vector<int32_t> k_dims{1, kHeads, kSeq, kDepth};
  const std::vector<int32_t> v_dims{1, kHeads, kDepth, kSeq};
  const std::vector<int32_t> update_dims{1, kHeads, kRows, kDepth};
  for (bool ring : {false, true}) {
    for (bool transposed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "ring=" << ring
                                     << " transposed=" << transposed);
      setenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES", "cache_update", 1);
      setenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE", "1", 1);
      setenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT",
             transposed ? "" : "native", 1);
      LiteRtModelT model;
      auto& graph = model.EmplaceSubgraph();
      const auto tensor = [&](const char* name, LiteRtElementType type,
                              const std::vector<int32_t>& dims)
          -> LiteRtTensorT& {
        auto& value = graph.EmplaceTensor();
        value.SetName(name);
        value.SetType(MakeRankedTensorType(type, dims));
        return value;
      };
      auto& update_k = tensor("update_k", kLiteRtElementTypeFloat16, update_dims);
      auto& update_v = tensor("update_v", kLiteRtElementTypeFloat16, update_dims);
      auto& params = tensor("params", kLiteRtElementTypeInt32, {1, 1, 1, 7});
      auto& cache_k = tensor("cache_k", kLiteRtElementTypeFloat16, k_dims);
      auto& cache_v = tensor("cache_v", kLiteRtElementTypeFloat16, v_dims);
      auto& start_k = tensor("start_k", kLiteRtElementTypeInt32, {4});
      auto& start_v = tensor("start_v", kLiteRtElementTypeInt32, {4});
      auto& query = tensor("query", kLiteRtElementTypeFloat16,
                           {1, kHeads, 1, kDepth});
      auto& probabilities = tensor("probabilities", kLiteRtElementTypeFloat16,
                                   {1, kHeads, 1, kSeq});
      graph.Inputs() = {&update_k, &update_v, &params, &cache_k, &cache_v,
                        &start_k, &start_v, &query, &probabilities};
      auto& output_k = tensor("output_k", kLiteRtElementTypeFloat16, k_dims);
      auto& output_v = tensor("output_v", kLiteRtElementTypeFloat16, v_dims);
      graph.Outputs() = {&output_k, &output_v};
      const auto reader = [&](LiteRtTensorT& lhs, LiteRtTensorT& rhs,
                              const char* name, int width) {
        auto& output = tensor(name, kLiteRtElementTypeFloat16,
                              {1, kHeads, 1, width});
        auto& op = graph.EmplaceOp();
        op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
        tflite::BatchMatMulOptionsT matmul;
        matmul.adj_y = true;
        tflite::BuiltinOptionsUnion options;
        options.Set(std::move(matmul));
        litert::internal::SetTflOptions(op, std::move(options));
        litert::internal::AttachInput(&lhs, op);
        litert::internal::AttachInput(&rhs, op);
        litert::internal::AttachOutput(&output, op);
        graph.Outputs().push_back(&output);
      };
      // Prefill reads old history and exports the updated caches. The native
      // write must wait for these readers and for patch preparation.
      reader(query, cache_k, "old_scores", kSeq);
      reader(probabilities, cache_v, "old_values", kDepth);
      auto& update = graph.EmplaceOp();
      update.SetOpCode(kLiteRtOpCodeShloComposite);
      tflite::StableHLOCompositeOptionsT composite;
      composite.name = "odml.cache_update";
      flexbuffers::Builder attributes;
      const auto map = attributes.StartMap();
      attributes.Bool("is_ring_buffer", ring);
      attributes.Int("cache_size", kSeq);
      attributes.Int("head_size", kDepth);
      attributes.EndMap(map);
      attributes.Finish();
      composite.composite_attributes = attributes.GetBuffer();
      tflite::BuiltinOptions2Union options;
      options.Set(std::move(composite));
      litert::internal::SetTflOptions2(update, std::move(options));
      for (auto* input : {&update_k, &update_v, &params, &cache_k, &cache_v,
                          &start_k, &start_v}) {
        litert::internal::AttachInput(input, update);
      }
      litert::internal::AttachOutput(&output_k, update);
      litert::internal::AttachOutput(&output_v, update);

      const auto* compiler_context = LrtGetCompilerContext();
      const bool supported = litert::nvidia::IsTensorRtOpSupported(
          litert::compiler::Op(compiler_context, &update));
      auto built = litert::nvidia::BuildTensorRtEngine(
          litert::compiler::Subgraph(compiler_context, &graph));
      unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES");
      unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE");
      unsetenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT");
      ASSERT_TRUE(supported);
      ASSERT_TRUE(built.HasValue()) << built.Error().Message();
      ASSERT_EQ(built->input_names.size(), 9);
      ASSERT_EQ(built->output_names.size(), 4);
      EXPECT_EQ(litert::nvidia::IsTransposedValueCacheTensor(
                    built->input_names[4]),
                transposed);
      EXPECT_EQ(litert::nvidia::IsTransposedValueCacheTensor(
                    built->output_names[1]),
                transposed);
      EXPECT_FALSE(litert::nvidia::IsTransposedValueCacheTensor(
          built->input_names[3]));

      litert::nvidia::TensorRtLogger logger;
      std::unique_ptr<nvinfer1::IRuntime> runtime(
          nvinfer1::createInferRuntime(logger));
      ASSERT_NE(runtime, nullptr);
      std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
          built->engine.data(), built->engine.size()));
      ASSERT_NE(engine, nullptr);
      EXPECT_EQ(engine->getTensorLocation(built->input_names[2].c_str()),
                nvinfer1::TensorLocation::kDEVICE);
      EXPECT_FALSE(engine->isShapeInferenceIO(built->input_names[2].c_str()));
      const auto v_shape = engine->getTensorShape(built->input_names[4].c_str());
      ASSERT_EQ(v_shape.nbDims, 4);
      const std::vector<int32_t> native_v_dims{1, kHeads * kDepth, kSeq, 1};
      const auto& expected_shape = transposed ? k_dims : native_v_dims;
      for (int i = 0; i < 4; ++i) EXPECT_EQ(v_shape.d[i], expected_shape[i]);
      std::unique_ptr<nvinfer1::IExecutionContext> context(
          engine->createExecutionContext());
      ASSERT_NE(context, nullptr);
      const std::array<size_t, 9> input_sizes{
          kUpdateElements * 2, kUpdateElements * 2, 7 * sizeof(int32_t),
          kCacheElements * 2, kCacheElements * 2, 4 * sizeof(int32_t),
          4 * sizeof(int32_t), kHeads * kDepth * 2, kHeads * kSeq * 2};
      const std::array<size_t, 4> output_sizes{
          kCacheElements * 2, kCacheElements * 2, kHeads * kSeq * 2,
          kHeads * kDepth * 2};
      std::array<void*, 9> inputs{};
      std::array<void*, 4> outputs{};
      std::array<bool, 4> output_owns_memory{};
      for (size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_EQ(cudaMalloc(&inputs[i], input_sizes[i]), cudaSuccess);
        ASSERT_TRUE(context->setTensorAddress(built->input_names[i].c_str(),
                                             inputs[i]));
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        const char* alias = engine->getAliasedInputTensor(
            built->output_names[i].c_str());
        if (i < 2) ASSERT_NE(alias, nullptr);
        if (alias != nullptr) {
          const auto it = std::find(built->input_names.begin(),
                                    built->input_names.end(), alias);
          ASSERT_NE(it, built->input_names.end());
          outputs[i] = inputs[it - built->input_names.begin()];
        } else {
          ASSERT_EQ(cudaMalloc(&outputs[i], output_sizes[i]), cudaSuccess);
          output_owns_memory[i] = true;
        }
        ASSERT_TRUE(context->setTensorAddress(built->output_names[i].c_str(),
                                             outputs[i]));
      }
      cudaStream_t stream = nullptr;
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
      std::vector<uint16_t> query_host(kHeads * kDepth, Fp16Bits(1.0f));
      std::vector<uint16_t> probability_host(kHeads * kSeq, Fp16Bits(1.0f));
      ASSERT_EQ(cudaMemcpy(inputs[7], query_host.data(), input_sizes[7],
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(inputs[8], probability_host.data(), input_sizes[8],
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      // The FP16 composite must use params, not its decomposition's start inputs.
      const std::array<int32_t, 4> ignored_starts{111, -222, 333, -444};
      for (int i : {5, 6}) {
        ASSERT_EQ(cudaMemcpy(inputs[i], ignored_starts.data(), input_sizes[i],
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      const auto v_index = [&](int h, int s, int d) {
        return transposed ? (h * kSeq + s) * kDepth + d
                          : (h * kDepth + d) * kSeq + s;
      };
      // Zero length, an interior full update, and a partially valid update.
      // The ring cases cross the end of the cache. Unused rows contain sentinels.
      for (const auto [write_index, valid_length] :
           {std::pair<int, int>{2, 0}, {2, kRows},
            {ring ? kSeq - 1 : kSeq - 3, 2},
            {ring ? kSeq - 1 : kSeq - kRows, kRows}, {kSeq - 1, 1}}) {
        SCOPED_TRACE(testing::Message() << "write=" << write_index
                                       << " valid=" << valid_length);
        std::vector<uint16_t> old_k(kCacheElements), old_v(kCacheElements);
        for (int h = 0; h < kHeads; ++h) {
          for (int s = 0; s < kSeq; ++s) {
            for (int d = 0; d < kDepth; ++d) {
              old_k[(h * kSeq + s) * kDepth + d] =
                  Fp16Bits(h * 16.0f + s * 2 + d);
              old_v[v_index(h, s, d)] =
                  Fp16Bits(100.0f + h * 16 + s * 2 + d);
            }
          }
        }
        auto expected_k = old_k;
        auto expected_v = old_v;
        std::vector<uint16_t> update_k_host(kUpdateElements);
        std::vector<uint16_t> update_v_host(kUpdateElements);
        for (int h = 0; h < kHeads; ++h) {
          for (int row = 0; row < kRows; ++row) {
            for (int d = 0; d < kDepth; ++d) {
              const int index = (h * kRows + row) * kDepth + d;
              update_k_host[index] =
                  Fp16Bits(row < valid_length ? 200.0f + h * 10 + row * 2 + d
                                             : 800.0f + row);
              update_v_host[index] =
                  Fp16Bits(row < valid_length ? 220.0f + h * 10 + row * 2 + d
                                             : 900.0f + row);
              if (row < valid_length) {
                const int s = ring ? (write_index + row) % kSeq
                                   : write_index + row;
                expected_k[(h * kSeq + s) * kDepth + d] = update_k_host[index];
                expected_v[v_index(h, s, d)] = update_v_host[index];
              }
            }
          }
        }
        const std::array<int32_t, 7> params_host{
            write_index, 0, 0, valid_length, 0, 0, 0};
        const std::array<const void*, 5> host_inputs{
            update_k_host.data(), update_v_host.data(), params_host.data(),
            old_k.data(), old_v.data()};
        for (int i = 0; i < 5; ++i) {
          ASSERT_EQ(cudaMemcpy(inputs[i], host_inputs[i], input_sizes[i],
                               cudaMemcpyHostToDevice),
                    cudaSuccess);
        }
        ASSERT_TRUE(context->enqueueV3(stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::array<std::vector<uint16_t>, 4> actual;
        for (size_t i = 0; i < actual.size(); ++i) {
          actual[i].resize(output_sizes[i] / 2);
          ASSERT_EQ(cudaMemcpy(actual[i].data(), outputs[i], output_sizes[i],
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
        }
        EXPECT_EQ(actual[0], expected_k);
        EXPECT_EQ(actual[1], expected_v);
        for (int h = 0; h < kHeads; ++h) {
          for (int s = 0; s < kSeq; ++s) {
            float sum = 0;
            for (int d = 0; d < kDepth; ++d) {
              sum += Fp16ToFloat(old_k[(h * kSeq + s) * kDepth + d]);
            }
            EXPECT_EQ(actual[2][h * kSeq + s], Fp16Bits(sum));
          }
          for (int d = 0; d < kDepth; ++d) {
            float sum = 0;
            for (int s = 0; s < kSeq; ++s) {
              sum += Fp16ToFloat(old_v[v_index(h, s, d)]);
            }
            EXPECT_EQ(actual[3][h * kDepth + d], Fp16Bits(sum));
          }
        }
      }
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (output_owns_memory[i]) EXPECT_EQ(cudaFree(outputs[i]), cudaSuccess);
      }
      for (void* input : inputs) EXPECT_EQ(cudaFree(input), cudaSuccess);
    }
  }
}

TEST(TensorRtGraphBuilderTest, PrefillAttentionJoinSurvivesCacheUpdate) {
  constexpr int kHeads = 2;
  constexpr int kSeq = 8;
  constexpr int kRows = 4;
  constexpr int kDepth = 4;
  constexpr int kColumns = kSeq + kRows;
  constexpr size_t kCacheElements = kHeads * kSeq * kDepth;
  constexpr size_t kUpdateElements = kHeads * kRows * kDepth;
  constexpr size_t kScoreElements = kHeads * kRows * kSeq;
  const std::vector<int32_t> k_dims{1, kHeads, kSeq, kDepth};
  const std::vector<int32_t> v_dims{1, kHeads, kDepth, kSeq};
  const std::vector<int32_t> update_dims{1, kHeads, kRows, kDepth};
  const std::vector<int32_t> score_dims{1, kHeads, kRows, kColumns};
  enum class JoinUse {
    kClosed,
    kEscapingScores,
    kEarlyConsumer,
    kEscapingIntermediate,
    kMultipleConsumers,
    kGraphOutput,
  };
  for (bool transposed : {false, true}) {
    for (JoinUse use : {JoinUse::kClosed, JoinUse::kEscapingScores,
                        JoinUse::kEarlyConsumer, JoinUse::kEscapingIntermediate,
                        JoinUse::kMultipleConsumers, JoinUse::kGraphOutput}) {
      SCOPED_TRACE(testing::Message() << "transposed=" << transposed
                                     << " join_use=" << static_cast<int>(use));
      const bool escape_scores = use == JoinUse::kEscapingScores;
      const bool forward_join = use == JoinUse::kClosed ||
                                use == JoinUse::kMultipleConsumers ||
                                use == JoinUse::kGraphOutput;
      LiteRtModelT model;
      auto& graph = model.EmplaceSubgraph();
      const auto tensor = [&](const char* name, LiteRtElementType type,
                              const std::vector<int32_t>& dims)
          -> LiteRtTensorT& {
        auto& value = graph.EmplaceTensor();
        value.SetName(name);
        value.SetType(MakeRankedTensorType(type, dims));
        return value;
      };
      const auto constant = [&](const char* name, LiteRtElementType type,
                                const std::vector<int32_t>& dims,
                                const void* data, size_t bytes)
          -> LiteRtTensorT& {
        auto& value = tensor(name, type, dims);
        SetWeightsFromUnownedBuffer(
            value.Weights(), litert::BufferRef<uint8_t>(
                                 static_cast<const uint8_t*>(data), bytes));
        return value;
      };
      auto& update_k = tensor("update_k", kLiteRtElementTypeFloat16, update_dims);
      auto& update_v = tensor("update_v", kLiteRtElementTypeFloat16, update_dims);
      auto& params = tensor("params", kLiteRtElementTypeInt32, {1, 1, 1, 7});
      auto& cache_k = tensor("cache_k", kLiteRtElementTypeFloat16, k_dims);
      auto& cache_v = tensor("cache_v", kLiteRtElementTypeFloat16, v_dims);
      auto& start_k = tensor("start_k", kLiteRtElementTypeInt32, {4});
      auto& start_v = tensor("start_v", kLiteRtElementTypeInt32, {4});
      auto& query = tensor("query", kLiteRtElementTypeFloat16, update_dims);
      auto& mask = tensor("mask", kLiteRtElementTypeBool, score_dims);
      graph.Inputs() = {&update_k, &update_v, &params, &cache_k, &cache_v,
                        &start_k, &start_v, &query, &mask};
      auto& output_k = tensor("output_k", kLiteRtElementTypeFloat16, k_dims);
      auto& output_v = tensor("output_v", kLiteRtElementTypeFloat16, v_dims);
      graph.Outputs() = {&output_k, &output_v};
      const auto matmul = [&](LiteRtTensorT& lhs, LiteRtTensorT& rhs,
                              const char* name, int width, bool transpose_rhs)
          -> LiteRtTensorT& {
        auto& result = tensor(name, kLiteRtElementTypeFloat16,
                              {1, kHeads, kRows, width});
        auto& op = graph.EmplaceOp();
        op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
        tflite::BatchMatMulOptionsT matmul;
        matmul.adj_y = transpose_rhs;
        tflite::BuiltinOptionsUnion options;
        options.Set(std::move(matmul));
        litert::internal::SetTflOptions(op, std::move(options));
        litert::internal::AttachInput(&lhs, op);
        litert::internal::AttachInput(&rhs, op);
        litert::internal::AttachOutput(&result, op);
        return result;
      };
      auto& old_scores = matmul(query, cache_k, "old_scores", kSeq, true);
      auto& fresh_scores =
          matmul(query, update_k, "fresh_scores", kRows, true);
      if (escape_scores) graph.Outputs().push_back(&old_scores);
      auto& scores = tensor("scores", kLiteRtElementTypeFloat16, score_dims);
      auto& concat = graph.EmplaceOp();
      concat.SetOpCode(kLiteRtOpCodeTflConcatenation);
      tflite::ConcatenationOptionsT concatenation;
      concatenation.axis = 3;
      tflite::BuiltinOptionsUnion concat_options;
      concat_options.Set(std::move(concatenation));
      litert::internal::SetTflOptions(concat, std::move(concat_options));
      litert::internal::AttachInput(&old_scores, concat);
      litert::internal::AttachInput(&fresh_scores, concat);
      litert::internal::AttachOutput(&scores, concat);

      const uint16_t negative_infinity = Fp16Bits(
          -std::numeric_limits<float>::infinity());
      auto& fill = constant("fill", kLiteRtElementTypeFloat16, {},
                            &negative_infinity, sizeof(negative_infinity));
      auto& masked = tensor("masked", kLiteRtElementTypeFloat16, score_dims);
      auto& select = graph.EmplaceOp();
      select.SetOpCode(kLiteRtOpCodeTflSelectV2);
      litert::internal::AttachInput(&mask, select);
      litert::internal::AttachInput(&scores, select);
      litert::internal::AttachInput(&fill, select);
      litert::internal::AttachOutput(&masked, select);
      auto& probabilities =
          tensor("probabilities", kLiteRtElementTypeFloat16, score_dims);
      auto& softmax = graph.EmplaceOp();
      softmax.SetOpCode(kLiteRtOpCodeTflSoftmax);
      tflite::SoftmaxOptionsT softmax_options;
      softmax_options.beta = 1.0f;
      tflite::BuiltinOptionsUnion options;
      options.Set(std::move(softmax_options));
      litert::internal::SetTflOptions(softmax, std::move(options));
      litert::internal::AttachInput(&masked, softmax);
      litert::internal::AttachOutput(&probabilities, softmax);
      const std::array<std::array<int32_t, 4>, 2> begins{
          std::array<int32_t, 4>{0, 0, 0, 0}, {0, 0, 0, kSeq}};
      const std::array<std::array<int32_t, 4>, 2> sizes{
          std::array<int32_t, 4>{1, kHeads, kRows, kSeq},
          {1, kHeads, kRows, kRows}};
      std::array<LiteRtTensorT*, 2> sliced{};
      for (int i = 0; i < 2; ++i) {
        auto& begin = constant(i == 0 ? "old_begin" : "fresh_begin",
                               kLiteRtElementTypeInt32, {4}, begins[i].data(),
                               sizeof(begins[i]));
        auto& size = constant(i == 0 ? "old_size" : "fresh_size",
                              kLiteRtElementTypeInt32, {4}, sizes[i].data(),
                              sizeof(sizes[i]));
        auto& result = tensor(i == 0 ? "old_probs" : "fresh_probs",
                              kLiteRtElementTypeFloat16,
                              {1, kHeads, kRows, sizes[i][3]});
        auto& slice = graph.EmplaceOp();
        slice.SetOpCode(kLiteRtOpCodeTflSlice);
        litert::internal::AttachInput(&probabilities, slice);
        litert::internal::AttachInput(&begin, slice);
        litert::internal::AttachInput(&size, slice);
        litert::internal::AttachOutput(&result, slice);
        sliced[i] = &result;
      }
      auto& old_values =
          matmul(*sliced[0], cache_v, "old_values", kDepth, true);
      auto& fresh_values =
          matmul(*sliced[1], update_v, "fresh_values", kDepth, false);
      const auto add = [&](LiteRtTensorT& lhs, LiteRtTensorT& rhs,
                           const char* name) -> LiteRtTensorT& {
        auto& result = tensor(name, kLiteRtElementTypeFloat16, update_dims);
        auto& op = graph.EmplaceOp();
        op.SetOpCode(kLiteRtOpCodeTflAdd);
        tflite::BuiltinOptionsUnion options;
        options.Set(tflite::AddOptionsT{});
        litert::internal::SetTflOptions(op, std::move(options));
        litert::internal::AttachInput(&lhs, op);
        litert::internal::AttachInput(&rhs, op);
        litert::internal::AttachOutput(&result, op);
        return result;
      };
      const uint16_t one = Fp16Bits(1.0f);
      auto& bias = constant("bias", kLiteRtElementTypeFloat16, {}, &one,
                            sizeof(one));
      auto& join = add(old_values, fresh_values, "attention_join");
      if (use == JoinUse::kEarlyConsumer) {
        // Remapping the join cannot redirect an already lowered consumer.
        auto& early = add(join, bias, "early_attention");
        graph.Outputs().push_back(&early);
      } else if (use == JoinUse::kGraphOutput) {
        // MarkOutputs must export the forwarded join, as well as forwarding
        // it to consumers lowered after the cache update.
        graph.Outputs().push_back(&join);
      }
      auto& update = graph.EmplaceOp();
      update.SetOpCode(kLiteRtOpCodeShloComposite);
      tflite::StableHLOCompositeOptionsT composite;
      composite.name = "odml.cache_update";
      flexbuffers::Builder attributes;
      const auto map = attributes.StartMap();
      attributes.Bool("is_ring_buffer", true);
      attributes.Int("cache_size", kSeq);
      attributes.Int("head_size", kDepth);
      attributes.EndMap(map);
      attributes.Finish();
      composite.composite_attributes = attributes.GetBuffer();
      tflite::BuiltinOptions2Union composite_options;
      composite_options.Set(std::move(composite));
      litert::internal::SetTflOptions2(update, std::move(composite_options));
      for (auto* input : {&update_k, &update_v, &params, &cache_k, &cache_v,
                          &start_k, &start_v}) {
        litert::internal::AttachInput(input, update);
      }
      litert::internal::AttachOutput(&output_k, update);
      litert::internal::AttachOutput(&output_v, update);
      if (use == JoinUse::kEscapingIntermediate) {
        // Both attention branches reach the join, but an intermediate also
        // escapes it through a later consumer. This needs full read barriers.
        auto& extra = tensor("squared_probabilities", kLiteRtElementTypeFloat16,
                             score_dims);
        auto& multiply = graph.EmplaceOp();
        multiply.SetOpCode(kLiteRtOpCodeTflMul);
        tflite::BuiltinOptionsUnion multiply_options;
        multiply_options.Set(tflite::MulOptionsT{});
        litert::internal::SetTflOptions(multiply, std::move(multiply_options));
        litert::internal::AttachInput(&probabilities, multiply);
        litert::internal::AttachInput(&probabilities, multiply);
        litert::internal::AttachOutput(&extra, multiply);
        graph.Outputs().push_back(&extra);
      } else if (use == JoinUse::kMultipleConsumers) {
        auto& extra = add(join, join, "double_attention");
        graph.Outputs().push_back(&extra);
      }
      // The join's consumer follows the cache update, as it does between
      // transformer blocks. It must consume the plugin's forwarded join, not
      // cause TensorRT to recompute attention against already modified caches.
      auto& attention = add(join, bias, "attention");
      graph.Outputs().push_back(&attention);

      setenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES", "cache_update", 1);
      setenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE", "1", 1);
      setenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT",
             transposed ? "" : "native", 1);
      auto built = litert::nvidia::BuildTensorRtEngine(
          litert::compiler::Subgraph(LrtGetCompilerContext(), &graph));
      unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES");
      unsetenv("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE");
      unsetenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT");
      ASSERT_TRUE(built.HasValue()) << built.Error().Message();
      ASSERT_EQ(built->input_names.size(), 9);
      ASSERT_EQ(built->output_names.size(), use == JoinUse::kClosed ? 3 : 4);
      EXPECT_EQ(litert::nvidia::IsTransposedValueCacheTensor(
                    built->input_names[4]),
                transposed);
      litert::nvidia::TensorRtLogger logger;
      std::unique_ptr<nvinfer1::IRuntime> runtime(
          nvinfer1::createInferRuntime(logger));
      ASSERT_NE(runtime, nullptr);
      std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
          built->engine.data(), built->engine.size()));
      ASSERT_NE(engine, nullptr);
      std::unique_ptr<nvinfer1::IEngineInspector> inspector(
          engine->createEngineInspector());
      ASSERT_NE(inspector, nullptr);
      const std::string layers = inspector->getEngineInformation(
          nvinfer1::LayerInformationFormat::kONELINE);
      // Default inspector verbosity preserves layer names, not plugin types.
      EXPECT_EQ(layers.find("cache_update_forward_attention") !=
                    std::string::npos,
                forward_join)
          << layers;
      EXPECT_EQ(layers.find("cache_update_read_barriers") != std::string::npos,
                !forward_join)
          << layers;
      std::unique_ptr<nvinfer1::IExecutionContext> context(
          engine->createExecutionContext());
      ASSERT_NE(context, nullptr);
      const std::array<size_t, 9> input_sizes{
          kUpdateElements * 2, kUpdateElements * 2, 7 * sizeof(int32_t),
          kCacheElements * 2, kCacheElements * 2, 4 * sizeof(int32_t),
          4 * sizeof(int32_t), kUpdateElements * 2,
          kHeads * kRows * kColumns * sizeof(uint8_t)};
      std::array<void*, 9> inputs{};
      for (size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_EQ(cudaMalloc(&inputs[i], input_sizes[i]), cudaSuccess);
        ASSERT_TRUE(context->setTensorAddress(built->input_names[i].c_str(),
                                             inputs[i]));
      }
      std::vector<size_t> output_elements{kCacheElements, kCacheElements};
      if (use != JoinUse::kClosed) {
        size_t extra_elements = kUpdateElements;
        if (escape_scores) {
          extra_elements = kScoreElements;
        } else if (use == JoinUse::kEscapingIntermediate) {
          extra_elements = kHeads * kRows * kColumns;
        }
        output_elements.push_back(extra_elements);
      }
      output_elements.push_back(kUpdateElements);
      std::vector<void*> outputs(output_elements.size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        const char* alias =
            engine->getAliasedInputTensor(built->output_names[i].c_str());
        if (i < 2) {
          ASSERT_NE(alias, nullptr);
          EXPECT_EQ(std::string(alias), built->input_names[i + 3]);
          outputs[i] = inputs[i + 3];
        } else {
          ASSERT_EQ(alias, nullptr);
          ASSERT_EQ(cudaMalloc(&outputs[i], output_elements[i] * 2),
                    cudaSuccess);
        }
        ASSERT_TRUE(context->setTensorAddress(built->output_names[i].c_str(),
                                             outputs[i]));
      }
      cudaStream_t stream = nullptr;
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
      const std::array<int32_t, 4> ignored_starts{111, -222, 333, -444};
      for (int i : {5, 6}) {
        ASSERT_EQ(cudaMemcpy(inputs[i], ignored_starts.data(), input_sizes[i],
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      const auto v_index = [&](int h, int s, int d) {
        return transposed ? (h * kSeq + s) * kDepth + d
                          : (h * kDepth + d) * kSeq + s;
      };
      std::vector<uint16_t> old_k(kCacheElements), old_v(kCacheElements);
      std::vector<uint16_t> queries(kUpdateElements, Fp16Bits(0.0f));
      for (int h = 0; h < kHeads; ++h) {
        for (int s = 0; s < kSeq; ++s) {
          for (int d = 0; d < kDepth; ++d) {
            old_k[(h * kSeq + s) * kDepth + d] =
                Fp16Bits((h + 1) * 0.125f + s * 0.0625f + d * 0.03125f);
            old_v[v_index(h, s, d)] =
                Fp16Bits(1.0f + h * 0.25f + s * 0.125f + d * 0.03125f);
          }
        }
        for (int r = 0; r < kRows; ++r) {
          queries[(h * kRows + r) * kDepth + r] = Fp16Bits(0.25f);
        }
      }
      for (const auto& [i, data] :
           {std::pair<int, const void*>{3, old_k.data()},
            {4, old_v.data()}, {7, queries.data()}}) {
        ASSERT_EQ(cudaMemcpy(inputs[i], data, input_sizes[i],
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
      }
      // Reuse the serialized engine and its aliased caches across first-chunk
      // padding, a wrap, a partial wrap, zero length, and invalid update params.
      for (const auto [write, valid] :
           {std::pair<int, int>{0, kRows - 1}, {kRows - 1, kRows},
            {kSeq - 1, 2}, {0, 0}, {-1, 2}, {kSeq - 1, kRows + 1}}) {
        SCOPED_TRACE(testing::Message() << "write=" << write
                                       << " valid=" << valid);
        const int attention_rows = valid >= 0 && valid <= kRows ? valid : 0;
        std::vector<uint16_t> fresh_k(kUpdateElements, Fp16Bits(64.0f));
        std::vector<uint16_t> fresh_v(kUpdateElements, Fp16Bits(64.0f));
        auto expected_k = old_k;
        auto expected_v = old_v;
        for (int h = 0; h < kHeads; ++h) {
          for (int r = 0; r < attention_rows; ++r) {
            for (int d = 0; d < kDepth; ++d) {
              const int index = (h * kRows + r) * kDepth + d;
              fresh_k[index] =
                  Fp16Bits(1.0f + h * 0.125f + r * 0.0625f + d * 0.03125f);
              fresh_v[index] =
                  Fp16Bits(4.0f + h * 0.25f + r * 0.125f + d * 0.03125f);
              if (write >= 0) {
                const int s = (write + r) % kSeq;
                expected_k[(h * kSeq + s) * kDepth + d] = fresh_k[index];
                expected_v[v_index(h, s, d)] = fresh_v[index];
              }
            }
          }
        }
        std::vector<uint8_t> mask_host(kHeads * kRows * kColumns);
        for (size_t i = 0; i < mask_host.size(); ++i) {
          mask_host[i] = i % kColumns < kSeq + attention_rows;
        }
        const std::array<int32_t, 7> params_host{write, 0, 0, valid, 0, 0, 0};
        for (const auto& [i, data] :
             {std::pair<int, const void*>{0, fresh_k.data()},
              {1, fresh_v.data()}, {2, params_host.data()},
              {8, mask_host.data()}}) {
          ASSERT_EQ(cudaMemcpy(inputs[i], data, input_sizes[i],
                               cudaMemcpyHostToDevice),
                    cudaSuccess);
        }
        ASSERT_TRUE(context->enqueueV3(stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::vector<std::vector<uint16_t>> actual(outputs.size());
        for (size_t i = 0; i < actual.size(); ++i) {
          actual[i].resize(output_elements[i]);
          ASSERT_EQ(cudaMemcpy(actual[i].data(), outputs[i],
                               output_elements[i] * 2, cudaMemcpyDeviceToHost),
                    cudaSuccess);
        }
        EXPECT_EQ(actual[0], expected_k);
        EXPECT_EQ(actual[1], expected_v);
        for (int h = 0; h < kHeads; ++h) {
          for (int r = 0; r < kRows; ++r) {
            std::array<float, kColumns> probabilities{};
            float denominator = 0.0f;
            for (int s = 0; s < kSeq + attention_rows; ++s) {
              const float score = 0.25f * Fp16ToFloat(
                  s < kSeq ? old_k[(h * kSeq + s) * kDepth + r]
                           : fresh_k[(h * kRows + s - kSeq) * kDepth + r]);
              probabilities[s] = std::exp(score);
              denominator += probabilities[s];
              if (escape_scores && s < kSeq) {
                EXPECT_EQ(actual[2][(h * kRows + r) * kSeq + s],
                          Fp16Bits(score));
              }
            }
            if (use == JoinUse::kEscapingIntermediate) {
              for (int s = 0; s < kColumns; ++s) {
                const float probability = probabilities[s] / denominator;
                EXPECT_NEAR(Fp16ToFloat(
                                actual[2][(h * kRows + r) * kColumns + s]),
                            probability * probability, 0.0002f)
                    << "h=" << h << " row=" << r << " column=" << s;
              }
            }
            for (int d = 0; d < kDepth; ++d) {
              float reference = 1.0f;  // Post-update attention consumer's bias.
              for (int s = 0; s < kSeq + attention_rows; ++s) {
                const float value = Fp16ToFloat(
                    s < kSeq ? old_v[v_index(h, s, d)]
                             : fresh_v[(h * kRows + s - kSeq) * kDepth + d]);
                reference += probabilities[s] / denominator * value;
              }
              EXPECT_NEAR(Fp16ToFloat(
                              actual.back()[(h * kRows + r) * kDepth + d]),
                          reference, 0.02f)
                  << "h=" << h << " row=" << r << " d=" << d;
              if (use == JoinUse::kEarlyConsumer ||
                  use == JoinUse::kMultipleConsumers ||
                  use == JoinUse::kGraphOutput) {
                float extra_reference = reference - 1.0f;
                if (use == JoinUse::kEarlyConsumer) {
                  extra_reference = reference;
                } else if (use == JoinUse::kMultipleConsumers) {
                  extra_reference *= 2.0f;
                }
                EXPECT_NEAR(Fp16ToFloat(
                                actual[2][(h * kRows + r) * kDepth + d]),
                            extra_reference, 0.04f)
                    << "h=" << h << " row=" << r << " d=" << d;
              }
            }
          }
        }
        old_k = std::move(expected_k);
        old_v = std::move(expected_v);
      }
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      for (size_t i = 2; i < outputs.size(); ++i) {
        EXPECT_EQ(cudaFree(outputs[i]), cudaSuccess);
      }
      for (void* input : inputs) EXPECT_EQ(cudaFree(input), cudaSuccess);
    }
  }
}

TEST(TensorRtGraphBuilderTest, TransposedCacheMatmul) {
  // Gemma 4 multiplies the query against the updated key cache ([B, H, S, D])
  // and the probabilities against the updated value cache (stored transposed,
  // [B, H, D, S]) through runtime_bmm or a transposed batch matmul. The
  // builder holds value caches as [B, H, S, D] in the engine so that product
  // needs no transpose; results must match the model layout for logically
  // equal caches, keys keep their layout, and a value reader that is not a
  // transposed matmul keeps the model layout.
  constexpr int kHeads = 2;
  constexpr int kDepth = 8;
  constexpr int kSeq = 16;
  constexpr int kPosition = 5;
  struct Case {
    bool key;
    bool composite;
    bool transposed_reader;
    bool transposed_layout;
    std::vector<int32_t> engine_dims;
    bool read_only = false;
    bool declare_read_only = false;
    int rows = 1;
  };
  const std::vector<Case> cases = {
      {false, true, true, true, {1, kHeads, kSeq, kDepth}},
      {false, true, true, false, {1, kHeads * kDepth, kSeq, 1}},
      {false, false, true, true, {1, kHeads, kSeq, kDepth}},
      {false, false, true, false, {1, kHeads * kDepth, kSeq, 1}},
      {false, false, false, true, {1, kHeads * kDepth, kSeq, 1}},
      {true, true, true, true, {1, kHeads, kSeq, kDepth}},
      {true, false, true, true, {1, kHeads, kSeq, kDepth}},
      // MTP reads another engine's populated V cache without updating it.
      {false, true, true, true, {1, kHeads, kSeq, kDepth}, true, true},
      {false, false, true, true, {1, kHeads, kSeq, kDepth}, true, true},
      // No explicit contract must preserve native input layout, even when
      // transposition is enabled for in-place cache producers by default.
      {false, true, true, true, {1, kHeads, kDepth, kSeq}, true, false},
      {false, false, true, false, {1, kHeads, kDepth, kSeq}, true, false},
      // The verifier writes and evaluates four tokens in one invocation.
      {true, true, true, true, {1, kHeads, kSeq, kDepth}, false, false, 4},
      {false, true, true, true, {1, kHeads, kSeq, kDepth}, false, false, 4},
      {false, true, true, false, {1, kHeads * kDepth, kSeq, 1}, false, false, 4},
      {false, false, true, true, {1, kHeads, kSeq, kDepth}, false, false, 4},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "key=" << test_case.key
                 << " composite=" << test_case.composite
                 << " transposed_reader=" << test_case.transposed_reader
                 << " transposed_layout=" << test_case.transposed_layout
                 << " read_only=" << test_case.read_only
                 << " declare_read_only=" << test_case.declare_read_only
                 << " rows=" << test_case.rows);
    setenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT",
           test_case.transposed_layout ? "" : "native", 1);
    // Model layouts: keys [B,H,S,D] with [B,H,M,D] updates, values
    // [B,H,D,S] with [B,H,D,M] updates.
    const std::vector<int32_t> cache_dims =
        test_case.key ? std::vector<int32_t>{1, kHeads, kSeq, kDepth}
                      : std::vector<int32_t>{1, kHeads, kDepth, kSeq};
    const std::vector<int32_t> update_dims =
        test_case.key ? std::vector<int32_t>{1, kHeads, test_case.rows, kDepth}
                      : std::vector<int32_t>{1, kHeads, kDepth, test_case.rows};
    // The transposed product contracts the cache's last model axis: q @ K^T
    // takes a [B, H, M, D] query and yields [B, H, M, S]; probs @ V^T takes
    // [B, H, M, S] probabilities and yields [B, H, M, D]. A plain product
    // contracts the other axis instead.
    const int contracted = test_case.transposed_reader ? cache_dims[3]
                                                       : cache_dims[2];
    const int produced = test_case.transposed_reader ? cache_dims[2]
                                                     : cache_dims[3];
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& cache = graph.EmplaceTensor();
    cache.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, cache_dims));
    cache.SetName("cache");
    graph.Inputs().push_back(&cache);
    auto& update = graph.EmplaceTensor();
    update.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, update_dims));
    update.SetName("update");
    if (!test_case.read_only) {
      graph.Inputs().push_back(&update);
    }
    auto& start = graph.EmplaceTensor();
    start.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
    start.SetName("start");
    if (!test_case.read_only) {
      graph.Inputs().push_back(&start);
    }
    auto& lhs = graph.EmplaceTensor();
    lhs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                     {1, kHeads, test_case.rows, contracted}));
    lhs.SetName("lhs");
    graph.Inputs().push_back(&lhs);
    auto& positions = graph.EmplaceTensor();
    positions.SetType(
        MakeRankedTensorType(kLiteRtElementTypeInt32, {test_case.rows}));
    positions.SetName("positions");
    if (test_case.composite) {
      graph.Inputs().push_back(&positions);
    }
    auto& cache_out = graph.EmplaceTensor();
    cache_out.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, cache_dims));
    cache_out.SetName("cache_out");
    if (!test_case.read_only) {
      graph.Outputs().push_back(&cache_out);
    }
    auto& result = graph.EmplaceTensor();
    result.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, kHeads, test_case.rows, produced}));
    result.SetName("result");
    graph.Outputs().push_back(&result);

    if (!test_case.read_only) {
      auto& dus = graph.EmplaceOp();
      dus.SetOpCode(kLiteRtOpCodeTflDynamicUpdateSlice);
      litert::internal::AttachInput(&cache, dus);
      litert::internal::AttachInput(&update, dus);
      litert::internal::AttachInput(&start, dus);
      litert::internal::AttachOutput(&cache_out, dus);
    }
    auto& matmul = graph.EmplaceOp();
    if (test_case.composite) {
      matmul.SetOpCode(kLiteRtOpCodeShloComposite);
      tflite::StableHLOCompositeOptionsT composite;
      composite.name = "odml.runtime_bmm";
      tflite::BuiltinOptions2Union options;
      options.Set(std::move(composite));
      litert::internal::SetTflOptions2(matmul, std::move(options));
    } else {
      matmul.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
      tflite::BatchMatMulOptionsT batch_matmul;
      batch_matmul.adj_y = test_case.transposed_reader;
      tflite::BuiltinOptionsUnion options;
      options.Set(std::move(batch_matmul));
      litert::internal::SetTflOptions(matmul, std::move(options));
    }
    litert::internal::AttachInput(&lhs, matmul);
    litert::internal::AttachInput(
        test_case.read_only ? &cache : &cache_out, matmul);
    if (test_case.composite) {
      litert::internal::AttachInput(&positions, matmul);
    }
    litert::internal::AttachOutput(&result, matmul);

    const std::vector<std::string> read_only_inputs =
        test_case.declare_read_only ? std::vector<std::string>{"cache"}
                                    : std::vector<std::string>{};
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(LrtGetCompilerContext(), &graph),
        read_only_inputs);
    unsetenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT");
    ASSERT_TRUE(built.HasValue()) << built.Error().Message();
    litert::nvidia::TensorRtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
        built->engine.data(), built->engine.size()));
    ASSERT_NE(engine, nullptr);
    const auto engine_dims =
        engine->getTensorShape(built->input_names[0].c_str());
    ASSERT_EQ(engine_dims.nbDims, 4);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(engine_dims.d[i], test_case.engine_dims[i]) << "dim " << i;
    }
    if (!test_case.transposed_reader) {
      continue;  // Layout coverage only; the product is a plain matmul.
    }
    // The engine buffer is [B, H, S, D] (sequence-major) for keys and for
    // values held transposed, [B, H, D, S] for values in the model layout.
    const bool seq_major =
        test_case.key ||
        (test_case.transposed_layout &&
         (!test_case.read_only || test_case.declare_read_only));
    EXPECT_EQ(litert::nvidia::IsTransposedValueCacheTensor(
                  built->input_names[0]),
              seq_major && !test_case.key);
    if (!test_case.read_only) {
      EXPECT_EQ(litert::nvidia::IsTransposedValueCacheTensor(
                    built->output_names[0]),
                seq_major && !test_case.key);
    }
    const auto cache_index = [&](int h, int seq, int d) -> size_t {
      return seq_major ? (h * kSeq + seq) * kDepth + d
                       : (h * kDepth + d) * kSeq + seq;
    };
    // Logical values are small multiples of 0.25 so FP16 holds them exactly.
    const auto logical_value = [&](int h, int seq, int d) -> float {
      return static_cast<float>((h * 3 + seq * 5 + d * 7) % 11 - 5) * 0.25f;
    };
    const auto update_value = [&](int h, int row, int d) -> float {
      return static_cast<float>(d - 3 + h + row * 2) * 0.5f;
    };
    const auto lhs_value = [&](int h, int row, int i) -> float {
      return static_cast<float>((h + row * 2 + i) % 5) * 0.125f;
    };
    const size_t cache_elements = kHeads * kDepth * kSeq;
    std::vector<uint16_t> cache_host(cache_elements);
    for (int h = 0; h < kHeads; ++h) {
      for (int seq = 0; seq < kSeq; ++seq) {
        for (int d = 0; d < kDepth; ++d) {
          cache_host[cache_index(h, seq, d)] =
              Fp16Bits(logical_value(h, seq, d));
        }
      }
    }
    std::vector<uint16_t> update_host(kHeads * test_case.rows * kDepth);
    for (int h = 0; h < kHeads; ++h) {
      for (int row = 0; row < test_case.rows; ++row) {
        for (int d = 0; d < kDepth; ++d) {
          const size_t index = test_case.key
                                   ? (h * test_case.rows + row) * kDepth + d
                                   : (h * kDepth + d) * test_case.rows + row;
          update_host[index] = Fp16Bits(update_value(h, row, d));
        }
      }
    }
    std::vector<uint16_t> lhs_host(kHeads * test_case.rows * contracted);
    for (int h = 0; h < kHeads; ++h) {
      for (int row = 0; row < test_case.rows; ++row) {
        for (int i = 0; i < contracted; ++i) {
          lhs_host[(h * test_case.rows + row) * contracted + i] =
              Fp16Bits(lhs_value(h, row, i));
        }
      }
    }
    std::array<int32_t, 4> start_host = {0, 0, 0, 0};
    start_host[test_case.key ? 2 : 3] = kPosition;
    std::vector<int32_t> positions_host(test_case.rows);
    for (int row = 0; row < test_case.rows; ++row) {
      positions_host[row] = kPosition + row;
    }
    const size_t positions_bytes = positions_host.size() * sizeof(int32_t);
    void* device_cache = nullptr;
    void* device_update = nullptr;
    void* device_start = nullptr;
    void* device_lhs = nullptr;
    void* device_positions = nullptr;
    void* device_result = nullptr;
    ASSERT_EQ(cudaMalloc(&device_cache, cache_elements * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_update, update_host.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_start, sizeof(start_host)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_lhs, lhs_host.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_positions, positions_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_result,
                         kHeads * test_case.rows * produced * 2),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_cache, cache_host.data(), cache_elements * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_update, update_host.data(),
                         update_host.size() * 2, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_start, start_host.data(), sizeof(start_host),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_lhs, lhs_host.data(), lhs_host.size() * 2,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_positions, positions_host.data(),
                         positions_bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                         device_cache));
    if (!test_case.read_only) {
      ASSERT_TRUE(context->setTensorAddress(built->input_names[1].c_str(),
                                           device_update));
      ASSERT_TRUE(context->setTensorAddress(built->input_names[2].c_str(),
                                           device_start));
    }
    ASSERT_TRUE(context->setTensorAddress(
        built->input_names[test_case.read_only ? 1 : 3].c_str(), device_lhs));
    if (test_case.composite) {
      ASSERT_EQ(built->input_names.size(), test_case.read_only ? 3u : 5u);
      ASSERT_TRUE(context->setTensorAddress(
          built->input_names[test_case.read_only ? 2 : 4].c_str(),
          device_positions));
    }
    if (!test_case.read_only) {
      ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                           device_cache));  // in place
    }
    ASSERT_TRUE(context->setTensorAddress(
        built->output_names[test_case.read_only ? 0 : 1].c_str(), device_result));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<uint16_t> cache_actual(cache_elements);
    std::vector<uint16_t> result_actual(kHeads * test_case.rows * produced);
    ASSERT_EQ(cudaMemcpy(cache_actual.data(), device_cache, cache_elements * 2,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(result_actual.data(), device_result,
                         result_actual.size() * 2, cudaMemcpyDeviceToHost),
              cudaSuccess);
    // The updated cache holds the update at the position; the product
    // contracts depth for keys (result per sequence position) and sequence
    // for values (result per depth).
    const auto updated = [&](int h, int seq, int d) -> float {
      return !test_case.read_only && seq >= kPosition &&
                     seq < kPosition + test_case.rows
                 ? update_value(h, seq - kPosition, d)
                 : logical_value(h, seq, d);
    };
    for (int h = 0; h < kHeads; ++h) {
      for (int seq = 0; seq < kSeq; ++seq) {
        for (int d = 0; d < kDepth; ++d) {
          ASSERT_EQ(cache_actual[cache_index(h, seq, d)],
                    Fp16Bits(updated(h, seq, d)))
              << "h=" << h << " seq=" << seq << " d=" << d;
        }
      }
      for (int row = 0; row < test_case.rows; ++row) {
        for (int i = 0; i < produced; ++i) {
          float expected = 0.0f;
          for (int j = 0; j < contracted; ++j) {
            expected += lhs_value(h, row, j) *
                        (test_case.key ? updated(h, i, j) : updated(h, j, i));
          }
          EXPECT_NEAR(
              Fp16ToFloat(result_actual[(h * test_case.rows + row) * produced +
                                       i]),
              expected, 0.02f)
              << "h=" << h << " row=" << row << " i=" << i;
        }
      }
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(device_result), cudaSuccess);
    EXPECT_EQ(cudaFree(device_positions), cudaSuccess);
    EXPECT_EQ(cudaFree(device_lhs), cudaSuccess);
    EXPECT_EQ(cudaFree(device_start), cudaSuccess);
    EXPECT_EQ(cudaFree(device_update), cudaSuccess);
    EXPECT_EQ(cudaFree(device_cache), cudaSuccess);
  }
}

TEST(TensorRtGraphBuilderTest, RejectsInvalidReadOnlyValueCacheContract) {
  struct Case {
    std::vector<int32_t> dims;
    std::vector<std::string> names = {"cache"};
    bool adj_y = true;
    bool extra_reader = false;
    LiteRtElementType element_type = kLiteRtElementTypeFloat16;
    const char* conflicting_environment = nullptr;
    const char* conflicting_value = nullptr;
  };
  const std::vector<Case> cases = {
      {{1, 2, 8, 16}, {"missing"}},
      {{1, 2, 8, 16}, {"cache", "cache"}},
      {{1, 2, 8, 16}, {""}},
      {{2, 8, 16}},
      {{2, 2, 8, 16}},
      {{1, 2, 8, -1}},
      {{1, 2, 8, 0}},
      {{1, 65536, 65536, 16}},
      {{1, 2, 8, 16}, {"cache"}, false},
      {{1, 2, 8, 16}, {"cache"}, true, true},
      {{1, 2, 8, 16}, {"cache"}, true, false, kLiteRtElementTypeInt8},
      {{1, 2, 8, 16}, {"cache"}, true, false, kLiteRtElementTypeFloat16,
       "LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT", "native"},
      {{1, 2, 8, 16}, {"cache"}, true, false, kLiteRtElementTypeFloat16,
       "LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE", "0"},
      {{1, 2, 8, 16}, {"cache"}, true, false, kLiteRtElementTypeFloat16,
       "LITERT_NVIDIA_TENSORRT_RUNTIME_BMM_CONTEXT_LIMIT", "16"},
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& test_case = cases[i];
    LiteRtModelT model;
    auto& graph = model.EmplaceSubgraph();
    auto& cache = graph.EmplaceTensor();
    cache.SetName("cache");
    cache.SetType(MakeRankedTensorType(test_case.element_type, test_case.dims));
    graph.Inputs().push_back(&cache);
    auto& lhs = graph.EmplaceTensor();
    lhs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 1, 16}));
    graph.Inputs().push_back(&lhs);
    auto& output = graph.EmplaceTensor();
    output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 1, 8}));
    graph.Outputs().push_back(&output);
    auto& matmul = graph.EmplaceOp();
    matmul.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
    tflite::BatchMatMulOptionsT matmul_options;
    matmul_options.adj_y = test_case.adj_y;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(matmul_options));
    litert::internal::SetTflOptions(matmul, std::move(options));
    litert::internal::AttachInput(&lhs, matmul);
    litert::internal::AttachInput(&cache, matmul);
    litert::internal::AttachOutput(&output, matmul);
    if (test_case.extra_reader) {
      auto& other_reader = graph.EmplaceOp();
      other_reader.SetOpCode(kLiteRtOpCodeTflTranspose);
      litert::internal::AttachInput(&cache, other_reader);
    }
    const char* previous = test_case.conflicting_environment == nullptr
                               ? nullptr
                               : std::getenv(test_case.conflicting_environment);
    const bool was_set = previous != nullptr;
    const std::string old_value = was_set ? previous : "";
    if (test_case.conflicting_environment != nullptr) {
      setenv(test_case.conflicting_environment, test_case.conflicting_value, 1);
    }
    auto built = litert::nvidia::BuildTensorRtEngine(
        litert::compiler::Subgraph(LrtGetCompilerContext(), &graph),
        test_case.names);
    if (test_case.conflicting_environment != nullptr) {
      if (was_set) {
        setenv(test_case.conflicting_environment, old_value.c_str(), 1);
      } else {
        unsetenv(test_case.conflicting_environment);
      }
    }
    ASSERT_FALSE(built.HasValue());
    EXPECT_EQ(built.Error().Status(), kLiteRtStatusErrorInvalidArgument);
  }
}

TEST(TensorRtGraphBuilderTest, DecodeAttentionPluginBlock) {
  // The decode attention block (in-place K/V updates, runtime_bmm scores,
  // select_v2 mask, softmax, runtime_bmm values) lowers to one CUDA plugin
  // launch when the experimental plugin is enabled; its result matches an
  // FP32 host reference and the default TensorRT lowering.
  struct Case {
    int heads;
    int rows;
    int seq;
    int depth;
    int mask_rows;
    uint16_t fill_bits;
  };
  const std::vector<Case> cases = {
      {2, 2, 300, 128, 2, 0xFC00},   // -inf fill, two chunks
      {1, 16, 600, 512, 1, 0xF0E2},  // -10000 fill, shared mask row
  };
  for (const auto& test_case : cases) {
    const int heads = test_case.heads;
    const int rows = test_case.rows;
    const int seq = test_case.seq;
    const int depth = test_case.depth;
    const int position = seq / 2 + 3;
    SCOPED_TRACE(::testing::Message()
                 << "heads=" << heads << " rows=" << rows << " seq=" << seq
                 << " depth=" << depth);
    // Deterministic inputs in FP16.
    uint32_t state = 12345u + seq;
    const auto next = [&]() -> float {
      state = state * 1664525u + 1013904223u;
      return static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
    };
    const size_t cache_elements = static_cast<size_t>(heads) * seq * depth;
    std::vector<uint16_t> k_host(cache_elements), v_host(cache_elements);
    std::vector<float> k_ref(cache_elements), v_ref(cache_elements);
    for (size_t i = 0; i < cache_elements; ++i) {
      k_host[i] = Fp16Bits(next() * 0.5f);
      v_host[i] = Fp16Bits(next());
      k_ref[i] = Fp16ToFloat(k_host[i]);
      v_ref[i] = Fp16ToFloat(v_host[i]);
    }
    std::vector<uint16_t> k_update(heads * depth), v_update(heads * depth);
    for (int i = 0; i < heads * depth; ++i) {
      k_update[i] = Fp16Bits(next() * 0.5f);
      v_update[i] = Fp16Bits(next());
    }
    std::vector<uint16_t> q_host(heads * rows * depth);
    std::vector<float> q_ref(q_host.size());
    for (size_t i = 0; i < q_host.size(); ++i) {
      q_host[i] = Fp16Bits(next() * 0.25f);
      q_ref[i] = Fp16ToFloat(q_host[i]);
    }
    std::vector<uint8_t> mask_host(test_case.mask_rows * seq);
    for (int r = 0; r < test_case.mask_rows; ++r) {
      for (int j = 0; j < seq; ++j) {
        mask_host[r * seq + j] = (j <= position - r && j % 7 != 3) ? 1 : 0;
      }
    }
    const float fill = Fp16ToFloat(test_case.fill_bits);
    // Host reference: caches after the update, masked softmax, weighted sum.
    for (int h = 0; h < heads; ++h) {
      for (int d = 0; d < depth; ++d) {
        const size_t idx = (static_cast<size_t>(h) * seq + position) * depth + d;
        k_ref[idx] = Fp16ToFloat(k_update[h * depth + d]);
        v_ref[idx] = Fp16ToFloat(v_update[h * depth + d]);
      }
    }
    std::vector<float> expected(heads * rows * depth, 0.0f);
    std::vector<float> scores(seq);
    for (int h = 0; h < heads; ++h) {
      for (int r = 0; r < rows; ++r) {
        float max_score = -INFINITY;
        for (int j = 0; j < seq; ++j) {
          float score = 0.0f;
          for (int d = 0; d < depth; ++d) {
            score += q_ref[(static_cast<size_t>(h) * rows + r) * depth + d] *
                     k_ref[(static_cast<size_t>(h) * seq + j) * depth + d];
          }
          const int mask_row = test_case.mask_rows == 1 ? 0 : r;
          scores[j] = mask_host[mask_row * seq + j] ? score : fill;
          max_score = std::max(max_score, scores[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < seq; ++j) {
          scores[j] = std::exp(scores[j] - max_score);
          sum += scores[j];
        }
        for (int j = 0; j < seq; ++j) {
          const float p = scores[j] / sum;
          for (int d = 0; d < depth; ++d) {
            expected[(static_cast<size_t>(h) * rows + r) * depth + d] +=
                p * v_ref[(static_cast<size_t>(h) * seq + j) * depth + d];
          }
        }
      }
    }

    for (const bool plugin : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "plugin=" << plugin);
      setenv("LITERT_NVIDIA_TENSORRT_DECODE_ATTENTION_PLUGIN",
             plugin ? "1" : "0", 1);
      LiteRtModelT model;
      auto& graph = model.EmplaceSubgraph();
      auto& k_cache = graph.EmplaceTensor();
      k_cache.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                          {1, heads, seq, depth}));
      k_cache.SetName("k_cache");
      graph.Inputs().push_back(&k_cache);
      auto& v_cache = graph.EmplaceTensor();
      v_cache.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                          {1, heads, depth, seq}));
      v_cache.SetName("v_cache");
      graph.Inputs().push_back(&v_cache);
      auto& k_upd = graph.EmplaceTensor();
      k_upd.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, heads, 1, depth}));
      k_upd.SetName("k_update");
      graph.Inputs().push_back(&k_upd);
      auto& v_upd = graph.EmplaceTensor();
      v_upd.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, heads, depth, 1}));
      v_upd.SetName("v_update");
      graph.Inputs().push_back(&v_upd);
      auto& k_start = graph.EmplaceTensor();
      k_start.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
      k_start.SetName("k_start");
      graph.Inputs().push_back(&k_start);
      auto& v_start = graph.EmplaceTensor();
      v_start.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {4}));
      v_start.SetName("v_start");
      graph.Inputs().push_back(&v_start);
      auto& q = graph.EmplaceTensor();
      q.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                    {1, heads, rows, depth}));
      q.SetName("q");
      graph.Inputs().push_back(&q);
      auto& mask = graph.EmplaceTensor();
      mask.SetType(MakeRankedTensorType(kLiteRtElementTypeBool,
                                       {1, 1, test_case.mask_rows, seq}));
      mask.SetName("mask");
      graph.Inputs().push_back(&mask);
      auto& positions = graph.EmplaceTensor();
      positions.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
      positions.SetName("positions");
      graph.Inputs().push_back(&positions);
      auto& fill_const = graph.EmplaceTensor();
      fill_const.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {}));
      fill_const.SetName("fill");
      const uint16_t fill_bits = test_case.fill_bits;
      SetWeightsFromUnownedBuffer(
          fill_const.Weights(),
          litert::BufferRef<uint8_t>(
              reinterpret_cast<const uint8_t*>(&fill_bits), sizeof(fill_bits)));
      auto& k_out = graph.EmplaceTensor();
      k_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, heads, seq, depth}));
      k_out.SetName("k_out");
      graph.Outputs().push_back(&k_out);
      auto& v_out = graph.EmplaceTensor();
      v_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, heads, depth, seq}));
      v_out.SetName("v_out");
      graph.Outputs().push_back(&v_out);
      auto& scores_t = graph.EmplaceTensor();
      scores_t.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                           {1, heads, rows, seq}));
      auto& masked = graph.EmplaceTensor();
      masked.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                         {1, heads, rows, seq}));
      auto& probs = graph.EmplaceTensor();
      probs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, heads, rows, seq}));
      auto& context_out = graph.EmplaceTensor();
      context_out.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                              {1, heads, rows, depth}));
      context_out.SetName("context");
      graph.Outputs().push_back(&context_out);

      auto& k_dus = graph.EmplaceOp();
      k_dus.SetOpCode(kLiteRtOpCodeTflDynamicUpdateSlice);
      litert::internal::AttachInput(&k_cache, k_dus);
      litert::internal::AttachInput(&k_upd, k_dus);
      litert::internal::AttachInput(&k_start, k_dus);
      litert::internal::AttachOutput(&k_out, k_dus);
      auto& v_dus = graph.EmplaceOp();
      v_dus.SetOpCode(kLiteRtOpCodeTflDynamicUpdateSlice);
      litert::internal::AttachInput(&v_cache, v_dus);
      litert::internal::AttachInput(&v_upd, v_dus);
      litert::internal::AttachInput(&v_start, v_dus);
      litert::internal::AttachOutput(&v_out, v_dus);
      const auto add_runtime_bmm = [&](LiteRtTensorT& lhs, LiteRtTensorT& rhs,
                                       LiteRtTensorT& out) {
        auto& op = graph.EmplaceOp();
        op.SetOpCode(kLiteRtOpCodeShloComposite);
        tflite::StableHLOCompositeOptionsT composite;
        composite.name = "odml.runtime_bmm";
        tflite::BuiltinOptions2Union options;
        options.Set(std::move(composite));
        litert::internal::SetTflOptions2(op, std::move(options));
        litert::internal::AttachInput(&lhs, op);
        litert::internal::AttachInput(&rhs, op);
        litert::internal::AttachInput(&positions, op);
        litert::internal::AttachOutput(&out, op);
      };
      add_runtime_bmm(q, k_out, scores_t);
      auto& select = graph.EmplaceOp();
      select.SetOpCode(kLiteRtOpCodeTflSelectV2);
      litert::internal::AttachInput(&mask, select);
      litert::internal::AttachInput(&scores_t, select);
      litert::internal::AttachInput(&fill_const, select);
      litert::internal::AttachOutput(&masked, select);
      auto& softmax = graph.EmplaceOp();
      softmax.SetOpCode(kLiteRtOpCodeTflSoftmax);
      {
        tflite::SoftmaxOptionsT softmax_options;
        softmax_options.beta = 1.0f;
        tflite::BuiltinOptionsUnion options;
        options.Set(std::move(softmax_options));
        litert::internal::SetTflOptions(softmax, std::move(options));
      }
      litert::internal::AttachInput(&masked, softmax);
      litert::internal::AttachOutput(&probs, softmax);
      add_runtime_bmm(probs, v_out, context_out);

      auto built = litert::nvidia::BuildTensorRtEngine(
          litert::compiler::Subgraph(LrtGetCompilerContext(), &graph));
      unsetenv("LITERT_NVIDIA_TENSORRT_DECODE_ATTENTION_PLUGIN");
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
      EXPECT_EQ(layers.find("PluginV3") != std::string::npos, plugin);
      EXPECT_EQ(layers.find("Matrix Multiply") == std::string::npos, plugin)
          << layers;
      ASSERT_EQ(built->input_names.size(), 9u);
      ASSERT_EQ(built->output_names.size(), 3u);
      std::unique_ptr<nvinfer1::IExecutionContext> context(
          engine->createExecutionContext());
      ASSERT_NE(context, nullptr);

      // The value cache buffer holds [B, H, S, D] (the engine layout); the
      // model's [B, H, D, S] update is a [B, H, 1, D] row in memory.
      void* d_k = nullptr;
      void* d_v = nullptr;
      void* d_ku = nullptr;
      void* d_vu = nullptr;
      void* d_ks = nullptr;
      void* d_vs = nullptr;
      void* d_q = nullptr;
      void* d_mask = nullptr;
      void* d_pos = nullptr;
      void* d_out = nullptr;
      const std::array<int32_t, 4> k_start_host = {0, 0, position, 0};
      const std::array<int32_t, 4> v_start_host = {0, 0, 0, position};
      const int32_t position_host = position;
      ASSERT_EQ(cudaMalloc(&d_k, cache_elements * 2), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_v, cache_elements * 2), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_ku, k_update.size() * 2), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_vu, v_update.size() * 2), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_ks, sizeof(k_start_host)), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_vs, sizeof(v_start_host)), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_q, q_host.size() * 2), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_mask, mask_host.size()), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_pos, sizeof(position_host)), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&d_out, expected.size() * 2), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_k, k_host.data(), cache_elements * 2,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_v, v_host.data(), cache_elements * 2,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_ku, k_update.data(), k_update.size() * 2,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_vu, v_update.data(), v_update.size() * 2,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_ks, k_start_host.data(), sizeof(k_start_host),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_vs, v_start_host.data(), sizeof(v_start_host),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_q, q_host.data(), q_host.size() * 2,
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_mask, mask_host.data(), mask_host.size(),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(d_pos, &position_host, sizeof(position_host),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      const std::array<void*, 9> input_ptrs = {d_k,  d_v, d_ku,   d_vu, d_ks,
                                               d_vs, d_q, d_mask, d_pos};
      for (size_t i = 0; i < input_ptrs.size(); ++i) {
        ASSERT_TRUE(context->setTensorAddress(built->input_names[i].c_str(),
                                             input_ptrs[i]))
            << built->input_names[i];
      }
      ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                           d_k));  // in place
      ASSERT_TRUE(context->setTensorAddress(built->output_names[1].c_str(),
                                           d_v));  // in place
      ASSERT_TRUE(context->setTensorAddress(built->output_names[2].c_str(),
                                           d_out));
      cudaStream_t stream = nullptr;
      ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
      ASSERT_TRUE(context->enqueueV3(stream));
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      std::vector<uint16_t> actual(expected.size());
      ASSERT_EQ(cudaMemcpy(actual.data(), d_out, actual.size() * 2,
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      float max_error = 0.0f;
      for (size_t i = 0; i < expected.size(); ++i) {
        max_error = std::max(
            max_error, std::abs(Fp16ToFloat(actual[i]) - expected[i]));
      }
      EXPECT_LT(max_error, 0.03f);
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      for (void* ptr : {d_out, d_pos, d_mask, d_q, d_vs, d_ks, d_vu, d_ku,
                        d_v, d_k}) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
      }
    }
  }
}

TEST(TensorRtGraphBuilderTest, RuntimeBmmResultConcatenation) {
  // Gemma 4 prefill concatenates the runtime_bmm scores against the cache
  // with the scores against the current chunk (a plain FP16 batch matmul
  // result); the BF16 runtime_bmm result must be reconciled with it.
  constexpr int kHeads = 2;
  constexpr int kRows = 4;
  constexpr int kDepth = 8;
  constexpr int kSeq = 16;
  constexpr int kChunk = 4;
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& q = graph.EmplaceTensor();
  q.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                 {1, kHeads, kRows, kDepth}));
  q.SetName("q");
  graph.Inputs().push_back(&q);
  auto& cache = graph.EmplaceTensor();
  cache.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                     {1, kHeads, kSeq, kDepth}));
  cache.SetName("cache");
  graph.Inputs().push_back(&cache);
  auto& positions = graph.EmplaceTensor();
  positions.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1}));
  positions.SetName("positions");
  graph.Inputs().push_back(&positions);
  auto& chunk_scores = graph.EmplaceTensor();
  chunk_scores.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                            {1, kHeads, kRows, kChunk}));
  chunk_scores.SetName("chunk_scores");
  graph.Inputs().push_back(&chunk_scores);
  auto& cache_scores = graph.EmplaceTensor();
  cache_scores.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                            {1, kHeads, kRows, kSeq}));
  auto& output = graph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                      {1, kHeads, kRows, kSeq + kChunk}));
  output.SetName("scores");
  graph.Outputs().push_back(&output);
  auto& bmm = graph.EmplaceOp();
  bmm.SetOpCode(kLiteRtOpCodeShloComposite);
  {
    tflite::StableHLOCompositeOptionsT composite;
    composite.name = "odml.runtime_bmm";
    tflite::BuiltinOptions2Union options;
    options.Set(std::move(composite));
    litert::internal::SetTflOptions2(bmm, std::move(options));
  }
  litert::internal::AttachInput(&q, bmm);
  litert::internal::AttachInput(&cache, bmm);
  litert::internal::AttachInput(&positions, bmm);
  litert::internal::AttachOutput(&cache_scores, bmm);
  auto& concat = graph.EmplaceOp();
  concat.SetOpCode(kLiteRtOpCodeTflConcatenation);
  {
    tflite::ConcatenationOptionsT concatenation;
    concatenation.axis = 3;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(concatenation));
    litert::internal::SetTflOptions(concat, std::move(options));
  }
  litert::internal::AttachInput(&cache_scores, concat);
  litert::internal::AttachInput(&chunk_scores, concat);
  litert::internal::AttachOutput(&output, concat);

  auto built = litert::nvidia::BuildTensorRtEngine(
      litert::compiler::Subgraph(LrtGetCompilerContext(), &graph));
  ASSERT_TRUE(built.HasValue()) << built.Error().Message();
  litert::nvidia::TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
      built->engine.data(), built->engine.size()));
  ASSERT_NE(engine, nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  std::vector<uint16_t> q_host(kHeads * kRows * kDepth);
  std::vector<uint16_t> cache_host(kHeads * kSeq * kDepth);
  std::vector<uint16_t> chunk_host(kHeads * kRows * kChunk);
  for (size_t i = 0; i < q_host.size(); ++i) {
    q_host[i] = Fp16Bits(static_cast<float>((i % 5)) * 0.25f);
  }
  for (size_t i = 0; i < cache_host.size(); ++i) {
    cache_host[i] = Fp16Bits(static_cast<float>(static_cast<int>(i % 7) - 3) * 0.5f);
  }
  for (size_t i = 0; i < chunk_host.size(); ++i) {
    chunk_host[i] = Fp16Bits(static_cast<float>(i) * 0.125f);
  }
  const int32_t position = 0;
  void* d_q = nullptr;
  void* d_cache = nullptr;
  void* d_positions = nullptr;
  void* d_chunk = nullptr;
  void* d_out = nullptr;
  const size_t out_elements = kHeads * kRows * (kSeq + kChunk);
  ASSERT_EQ(cudaMalloc(&d_q, q_host.size() * 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_cache, cache_host.size() * 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_positions, sizeof(position)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_chunk, chunk_host.size() * 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, out_elements * 2), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_q, q_host.data(), q_host.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_cache, cache_host.data(), cache_host.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_positions, &position, sizeof(position),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_chunk, chunk_host.data(), chunk_host.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(built->input_names.size(), 4u);
  ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(), d_q));
  ASSERT_TRUE(
      context->setTensorAddress(built->input_names[1].c_str(), d_cache));
  ASSERT_TRUE(
      context->setTensorAddress(built->input_names[2].c_str(), d_positions));
  ASSERT_TRUE(
      context->setTensorAddress(built->input_names[3].c_str(), d_chunk));
  ASSERT_TRUE(
      context->setTensorAddress(built->output_names[0].c_str(), d_out));
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_TRUE(context->enqueueV3(stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::vector<uint16_t> actual(out_elements);
  ASSERT_EQ(cudaMemcpy(actual.data(), d_out, out_elements * 2,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int h = 0; h < kHeads; ++h) {
    for (int r = 0; r < kRows; ++r) {
      for (int j = 0; j < kSeq + kChunk; ++j) {
        float expected = 0.0f;
        if (j < kSeq) {
          for (int d = 0; d < kDepth; ++d) {
            expected += Fp16ToFloat(q_host[(h * kRows + r) * kDepth + d]) *
                        Fp16ToFloat(cache_host[(h * kSeq + j) * kDepth + d]);
          }
        } else {
          expected = Fp16ToFloat(chunk_host[(h * kRows + r) * kChunk + j - kSeq]);
        }
        const float value =
            Fp16ToFloat(actual[(h * kRows + r) * (kSeq + kChunk) + j]);
        EXPECT_NEAR(value, expected, 0.05f + 0.01f * std::abs(expected))
            << "h=" << h << " r=" << r << " j=" << j;
      }
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  for (void* ptr : {d_out, d_chunk, d_positions, d_cache, d_q}) {
    EXPECT_EQ(cudaFree(ptr), cudaSuccess);
  }
}

TEST(TensorRtGraphBuilderTest, MixedPrecisionBatchMatmul) {
  // In the BF16 activation mode an FP32 input becomes BF16 while FP16 inputs
  // stay FP16; a batch matmul over both must reconcile the operand types
  // (Gemma 4 prefill multiplies BF16 probabilities against FP16 chunk values).
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& lhs = graph.EmplaceTensor();
  lhs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 2, 4, 16}));
  lhs.SetName("lhs");
  graph.Inputs().push_back(&lhs);
  auto& rhs = graph.EmplaceTensor();
  rhs.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 16, 8}));
  rhs.SetName("rhs");
  graph.Inputs().push_back(&rhs);
  auto& output = graph.EmplaceTensor();
  output.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16, {1, 2, 4, 8}));
  output.SetName("output");
  graph.Outputs().push_back(&output);
  auto& op = graph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflBatchMatmul);
  {
    tflite::BatchMatMulOptionsT matmul;
    tflite::BuiltinOptionsUnion options;
    options.Set(std::move(matmul));
    litert::internal::SetTflOptions(op, std::move(options));
  }
  litert::internal::AttachInput(&lhs, op);
  litert::internal::AttachInput(&rhs, op);
  litert::internal::AttachOutput(&output, op);
  auto built = litert::nvidia::BuildTensorRtEngine(
      litert::compiler::Subgraph(LrtGetCompilerContext(), &graph));
  ASSERT_TRUE(built.HasValue()) << built.Error().Message();
  litert::nvidia::TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
      built->engine.data(), built->engine.size()));
  ASSERT_NE(engine, nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  std::vector<float> lhs_host(2 * 4 * 16);
  std::vector<uint16_t> rhs_host(2 * 16 * 8);
  for (size_t i = 0; i < lhs_host.size(); ++i) {
    lhs_host[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.25f;
  }
  for (size_t i = 0; i < rhs_host.size(); ++i) {
    rhs_host[i] = Fp16Bits(static_cast<float>(static_cast<int>(i % 7) - 3) * 0.5f);
  }
  void* d_lhs = nullptr;
  void* d_rhs = nullptr;
  void* d_out = nullptr;
  ASSERT_EQ(cudaMalloc(&d_lhs, lhs_host.size() * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_rhs, rhs_host.size() * 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_out, 2 * 4 * 8 * 2), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_lhs, lhs_host.data(), lhs_host.size() * 4,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_rhs, rhs_host.data(), rhs_host.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(), d_lhs));
  ASSERT_TRUE(context->setTensorAddress(built->input_names[1].c_str(), d_rhs));
  ASSERT_TRUE(
      context->setTensorAddress(built->output_names[0].c_str(), d_out));
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_TRUE(context->enqueueV3(stream));
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::vector<uint16_t> actual(2 * 4 * 8);
  ASSERT_EQ(cudaMemcpy(actual.data(), d_out, actual.size() * 2,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int b = 0; b < 2; ++b) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 8; ++c) {
        float expected = 0.0f;
        for (int k = 0; k < 16; ++k) {
          expected += lhs_host[(b * 4 + r) * 16 + k] *
                      Fp16ToFloat(rhs_host[(b * 16 + k) * 8 + c]);
        }
        EXPECT_NEAR(Fp16ToFloat(actual[(b * 4 + r) * 8 + c]), expected,
                    0.05f + 0.01f * std::abs(expected))
            << "b=" << b << " r=" << r << " c=" << c;
      }
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  EXPECT_EQ(cudaFree(d_out), cudaSuccess);
  EXPECT_EQ(cudaFree(d_rhs), cudaSuccess);
  EXPECT_EQ(cudaFree(d_lhs), cudaSuccess);
}

TEST(TensorRtGraphBuilderTest, LongContextSoftmaxSupport) {
  // Gemma 4 12B prefill_1024 at 32K depth. Only construct tensor metadata;
  // partition eligibility must not depend on allocating this large tensor.
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& input = graph.EmplaceTensor();
  auto& output = graph.EmplaceTensor();
  for (auto* tensor : {&input, &output}) {
    tensor->SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                        {1, 1, 16384, 34818}));
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
  query.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                    {1, 1, 16384, 512}));
  auto& key = graph.EmplaceTensor();
  key.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                  {1, 1, 34818, 512}));
  auto& scores = graph.EmplaceTensor();
  scores.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                     {1, 1, 16384, 34818}));
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
  scores.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32,
                                     {1, 1, 16384, 34818}));
  EXPECT_FALSE(litert::nvidia::IsTensorRtOpSupported(
      litert::compiler::Op(LrtGetCompilerContext(), &op)));
}

TEST(TensorRtGraphBuilderTest, LongContextRuntimeBatchMatmulSupport) {
  LiteRtModelT model;
  auto& graph = model.EmplaceSubgraph();
  auto& query = graph.EmplaceTensor();
  query.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                    {1, 1, 16384, 512}));
  auto& key = graph.EmplaceTensor();
  key.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                  {1, 1, 34818, 512}));
  auto& positions = graph.EmplaceTensor();
  positions.SetType(MakeRankedTensorType(kLiteRtElementTypeInt32, {1024}));
  auto& scores = graph.EmplaceTensor();
  scores.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                     {1, 1, 16384, 34818}));
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
  scores.SetType(MakeRankedTensorType(kLiteRtElementTypeFloat16,
                                     {1, 1, 16384, 34817}));
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
    cast_output.SetType(
        MakeRankedTensorType(kLiteRtElementTypeFloat16, shape));
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
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
        built->engine.data(), built->engine.size()));
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
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress(built->input_names[0].c_str(),
                                         device_input));
    ASSERT_TRUE(context->setTensorAddress(built->output_names[0].c_str(),
                                         device_output));
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), device_output, sizeof(actual),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    const std::array<uint16_t, 4> expected =
        scalar ? zeros : std::array<uint16_t, 4>{0, 0x3c00, 0xbc00, 0x7bff};
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
  }
}
