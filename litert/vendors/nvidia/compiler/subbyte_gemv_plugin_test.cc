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

#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"

namespace litert::nvidia {
namespace {

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      ADD_FAILURE() << "TensorRT-RTX: " << message;
    }
  }
};

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

TEST(SubbyteGemvPluginTest, SerializesAndMatchesReference) {
  // {bit width, output channels, input channels}. Include Gemma 4 12B's
  // INT4 feed-forward projections and its 480 MiB vocabulary projection.
  // The latter exercises large packed constants through engine serialization,
  // not just a direct kernel launch. The odd row count covers a partial CTA.
  constexpr std::array<std::array<int32_t, 3>, 6> kCases = {{
      {2, 8, 64},
      {4, 8, 64},
      {4, 9, 3840},
      {4, 15360, 3840},
      {4, 3840, 15360},
      {4, 262144, 3840},
  }};
  for (const auto& [bit_width, kRows, kColumns] : kCases) {
    SCOPED_TRACE(::testing::Message() << "bit_width=" << bit_width << " rows="
                                      << kRows << " columns=" << kColumns);
    const int values_per_byte = 8 / bit_width;
    const int value_count = 1 << bit_width;
    std::vector<uint16_t> activation(kColumns);
    std::vector<uint8_t> packed(kRows * kColumns / values_per_byte, 0);
    std::vector<uint16_t> scales(kRows);
    for (int column = 0; column < kColumns; ++column) {
      activation[column] =
          FloatToBf16Bits(static_cast<float>((column * 7) % 23 - 11) / 8.0f);
    }
    for (int row = 0; row < kRows; ++row) {
      scales[row] = FloatToBf16Bits(0.125f * static_cast<float>(row % 8 + 1));
      for (int column = 0; column < kColumns; ++column) {
        const int8_t value =
            static_cast<int8_t>((row + 3 * column) & (value_count - 1)) -
            value_count / 2;
        const size_t index = static_cast<size_t>(row) * kColumns + column;
        packed[index / values_per_byte] |=
            (static_cast<uint8_t>(value) & (value_count - 1))
            << (bit_width * (index % values_per_byte));
      }
    }

    TestLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(/*flags=*/0));
    ASSERT_NE(network, nullptr);
    auto* activation_input =
        network->addInput("activation", nvinfer1::DataType::kBF16,
                          nvinfer1::Dims{4, {1, 1, 1, kColumns}});
    ASSERT_NE(activation_input, nullptr);

    nvinfer1::Weights packed_weights{nvinfer1::DataType::kINT8, packed.data(),
                                     static_cast<int64_t>(packed.size())};
    auto* packed_layer = network->addConstant(
        nvinfer1::Dims{1, {static_cast<int32_t>(packed.size())}},
        packed_weights);
    ASSERT_NE(packed_layer, nullptr);
    nvinfer1::Weights scale_weights{nvinfer1::DataType::kBF16, scales.data(),
                                    static_cast<int64_t>(scales.size())};
    auto* scale_layer =
        network->addConstant(nvinfer1::Dims{1, {kRows}}, scale_weights);
    ASSERT_NE(scale_layer, nullptr);

    std::unique_ptr<nvinfer1::IPluginV3> plugin(
        CreateSubbyteGemvPlugin(bit_width, kRows, kColumns));
    ASSERT_NE(plugin, nullptr);
    nvinfer1::ITensor* inputs[] = {activation_input, packed_layer->getOutput(0),
                                   scale_layer->getOutput(0)};
    auto* plugin_layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network, inputs, std::size(inputs), *plugin);
    ASSERT_NE(plugin_layer, nullptr);
    auto* output = plugin_layer->getOutput(0);
    ASSERT_NE(output, nullptr);
    output->setName("output");
    network->markOutput(*output);

    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    ASSERT_NE(config, nullptr);
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(serialized, nullptr);
    network.reset();
    plugin.reset();

    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    ASSERT_NE(engine, nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);

    uint16_t* device_activation = nullptr;
    uint16_t* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_activation),
                         activation.size() * sizeof(uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_output),
                         kRows * sizeof(uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_activation, activation.data(),
                         activation.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress("activation", device_activation));
    ASSERT_TRUE(context->setTensorAddress("output", device_output));
    ASSERT_TRUE(context->enqueueV3(/*stream=*/nullptr));
    std::vector<uint16_t> actual(kRows);
    ASSERT_EQ(
        cudaMemcpy(actual.data(), device_output,
                   actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaFree(device_activation);
    cudaFree(device_output);

    for (int row = 0; row < kRows; ++row) {
      float accumulator = 0.0f;
      for (int column = 0; column < kColumns; ++column) {
        const size_t index = static_cast<size_t>(row) * kColumns + column;
        const uint8_t bits = (packed[index / values_per_byte] >>
                              (bit_width * (index % values_per_byte))) &
                             (value_count - 1);
        const int sign_bit = 1 << (bit_width - 1);
        const int weight = (bits ^ sign_bit) - sign_bit;
        accumulator += Bf16BitsToFloat(activation[column]) * weight;
      }
      const uint16_t expected =
          FloatToBf16Bits(accumulator * Bf16BitsToFloat(scales[row]));
      EXPECT_EQ(actual[row], expected)
          << "bit_width=" << bit_width << " row=" << row;
    }
  }
}

TEST(SubbyteGemvPluginTest, ManyRowsRunAsGemm) {
  // Prefill: 128 activation rows against INT4 weights, plain and with the
  // gate and up projections of a feed-forward block fused (GELU gate).
  struct Case {
    int32_t channels;  // output channels
    int32_t columns;
    int32_t gate;
  };
  constexpr int kActivationRows = 128;
  for (const Case& test_case : {Case{130, 256, 0}, Case{3840, 1024, 0},
                                Case{64, 512, 1}, Case{250, 256, 2}}) {
    SCOPED_TRACE(::testing::Message() << "channels=" << test_case.channels
                                      << " columns=" << test_case.columns
                                      << " gate=" << test_case.gate);
    const int columns = test_case.columns;
    const int weight_rows =
        test_case.gate != 0 ? 2 * test_case.channels : test_case.channels;
    std::vector<uint16_t> activation(kActivationRows * columns);
    std::vector<uint8_t> packed(weight_rows * columns / 2, 0);
    std::vector<uint16_t> scales(weight_rows);
    uint32_t state = 12345u + weight_rows + columns;
    const auto next = [&]() {
      state = state * 1664525u + 1013904223u;
      return state >> 8;
    };
    for (auto& value : activation) {
      value =
          FloatToBf16Bits(static_cast<float>(next() % 2048) / 512.0f - 2.0f);
    }
    for (auto& byte : packed) {
      byte = static_cast<uint8_t>(next());
    }
    for (auto& scale : scales) {
      scale =
          FloatToBf16Bits(0.004f + static_cast<float>(next() % 8) / 1024.0f);
    }

    const LiteRtNvidiaGemmShape shape = {kActivationRows, columns,
                                         test_case.channels, test_case.gate};
    ASSERT_EQ(LiteRtNvidiaSubbyteGemmWeightBytes(&shape), packed.size());

    TestLogger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger));
    ASSERT_NE(builder, nullptr);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(/*flags=*/0));
    ASSERT_NE(network, nullptr);
    auto* activation_input =
        network->addInput("activation", nvinfer1::DataType::kBF16,
                          nvinfer1::Dims{3, {1, kActivationRows, columns}});
    ASSERT_NE(activation_input, nullptr);
    nvinfer1::Weights packed_weights{nvinfer1::DataType::kINT8, packed.data(),
                                     static_cast<int64_t>(packed.size())};
    auto* packed_layer = network->addConstant(
        nvinfer1::Dims{1, {static_cast<int32_t>(packed.size())}},
        packed_weights);
    ASSERT_NE(packed_layer, nullptr);
    nvinfer1::Weights scale_weights{nvinfer1::DataType::kBF16, scales.data(),
                                    static_cast<int64_t>(scales.size())};
    auto* scale_layer =
        network->addConstant(nvinfer1::Dims{1, {weight_rows}}, scale_weights);
    ASSERT_NE(scale_layer, nullptr);
    std::unique_ptr<nvinfer1::IPluginV3> plugin(CreateSubbyteGemvPlugin(
        /*bit_width=*/4, weight_rows, columns, test_case.gate,
        /*gemm=*/true));
    ASSERT_NE(plugin, nullptr);
    nvinfer1::ITensor* inputs[] = {activation_input, packed_layer->getOutput(0),
                                   scale_layer->getOutput(0)};
    auto* plugin_layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network, inputs, std::size(inputs), *plugin);
    ASSERT_NE(plugin_layer, nullptr);
    auto* output = plugin_layer->getOutput(0);
    ASSERT_NE(output, nullptr);
    const auto output_dims = output->getDimensions();
    ASSERT_EQ(output_dims.nbDims, 3);
    EXPECT_EQ(output_dims.d[1], kActivationRows);
    EXPECT_EQ(output_dims.d[2], test_case.channels);
    output->setName("output");
    network->markOutput(*output);
    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());
    ASSERT_NE(config, nullptr);
    std::unique_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    ASSERT_NE(serialized, nullptr);
    network.reset();
    plugin.reset();

    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    ASSERT_NE(runtime, nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
    ASSERT_NE(engine, nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    ASSERT_NE(context, nullptr);
    uint16_t* device_activation = nullptr;
    uint16_t* device_output = nullptr;
    const size_t output_elements =
        static_cast<size_t>(kActivationRows) * test_case.channels;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_activation),
                         activation.size() * sizeof(uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_output),
                         output_elements * sizeof(uint16_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_activation, activation.data(),
                         activation.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress("activation", device_activation));
    ASSERT_TRUE(context->setTensorAddress("output", device_output));
    ASSERT_TRUE(context->enqueueV3(/*stream=*/nullptr));
    std::vector<uint16_t> actual(output_elements);
    ASSERT_EQ(
        cudaMemcpy(actual.data(), device_output,
                   actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        cudaSuccess);
    cudaFree(device_activation);
    cudaFree(device_output);

    const auto dot = [&](int row, int channel) {
      double sum = 0.0;
      for (int column = 0; column < columns; ++column) {
        const size_t index = static_cast<size_t>(channel) * columns + column;
        const int bits = (packed[index / 2] >> (4 * (index % 2))) & 15;
        sum += static_cast<double>(
                   Bf16BitsToFloat(activation[row * columns + column])) *
               ((bits ^ 8) - 8);
      }
      return sum * Bf16BitsToFloat(scales[channel]);
    };
    const auto gelu = [&](double x) {
      return test_case.gate == 1
                 ? 0.5 * x *
                       (1.0 + std::tanh(0.7978845608028654 *
                                        (x + 0.044715 * x * x * x)))
                 : 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
    };
    for (int row : {0, 1, 63, 127}) {
      double error = 0.0;
      double norm = 0.0;
      for (int channel = 0; channel < test_case.channels; ++channel) {
        const double expected = test_case.gate != 0
                                    ? gelu(dot(row, channel)) *
                                          dot(row, test_case.channels + channel)
                                    : dot(row, channel);
        const double value =
            Bf16BitsToFloat(actual[row * test_case.channels + channel]);
        error += (value - expected) * (value - expected);
        norm += expected * expected;
      }
      // BF16 rounding alone is about 1.7e-3 of the norm of a row.
      EXPECT_LE(std::sqrt(error), 3.0e-3 * std::sqrt(norm)) << "row=" << row;
    }
  }
}

TEST(SubbyteGemvPluginTest, ReadsWeightsAtAnOffsetInAHolder) {
  // Two projections read their packed weights from one INT64 holder constant:
  // the segment starts at the first granule boundary in the holder (here its
  // first byte, with a 16-byte granule), a GEMV at offset 128 and a GEMM at
  // the offset after it.
  constexpr int kColumns = 256;
  constexpr int kGemvRows = 9;
  constexpr int kGemmChannels = 130;
  constexpr int kActivationRows = 128;
  constexpr int64_t kGranule = 16;
  constexpr int64_t kGemvOffset = 128;
  const int64_t gemv_bytes = kGemvRows * kColumns / 2;
  const int64_t gemm_offset = (kGemvOffset + gemv_bytes + 127) / 128 * 128;
  const int64_t gemm_bytes = kGemmChannels * kColumns / 2;
  const int64_t segment_bytes = (gemm_offset + gemm_bytes + 127) / 128 * 128;
  std::vector<uint64_t> holder((segment_bytes + kGranule) / 8,
                               0xa5a5a5a5a5a5a5a5ull);
  auto* holder_bytes = reinterpret_cast<uint8_t*>(holder.data());
  uint32_t state = 77;
  const auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
  };
  for (int64_t i = 0; i < gemv_bytes; ++i) {
    holder_bytes[kGemvOffset + i] = static_cast<uint8_t>(next());
  }
  for (int64_t i = 0; i < gemm_bytes; ++i) {
    holder_bytes[gemm_offset + i] = static_cast<uint8_t>(next());
  }
  std::vector<uint16_t> gemv_scales(kGemvRows);
  std::vector<uint16_t> gemm_scales(kGemmChannels);
  for (auto& scale : gemv_scales) {
    scale = FloatToBf16Bits(0.125f * static_cast<float>(next() % 8 + 1));
  }
  for (auto& scale : gemm_scales) {
    scale = FloatToBf16Bits(0.004f + static_cast<float>(next() % 8) / 1024.0f);
  }
  std::vector<uint16_t> row(kColumns);
  std::vector<uint16_t> rows(static_cast<size_t>(kActivationRows) * kColumns);
  for (auto& value : row) {
    value = FloatToBf16Bits(static_cast<float>(next() % 23) / 8.0f - 1.375f);
  }
  for (auto& value : rows) {
    value = FloatToBf16Bits(static_cast<float>(next() % 2048) / 512.0f - 2.0f);
  }

  TestLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  ASSERT_NE(builder, nullptr);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(/*flags=*/0));
  ASSERT_NE(network, nullptr);
  auto* row_input = network->addInput("row", nvinfer1::DataType::kBF16,
                                      nvinfer1::Dims{2, {1, kColumns}});
  auto* rows_input =
      network->addInput("rows", nvinfer1::DataType::kBF16,
                        nvinfer1::Dims{2, {kActivationRows, kColumns}});
  ASSERT_NE(row_input, nullptr);
  ASSERT_NE(rows_input, nullptr);
  auto* holder_layer = network->addConstant(
      nvinfer1::Dims{1, {static_cast<int64_t>(holder.size())}},
      nvinfer1::Weights{nvinfer1::DataType::kINT64, holder.data(),
                        static_cast<int64_t>(holder.size())});
  ASSERT_NE(holder_layer, nullptr);
  auto* gemv_scale_layer =
      network->addConstant(nvinfer1::Dims{1, {kGemvRows}},
                           nvinfer1::Weights{nvinfer1::DataType::kBF16,
                                             gemv_scales.data(), kGemvRows});
  auto* gemm_scale_layer = network->addConstant(
      nvinfer1::Dims{1, {kGemmChannels}},
      nvinfer1::Weights{nvinfer1::DataType::kBF16, gemm_scales.data(),
                        kGemmChannels});
  ASSERT_NE(gemv_scale_layer, nullptr);
  ASSERT_NE(gemm_scale_layer, nullptr);
  std::unique_ptr<nvinfer1::IPluginV3> gemv(CreateSubbyteGemvPlugin(
      /*bit_width=*/4, kGemvRows, kColumns, /*gate=*/0, /*gemm=*/false,
      kGemvOffset, kGranule));
  std::unique_ptr<nvinfer1::IPluginV3> gemm(CreateSubbyteGemvPlugin(
      /*bit_width=*/4, kGemmChannels, kColumns, /*gate=*/0, /*gemm=*/true,
      gemm_offset, kGranule));
  ASSERT_NE(gemv, nullptr);
  ASSERT_NE(gemm, nullptr);
  nvinfer1::ITensor* gemv_inputs[] = {row_input, holder_layer->getOutput(0),
                                      gemv_scale_layer->getOutput(0)};
  nvinfer1::ITensor* gemm_inputs[] = {rows_input, holder_layer->getOutput(0),
                                      gemm_scale_layer->getOutput(0)};
  auto* gemv_layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *network, gemv_inputs, std::size(gemv_inputs), *gemv);
  auto* gemm_layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *network, gemm_inputs, std::size(gemm_inputs), *gemm);
  ASSERT_NE(gemv_layer, nullptr);
  ASSERT_NE(gemm_layer, nullptr);
  gemv_layer->getOutput(0)->setName("row_output");
  gemm_layer->getOutput(0)->setName("rows_output");
  network->markOutput(*gemv_layer->getOutput(0));
  network->markOutput(*gemm_layer->getOutput(0));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  ASSERT_NE(config, nullptr);
  std::unique_ptr<nvinfer1::IHostMemory> serialized(
      builder->buildSerializedNetwork(*network, *config));
  ASSERT_NE(serialized, nullptr);
  network.reset();
  gemv.reset();
  gemm.reset();

  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
  ASSERT_NE(engine, nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  uint16_t* device_row = nullptr;
  uint16_t* device_rows = nullptr;
  uint16_t* device_row_output = nullptr;
  uint16_t* device_rows_output = nullptr;
  const size_t rows_output_count =
      static_cast<size_t>(kActivationRows) * kGemmChannels;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_row), row.size() * 2),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_rows), rows.size() * 2),
            cudaSuccess);
  ASSERT_EQ(
      cudaMalloc(reinterpret_cast<void**>(&device_row_output), kGemvRows * 2),
      cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_rows_output),
                       rows_output_count * 2),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_row, row.data(), row.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_rows, rows.data(), rows.size() * 2,
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_TRUE(context->setTensorAddress("row", device_row));
  ASSERT_TRUE(context->setTensorAddress("rows", device_rows));
  ASSERT_TRUE(context->setTensorAddress("row_output", device_row_output));
  ASSERT_TRUE(context->setTensorAddress("rows_output", device_rows_output));
  ASSERT_TRUE(context->enqueueV3(/*stream=*/nullptr));
  std::vector<uint16_t> row_output(kGemvRows);
  std::vector<uint16_t> rows_output(rows_output_count);
  ASSERT_EQ(cudaMemcpy(row_output.data(), device_row_output, kGemvRows * 2,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(rows_output.data(), device_rows_output,
                       rows_output_count * 2, cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(device_row);
  cudaFree(device_rows);
  cudaFree(device_row_output);
  cudaFree(device_rows_output);

  const auto weight = [&](int64_t offset, int channel, int column) {
    const uint8_t byte =
        holder_bytes[offset +
                     (static_cast<int64_t>(channel) * kColumns + column) / 2];
    const int nibble = (byte >> ((column & 1) * 4)) & 15;
    return (nibble ^ 8) - 8;
  };
  for (int channel = 0; channel < kGemvRows; ++channel) {
    float accumulator = 0.0f;
    for (int column = 0; column < kColumns; ++column) {
      accumulator +=
          Bf16BitsToFloat(row[column]) * weight(kGemvOffset, channel, column);
    }
    EXPECT_EQ(
        row_output[channel],
        FloatToBf16Bits(accumulator * Bf16BitsToFloat(gemv_scales[channel])))
        << "channel=" << channel;
  }
  for (int m : {0, 77, kActivationRows - 1}) {
    double error = 0.0;
    double norm = 0.0;
    for (int channel = 0; channel < kGemmChannels; ++channel) {
      double sum = 0.0;
      for (int column = 0; column < kColumns; ++column) {
        sum += static_cast<double>(Bf16BitsToFloat(
                   rows[static_cast<size_t>(m) * kColumns + column])) *
               weight(gemm_offset, channel, column);
      }
      const double expected = sum * Bf16BitsToFloat(gemm_scales[channel]);
      const double value = Bf16BitsToFloat(
          rows_output[static_cast<size_t>(m) * kGemmChannels + channel]);
      error += (value - expected) * (value - expected);
      norm += expected * expected;
    }
    EXPECT_LE(std::sqrt(error), 3.0e-3 * std::sqrt(norm)) << "row=" << m;
  }
}

TEST(SubbyteGemvPluginTest, RejectsInvalidHolders) {
  const auto create = [](int64_t weight_offset, int64_t holder_granule) {
    return std::unique_ptr<nvinfer1::IPluginV3>(CreateSubbyteGemvPlugin(
        /*bit_width=*/4, /*rows=*/128, /*columns=*/256, /*gate=*/0,
        /*gemm=*/false, weight_offset, holder_granule));
  };
  EXPECT_NE(create(0, 0), nullptr);
  EXPECT_NE(create(0, 2 << 20), nullptr);
  EXPECT_NE(create(128, 2 << 20), nullptr);
  // An offset needs a holder; the granule is a power of two of at least 16
  // bytes and the offset keeps the kernels' 16-byte alignment.
  EXPECT_EQ(create(128, 0), nullptr);
  EXPECT_EQ(create(0, 3 << 20), nullptr);
  EXPECT_EQ(create(0, 8), nullptr);
  EXPECT_EQ(create(-128, 2 << 20), nullptr);
  EXPECT_EQ(create(8, 2 << 20), nullptr);
}

TEST(SubbyteGemvPluginTest, RejectsInvalidGatesAndGemmShapes) {
  const auto create = [](int32_t bit_width, int32_t rows, int32_t columns,
                         int32_t gate, bool gemm) {
    return std::unique_ptr<nvinfer1::IPluginV3>(
        CreateSubbyteGemvPlugin(bit_width, rows, columns, gate, gemm));
  };
  EXPECT_NE(create(4, 128, 256, /*gate=*/1, /*gemm=*/true), nullptr);
  EXPECT_NE(create(4, 130, 256, /*gate=*/0, /*gemm=*/true), nullptr);
  EXPECT_EQ(create(4, 128, 256, /*gate=*/3, /*gemm=*/true), nullptr);
  EXPECT_EQ(create(4, 129, 256, /*gate=*/1, /*gemm=*/true), nullptr);
  // The GEMM takes INT4 weights and input dims in multiples of 128, and a
  // gate needs the GEMM.
  EXPECT_EQ(create(2, 128, 256, /*gate=*/1, /*gemm=*/true), nullptr);
  EXPECT_EQ(create(2, 128, 256, /*gate=*/0, /*gemm=*/true), nullptr);
  EXPECT_EQ(create(4, 128, 272, /*gate=*/0, /*gemm=*/true), nullptr);
  EXPECT_EQ(create(4, 128, 256, /*gate=*/1, /*gemm=*/false), nullptr);
}

}  // namespace
}  // namespace litert::nvidia
