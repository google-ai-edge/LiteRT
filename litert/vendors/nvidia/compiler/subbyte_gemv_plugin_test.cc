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
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "litert/vendors/nvidia/tensorrt_rtx/include/NvInfer.h"

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
  constexpr int32_t kRows = 8;
  constexpr int32_t kColumns = 64;
  for (const int bit_width : {2, 4}) {
    const int values_per_byte = 8 / bit_width;
    const int value_count = 1 << bit_width;
    std::vector<uint16_t> activation(kColumns);
    std::vector<uint8_t> packed(kRows * kColumns / values_per_byte, 0);
    std::array<uint16_t, kRows> scales;
    for (int column = 0; column < kColumns; ++column) {
      activation[column] =
          FloatToBf16Bits(static_cast<float>((column * 7) % 23 - 11) / 8.0f);
    }
    for (int row = 0; row < kRows; ++row) {
      scales[row] = FloatToBf16Bits(0.125f * static_cast<float>(row + 1));
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
                                    scales.size()};
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
    std::array<uint16_t, kRows> actual;
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

}  // namespace
}  // namespace litert::nvidia
