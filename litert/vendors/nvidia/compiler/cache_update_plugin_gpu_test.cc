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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/cache_update_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "NvInfer.h"

namespace litert::nvidia {
namespace {

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (data_ != nullptr) cudaFree(data_);
  }
  cudaError_t Allocate(size_t bytes) { return cudaMalloc(&data_, bytes); }
  void* get() const { return data_; }

 private:
  void* data_ = nullptr;
};

class TestStream {
 public:
  ~TestStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }
  cudaError_t Create() { return cudaStreamCreate(&stream_); }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[CacheUpdatePatchTest][%d] %s\n",
                   static_cast<int>(severity), message);
    }
    if (severity <= Severity::kERROR) ADD_FAILURE() << message;
  }
};

// Positive normal HALF values are sufficient here. Integer-valued inputs make
// every old-cache reduction exact in FLOAT, including the 1152-row case.
float SumHalfValues(const std::vector<uint16_t>& values) {
  float sum = 0.0f;
  for (uint16_t bits : values) {
    sum += std::ldexp(static_cast<float>(1024 + (bits & 1023)),
                      static_cast<int>((bits >> 10) & 31) - 25);
  }
  return sum;
}

void CheckSerializedNativeWriter(bool ring, bool transposed, int capacity,
                                 int rows) {
  SCOPED_TRACE(::testing::Message()
               << "ring=" << ring << " transposed=" << transposed
               << " capacity=" << capacity << " rows=" << rows);
  constexpr int kHeads = 2;
  constexpr int kDepth = 16;
  constexpr std::array<const char*, 5> kInputs = {
      "cache_k", "cache_v", "update_k", "update_v", "params"};
  constexpr std::array<const char*, 2> kOutputs = {"updated_k", "updated_v"};
  constexpr std::array<const char*, 2> kOldSums = {"old_k_sum", "old_v_sum"};
  const int patch_rows = ring ? capacity : rows;
  TestLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger));
  ASSERT_NE(builder, nullptr);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0));
  ASSERT_NE(network, nullptr);
  const nvinfer1::Dims4 cache_dims{1, kHeads, capacity, kDepth};
  const nvinfer1::Dims4 value_dims =
      transposed ? cache_dims
                 : nvinfer1::Dims4{1, kHeads * kDepth, capacity, 1};
  const std::array<nvinfer1::Dims4, 5> dimensions = {
      cache_dims, value_dims, nvinfer1::Dims4{1, kHeads, rows, kDepth},
      nvinfer1::Dims4{1, kHeads, rows, kDepth}, nvinfer1::Dims4{1, 1, 1, 7}};
  std::array<nvinfer1::ITensor*, 5> inputs{};
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = network->addInput(
        kInputs[i],
        i == 4 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kHALF,
        dimensions[i]);
    ASSERT_NE(inputs[i], nullptr);
  }

  // These branches observe the original cache. Pass both complete results as
  // opaque dependencies: the native writer alone cannot enforce old-reader
  // ordering across the patch plugin boundary.
  std::vector<nvinfer1::ITensor*> plugin_inputs(inputs.begin(), inputs.end());
  for (int i = 0; i < 2; ++i) {
    auto* cast = network->addCast(*inputs[i], nvinfer1::DataType::kFLOAT);
    ASSERT_NE(cast, nullptr);
    ASSERT_NE(cast->getOutput(0), nullptr);
    auto* sum = network->addReduce(*cast->getOutput(0),
                                   nvinfer1::ReduceOperation::kSUM, 0xf, false);
    ASSERT_NE(sum, nullptr);
    ASSERT_NE(sum->getOutput(0), nullptr);
    sum->getOutput(0)->setName(kOldSums[i]);
    network->markOutput(*sum->getOutput(0));
    plugin_inputs.push_back(sum->getOutput(0));
  }

  std::unique_ptr<nvinfer1::IPluginV3> plugin(
      CreateCacheUpdatePlugin(ring, transposed));
  ASSERT_NE(plugin, nullptr);
  auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *network, plugin_inputs.data(), plugin_inputs.size(), *plugin);
  ASSERT_NE(layer, nullptr);
  ASSERT_EQ(layer->getNbOutputs(), 2);
  ASSERT_NE(layer->getOutput(0), nullptr);
  ASSERT_NE(layer->getOutput(1), nullptr);
  EXPECT_EQ(layer->getOutput(0)->getDimensions().d[2], patch_rows);
  EXPECT_EQ(layer->getOutput(1)->getDimensions().d[2], patch_rows);

  // Only the native layer aliases network I/O. For a ring it commits the
  // complete prepared ring; the linear case commits a bounded U-row window.
  // Both indices are execution tensors, with no shape-value profile required.
  const std::array<int32_t, 2> constants = {0, capacity - rows};
  auto* zero = network->addConstant(
      nvinfer1::Dims{1, {1}},
      nvinfer1::Weights{nvinfer1::DataType::kINT32, constants.data(), 1});
  ASSERT_NE(zero, nullptr);
  nvinfer1::ITensor* anchor = zero->getOutput(0);
  ASSERT_NE(anchor, nullptr);
  if (!ring) {
    auto* flat = network->addShuffle(*inputs[4]);
    ASSERT_NE(flat, nullptr);
    flat->setReshapeDimensions(nvinfer1::Dims{1, {7}});
    ASSERT_NE(flat->getOutput(0), nullptr);
    auto* write =
        network->addSlice(*flat->getOutput(0), nvinfer1::Dims{1, {0}},
                          nvinfer1::Dims{1, {1}}, nvinfer1::Dims{1, {1}});
    ASSERT_NE(write, nullptr);
    ASSERT_NE(write->getOutput(0), nullptr);
    auto* upper = network->addConstant(
        nvinfer1::Dims{1, {1}},
        nvinfer1::Weights{nvinfer1::DataType::kINT32, constants.data() + 1, 1});
    ASSERT_NE(upper, nullptr);
    ASSERT_NE(upper->getOutput(0), nullptr);
    auto* lower_clamp = network->addElementWise(
        *write->getOutput(0), *anchor, nvinfer1::ElementWiseOperation::kMAX);
    ASSERT_NE(lower_clamp, nullptr);
    ASSERT_NE(lower_clamp->getOutput(0), nullptr);
    auto* upper_clamp = network->addElementWise(
        *lower_clamp->getOutput(0), *upper->getOutput(0),
        nvinfer1::ElementWiseOperation::kMIN);
    ASSERT_NE(upper_clamp, nullptr);
    anchor = upper_clamp->getOutput(0);
    ASSERT_NE(anchor, nullptr);
  }
  for (int i = 0; i < 2; ++i) {
    auto* writer =
        network->addKVCacheUpdate(*inputs[i], *layer->getOutput(i), *anchor,
                                  nvinfer1::KVCacheMode::kLINEAR);
    ASSERT_NE(writer, nullptr);
    ASSERT_NE(writer->getOutput(0), nullptr);
    writer->getOutput(0)->setName(kOutputs[i]);
    network->markOutput(*writer->getOutput(0));
  }
  EXPECT_FALSE(inputs[4]->isShapeTensor());
  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig());
  ASSERT_NE(config, nullptr);
  config->setMaxAuxStreams(2);
  // Deliberately do not enable the experimental aliased-plugin-I/O feature.
  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*network, *config));
  ASSERT_NE(plan, nullptr);
  network.reset();
  plugin.reset();
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  ASSERT_NE(engine, nullptr);
  for (int i = 0; i < 2; ++i) {
    ASSERT_STREQ(engine->getAliasedInputTensor(kOutputs[i]), kInputs[i]);
    EXPECT_EQ(engine->getAliasedInputTensor(kOldSums[i]), nullptr);
  }
  EXPECT_FALSE(engine->isShapeInferenceIO("params"));
  ASSERT_EQ(engine->getTensorLocation("params"),
            nvinfer1::TensorLocation::kDEVICE);
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  ASSERT_NE(context, nullptr);
  TestStream stream;
  ASSERT_EQ(stream.Create(), cudaSuccess);

  const size_t cache_count = kHeads * capacity * kDepth;
  const size_t update_count = kHeads * rows * kDepth;
  const std::array<size_t, 5> bytes = {
      cache_count * sizeof(uint16_t), cache_count * sizeof(uint16_t),
      update_count * sizeof(uint16_t), update_count * sizeof(uint16_t),
      7 * sizeof(int32_t)};
  std::array<DeviceAllocation, 5> buffers;
  std::array<DeviceAllocation, 2> old_sums;
  for (size_t i = 0; i < buffers.size(); ++i) {
    ASSERT_EQ(buffers[i].Allocate(bytes[i]), cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress(kInputs[i], buffers[i].get()));
  }
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(context->setTensorAddress(kOutputs[i], buffers[i].get()));
    ASSERT_EQ(old_sums[i].Allocate(sizeof(float)), cudaSuccess);
    ASSERT_TRUE(context->setTensorAddress(kOldSums[i], old_sums[i].get()));
  }

  constexpr std::array<uint16_t, 4> kInitialBits = {0x3c00, 0x4000, 0x4200,
                                                    0x4400};  // HALF 1,2,3,4.
  constexpr std::array<uint16_t, 4> kNewKeyBits = {0x4500, 0x4600, 0x4700,
                                                   0x4800};  // HALF 5,6,7,8.
  constexpr std::array<uint16_t, 4> kNewValueBits = {
      0x4880, 0x4900, 0x4980, 0x4a00};  // HALF 9,10,11,12.
  std::vector<uint16_t> expected_k(cache_count), expected_v(cache_count);
  for (size_t i = 0; i < cache_count; ++i) {
    expected_k[i] = kInitialBits[i % kInitialBits.size()];
    expected_v[i] = kInitialBits[(i + 2) % kInitialBits.size()];
  }
  ASSERT_EQ(cudaMemcpy(buffers[0].get(), expected_k.data(), bytes[0],
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(buffers[1].get(), expected_v.data(), bytes[1],
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  const std::array<std::array<int32_t, 2>, 12> cases = {
      {{0, rows - 1},
       {rows - 1, rows},
       {capacity - 2, 3},
       {capacity + 1, 2},
       {capacity - 1, 0},
       {-1, 2},
       {0, rows + 1},
       {0, -1},
       {capacity - 1, 1},
       {0, rows},
       {std::numeric_limits<int32_t>::max(), 2},
       {std::numeric_limits<int32_t>::max() - 1, rows}}};
  int invocation = 0;
  for (const auto& [write, valid] : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "invocation=" << invocation << " write=" << write
                 << " valid=" << valid);
    std::vector<uint16_t> update_k(update_count), update_v(update_count);
    for (size_t i = 0; i < update_count; ++i) {
      const int row = (i / kDepth) % rows;
      // Padded rows contain max-HALF poison: copying even one must fail the
      // complete-cache comparison. Change valid values between invocations.
      update_k[i] = row >= valid ? 0x7bff : kNewKeyBits[(i + invocation) % 4];
      update_v[i] = row >= valid ? 0x7bff : kNewValueBits[(i + invocation) % 4];
    }
    const std::array<float, 2> expected_old_sums = {SumHalfValues(expected_k),
                                                    SumHalfValues(expected_v)};
    if (write >= 0 && valid >= 0 && valid <= rows) {
      for (int h = 0; h < kHeads; ++h) {
        for (int row = 0; row < valid; ++row) {
          int64_t destination = static_cast<int64_t>(write) + row;
          if (ring) destination %= capacity;
          if (destination >= capacity) continue;
          for (int c = 0; c < kDepth; ++c) {
            const size_t source = (h * rows + row) * kDepth + c;
            const size_t k = (h * capacity + destination) * kDepth + c;
            const size_t v =
                transposed ? k : (h * kDepth + c) * capacity + destination;
            expected_k[k] = update_k[source];
            expected_v[v] = update_v[source];
          }
        }
      }
    }
    const std::array<int32_t, 7> params = {write, 0, 0, valid, 0, 0, 0};
    ASSERT_EQ(cudaMemcpy(buffers[2].get(), update_k.data(), bytes[2],
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(buffers[3].get(), update_v.data(), bytes[3],
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(buffers[4].get(), params.data(), bytes[4],
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_TRUE(context->enqueueV3(stream.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    for (int i = 0; i < 2; ++i) {
      float actual_sum = 0.0f;
      ASSERT_EQ(cudaMemcpy(&actual_sum, old_sums[i].get(), sizeof(float),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_FLOAT_EQ(actual_sum, expected_old_sums[i]);
    }
    std::vector<uint16_t> actual_k(cache_count), actual_v(cache_count);
    ASSERT_EQ(cudaMemcpy(actual_k.data(), buffers[0].get(), bytes[0],
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_v.data(), buffers[1].get(), bytes[1],
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual_k, expected_k);
    EXPECT_EQ(actual_v, expected_v);
    ++invocation;
  }
}

TEST(CacheUpdatePluginGpuTest, NativeRingWriterPreservesPaddingAndOldReaders) {
  for (bool transposed : {false, true}) {
    CheckSerializedNativeWriter(/*ring=*/true, transposed,
                                /*capacity=*/8, /*rows=*/4);
    CheckSerializedNativeWriter(/*ring=*/true, transposed,
                                /*capacity=*/1152, /*rows=*/1024);
  }
}

TEST(CacheUpdatePluginGpuTest,
     NativeLinearWriterPreservesPaddingAndOldReaders) {
  for (bool transposed : {false, true}) {
    CheckSerializedNativeWriter(/*ring=*/false, transposed,
                                /*capacity=*/8, /*rows=*/4);
    CheckSerializedNativeWriter(/*ring=*/false, transposed,
                                /*capacity=*/1152, /*rows=*/1024);
  }
}

}  // namespace
}  // namespace litert::nvidia
