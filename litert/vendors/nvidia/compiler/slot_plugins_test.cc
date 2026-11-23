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

#include "litert/vendors/nvidia/compiler/slot_plugins.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"

namespace litert::nvidia {
namespace {

class TestLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[SlotPluginsTest][%d] %s\n",
                   static_cast<int>(severity), message);
    }
  }
};

struct CudaDeleter {
  void operator()(void* p) const { cudaFree(p); }
};
using Device = std::unique_ptr<void, CudaDeleter>;

Device Upload(const std::vector<uint8_t>& host) {
  void* p = nullptr;
  if (cudaMalloc(&p, host.size()) != cudaSuccess) return nullptr;
  cudaMemcpy(p, host.data(), host.size(), cudaMemcpyHostToDevice);
  return Device(p);
}

// Bytes that are finite FP16 or BF16 values of moderate size when read two at
// a time (the high byte keeps the exponent in range), or any INT8 values.
std::vector<uint8_t> Bytes(size_t count, uint32_t seed, bool floats) {
  std::vector<uint8_t> bytes(count);
  uint32_t state = seed * 2654435761u + 12345u;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    const uint8_t byte = static_cast<uint8_t>(state >> 24);
    bytes[i] = floats && i % 2 == 1 ? static_cast<uint8_t>(0x3c + (byte & 0x83))
                                    : byte;
  }
  return bytes;
}

// A decode attention layer of a network: q [1, H, rows, D], caches of
// `cache_len` keys, mask [1, 1, 1, cache_len], and a window's start.
struct Layer {
  TestLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::INetworkDefinition> network;
  std::unique_ptr<nvinfer1::IPluginV3> plugin;
  nvinfer1::IPluginV3Layer* layer = nullptr;
  std::vector<nvinfer1::Dims> dims;
  std::vector<nvinfer1::DataType> types;
};

std::unique_ptr<Layer> MakeLayer(nvinfer1::IPluginV3* plugin, int heads,
                                 int rows, int depth, int cache_len,
                                 int int8_cache, int window) {
  auto result = std::make_unique<Layer>();
  result->plugin.reset(plugin);
  if (plugin == nullptr) return nullptr;
  result->builder.reset(nvinfer1::createInferBuilder(result->logger));
  if (!result->builder) return nullptr;
  result->network.reset(result->builder->createNetworkV2(0));
  if (!result->network) return nullptr;
  const nvinfer1::DataType cache_type =
      int8_cache != 0 ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kHALF;
  result->dims = {nvinfer1::Dims4{1, heads, rows, depth},
                  nvinfer1::Dims4{1, heads, cache_len, depth},
                  int8_cache == 2 ? nvinfer1::Dims4{1, heads, depth, cache_len}
                                  : nvinfer1::Dims4{1, heads, cache_len, depth},
                  nvinfer1::Dims4{1, 1, 1, cache_len}};
  result->types = {nvinfer1::DataType::kBF16, cache_type, cache_type,
                   nvinfer1::DataType::kBOOL};
  if (window != 0) {
    nvinfer1::Dims one{};
    one.nbDims = 1;
    one.d[0] = 1;
    result->dims.push_back(one);
    result->types.push_back(nvinfer1::DataType::kINT32);
  }
  std::vector<nvinfer1::ITensor*> inputs;
  for (size_t i = 0; i < result->dims.size(); ++i) {
    inputs.push_back(
        result->network->addInput(("input" + std::to_string(i)).c_str(),
                                  result->types[i], result->dims[i]));
    if (inputs.back() == nullptr) return nullptr;
  }
  result->layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
      *result->network, inputs.data(), static_cast<int32_t>(inputs.size()),
      *result->plugin);
  return result->layer != nullptr ? std::move(result) : nullptr;
}

nvinfer1::IPluginV3OneRuntime* Runtime(nvinfer1::IPluginV3* plugin) {
  return static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
}

// The twin's outputs for `slots` slots against the single-sequence plugin's
// for each slot alone, through the plugins' own enqueue.
void CheckTwin(int slots, int heads, int rows, int depth, int cache_len,
               int int8_cache, int window) {
  SCOPED_TRACE(::testing::Message()
               << slots << " slots, heads " << heads << ", rows " << rows
               << ", depth " << depth << ", " << cache_len << " keys, int8 "
               << int8_cache << ", window " << window);
  const LiteRtNvidiaInt8CacheParams int8 = {0.0125f, -3, 0.02f, 5,
                                            int8_cache == 2};
  auto layer = MakeLayer(
      CreateDecodeAttentionPlugin(-10000.0f, /*new_keys=*/false,
                                  int8_cache != 0 ? &int8 : nullptr, window),
      heads, rows, depth, cache_len, int8_cache, window);
  ASSERT_NE(layer, nullptr);
  std::unique_ptr<nvinfer1::IPluginV3> twin(
      CreateSlotsTwin(*layer->layer, slots));
  ASSERT_NE(twin, nullptr);
  // The twin as an engine holds it: made again from its serialized fields.
  auto* twin_core = static_cast<nvinfer1::IPluginV3OneCore*>(
      twin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE));
  ASSERT_NE(twin_core, nullptr);
  auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(
      getPluginRegistry()->getCreator(twin_core->getPluginName(),
                                      twin_core->getPluginVersion(),
                                      twin_core->getPluginNamespace()));
  ASSERT_NE(creator, nullptr);
  std::unique_ptr<nvinfer1::IPluginV3> restored(
      creator->createPlugin("twin", Runtime(twin.get())->getFieldsToSerialize(),
                            nvinfer1::TensorRTPhase::kRUNTIME));
  ASSERT_NE(restored, nullptr);

  const int n = static_cast<int>(layer->dims.size());
  std::vector<nvinfer1::PluginTensorDesc> single_in(n), twin_in(slots * n),
      twin_out(slots);
  for (int i = 0; i < n; ++i) {
    single_in[i].dims = layer->dims[i];
    single_in[i].type = layer->types[i];
    single_in[i].format = nvinfer1::TensorFormat::kLINEAR;
  }
  for (int s = 0; s < slots; ++s) {
    for (int i = 0; i < n; ++i) twin_in[s * n + i] = single_in[i];
    twin_out[s] = single_in[0];
  }
  auto* single_runtime = Runtime(layer->plugin.get());
  auto* twin_runtime = Runtime(restored.get());
  ASSERT_EQ(
      single_runtime->onShapeChange(single_in.data(), n, &single_in[0], 1), 0);
  ASSERT_EQ(twin_runtime->onShapeChange(twin_in.data(), slots * n,
                                        twin_out.data(), slots),
            0);
  const int read = window != 0 ? window : cache_len;
  const size_t single_ws =
      LiteRtNvidiaDecodeAttentionWorkspaceBytes(heads, rows, read, depth);
  auto* twin_build = static_cast<nvinfer1::IPluginV3OneBuild*>(
      restored->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
  std::vector<nvinfer1::DynamicPluginTensorDesc> dynamic(slots * n),
      dynamic_out(slots);
  for (int i = 0; i < slots * n; ++i) {
    dynamic[i].desc = twin_in[i];
    dynamic[i].min = dynamic[i].opt = dynamic[i].max = twin_in[i].dims;
  }
  for (int s = 0; s < slots; ++s) dynamic_out[s].desc = twin_out[s];
  ASSERT_EQ(twin_build->getWorkspaceSize(dynamic.data(), slots * n,
                                         dynamic_out.data(), slots),
            slots * single_ws);

  const size_t q_bytes = static_cast<size_t>(heads) * rows * depth * 2;
  const size_t cache_bytes = static_cast<size_t>(heads) * cache_len * depth *
                             (int8_cache != 0 ? 1 : 2);
  std::vector<Device> buffers;
  std::vector<const void*> twin_inputs;
  std::vector<void*> twin_outputs;
  std::vector<std::vector<uint8_t>> expected(slots);
  Device single_workspace = Upload(std::vector<uint8_t>(single_ws));
  Device twin_workspace = Upload(std::vector<uint8_t>(slots * single_ws));
  ASSERT_TRUE(single_workspace && twin_workspace);
  for (int s = 0; s < slots; ++s) {
    // Slot s sees the first keys up to its own position.
    const int visible = cache_len * (s + 1) / slots;
    std::vector<uint8_t> mask(cache_len, 0);
    std::fill_n(mask.begin(), visible, 1);
    const int32_t start =
        window != 0 ? std::max(0, visible - window) / 16 * 16 : 0;
    std::vector<uint8_t> start_bytes(sizeof(start));
    std::memcpy(start_bytes.data(), &start, sizeof(start));
    std::vector<const void*> slot_inputs;
    for (const auto& host :
         {Bytes(q_bytes, 10 * s + 1, true),
          Bytes(cache_bytes, 10 * s + 2, int8_cache == 0),
          Bytes(cache_bytes, 10 * s + 3, int8_cache == 0), mask, start_bytes}) {
      if (static_cast<int>(slot_inputs.size()) == n) break;
      buffers.push_back(Upload(host));
      ASSERT_NE(buffers.back(), nullptr);
      slot_inputs.push_back(buffers.back().get());
    }
    buffers.push_back(Upload(std::vector<uint8_t>(q_bytes)));
    void* single_out = buffers.back().get();
    buffers.push_back(Upload(std::vector<uint8_t>(q_bytes)));
    twin_outputs.push_back(buffers.back().get());
    twin_inputs.insert(twin_inputs.end(), slot_inputs.begin(),
                       slot_inputs.end());
    ASSERT_EQ(single_runtime->enqueue(single_in.data(), &single_in[0],
                                      slot_inputs.data(), &single_out,
                                      single_workspace.get(), nullptr),
              0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expected[s].resize(q_bytes);
    ASSERT_EQ(cudaMemcpy(expected[s].data(), single_out, q_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  ASSERT_EQ(
      twin_runtime->enqueue(twin_in.data(), twin_out.data(), twin_inputs.data(),
                            twin_outputs.data(), twin_workspace.get(), nullptr),
      0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> actual(q_bytes);
  for (int s = 0; s < slots; ++s) {
    ASSERT_EQ(cudaMemcpy(actual.data(), twin_outputs[s], q_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_NE(std::count(actual.begin(), actual.end(), 0),
              static_cast<ptrdiff_t>(actual.size()));
    EXPECT_EQ(std::memcmp(actual.data(), expected[s].data(), q_bytes), 0)
        << "slot " << s;
  }
}

TEST(SlotPluginsTest, DecodeAttentionTwinMatchesTheLayerOnEachSlot) {
  // Gemma 4 26B-A4B's local and global layers, E4B's local layer through a
  // window and its global layer.
  for (const int slots : {2, 3, 8}) {
    CheckTwin(slots, 8, 2, 256, 1024, /*int8_cache=*/0, /*window=*/0);
    CheckTwin(slots, 2, 8, 512, 2048, 0, 0);
    CheckTwin(slots, 2, 4, 256, 2048, 2, 512);
    CheckTwin(slots, 1, 8, 512, 2048, 2, 0);
    CheckTwin(slots, 2, 4, 256, 1024, 1, 0);
  }
}

TEST(SlotPluginsTest, DeclinesLayersWithoutATwin) {
  // One slot is the layer itself; 16 rows over a cache of depth 512 run on
  // the tensor-core kernel; new keys and row tokens are prefill and verify
  // forms.
  auto decode =
      MakeLayer(CreateDecodeAttentionPlugin(-10000.0f), 8, 2, 256, 1024, 0, 0);
  ASSERT_NE(decode, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*decode->layer, 1), nullptr);
  EXPECT_EQ(CreateSlotsTwin(*decode->layer, 9), nullptr);
  auto streaming =
      MakeLayer(CreateDecodeAttentionPlugin(-10000.0f), 1, 16, 512, 2048, 0, 0);
  ASSERT_NE(streaming, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*streaming->layer, 2), nullptr);
  auto tokens =
      MakeLayer(CreateDecodeAttentionPlugin(-10000.0f, false, nullptr, 0,
                                            /*row_tokens=*/4),
                2, 8, 512, 2048, 0, 0);
  ASSERT_NE(tokens, nullptr);
  EXPECT_EQ(CreateSlotsTwin(*tokens->layer, 2), nullptr);
}

}  // namespace
}  // namespace litert::nvidia
